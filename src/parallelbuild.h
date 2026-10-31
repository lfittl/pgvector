#ifndef PARALLELBUILD_H
#define PARALLELBUILD_H

/*
 * Parallel index build infrastructure
 *
 * Postgres 20+ provides shared helpers for the parts of a parallel index build
 * that are common to all access methods (access/parallel_index_build.h), as
 * well as for passing the query text to workers (access/parallel.h).  For
 * earlier versions, provide equivalent definitions here, so HNSW and IVFFlat
 * can use a single code path.
 */

#include "postgres.h"

#include "access/parallel.h"

#if PG_VERSION_NUM >= 200000

#include "access/parallel_index_build.h"

#else

#include "access/genam.h"
#include "access/relscan.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "catalog/index.h"
#include "executor/instrument.h"
#include "pgstat.h"
#include "storage/condition_variable.h"
#include "storage/spin.h"
#include "tcop/tcopprot.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/wait_event.h"

#if PG_VERSION_NUM < 190000
#include "storage/shmem.h"		/* for add_size() in some versions */
#endif

typedef struct ParallelIndexBuildShared
{
	/* Immutable state, set by the leader before launching workers */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;
	int			scantuplesortstates;

	/* Query ID, for report in worker processes */
#if PG_VERSION_NUM >= 180000
	int64		queryid;
#else
	uint64		queryid;
#endif

	/* Worker progress */
	ConditionVariable workersdonecv;

	/* Mutex for mutable state */
	slock_t		mutex;

	/* Mutable state */
	int			nparticipantsdone;
	double		reltuples;
	double		indtuples;
	bool		havedead;
	bool		brokenhotchain;
}			ParallelIndexBuildShared;

static inline ParallelContext *
ParallelIndexBuildCreateContext(const char *library_name,
								const char *worker_function, int nworkers)
{
	Assert(nworkers > 0);

	EnterParallelMode();

	return CreateParallelContext(library_name, worker_function, nworkers);
}

static inline Snapshot
ParallelIndexBuildGetSnapshot(bool isconcurrent)
{
	if (!isconcurrent)
		return SnapshotAny;

	return RegisterSnapshot(GetTransactionSnapshot());
}

static inline void
ParallelIndexBuildEstimate(ParallelContext *pcxt, Relation heap,
						   Snapshot snapshot)
{
	/* Space for the parallel table scan descriptor */
	shm_toc_estimate_chunk(&pcxt->estimator,
						   table_parallelscan_estimate(heap, snapshot));
	shm_toc_estimate_keys(&pcxt->estimator, 1);
}

static inline void
ParallelIndexBuildInitShared(ParallelContext *pcxt,
							 ParallelIndexBuildShared * shared,
							 uint64 pscan_key,
							 Relation heap, Relation index,
							 bool isconcurrent, int scantuplesortstates,
							 Snapshot snapshot)
{
	ParallelTableScanDesc pscan;

	/* Initialize immutable state */
	shared->heaprelid = RelationGetRelid(heap);
	shared->indexrelid = RelationGetRelid(index);
	shared->isconcurrent = isconcurrent;
	shared->scantuplesortstates = scantuplesortstates;
#if PG_VERSION_NUM >= 140000
	shared->queryid = pgstat_get_my_query_id();
#else
	shared->queryid = 0;
#endif
	ConditionVariableInit(&shared->workersdonecv);
	SpinLockInit(&shared->mutex);

	/* Initialize mutable state */
	shared->nparticipantsdone = 0;
	shared->reltuples = 0.0;
	shared->indtuples = 0.0;
	shared->havedead = false;
	shared->brokenhotchain = false;

	/* Set up the parallel table scan, in its own TOC entry */
	pscan = (ParallelTableScanDesc)
		shm_toc_allocate(pcxt->toc,
						 table_parallelscan_estimate(heap, snapshot));
	table_parallelscan_initialize(heap, pscan, snapshot);
	shm_toc_insert(pcxt->toc, pscan_key, pscan);
}

static inline void
ParallelIndexBuildWaitForWorkers(ParallelIndexBuildShared * shared,
								 int nparticipants, IndexInfo *indexInfo,
								 double *reltuples, double *indtuples,
								 bool *havedead)
{
	for (;;)
	{
		SpinLockAcquire(&shared->mutex);
		if (shared->nparticipantsdone == nparticipants)
		{
			*reltuples = shared->reltuples;
			*indtuples = shared->indtuples;
			if (havedead)
				*havedead = shared->havedead;
			if (shared->brokenhotchain)
				indexInfo->ii_BrokenHotChain = true;
			SpinLockRelease(&shared->mutex);
			break;
		}
		SpinLockRelease(&shared->mutex);

		ConditionVariableSleep(&shared->workersdonecv,
							   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}

	ConditionVariableCancelSleep();
}

static inline void
ParallelIndexBuildAbort(ParallelContext *pcxt, Snapshot snapshot)
{
	Assert(pcxt->nworkers_launched == 0);

	if (IsMVCCSnapshot(snapshot))
		UnregisterSnapshot(snapshot);
	DestroyParallelContext(pcxt);
	ExitParallelMode();
}

static inline void
ParallelIndexBuildEnd(ParallelContext *pcxt, BufferUsage *bufferusage,
					  WalUsage *walusage, Snapshot snapshot)
{
	/* Shutdown worker processes */
	WaitForParallelWorkersToFinish(pcxt);

	/* Accumulate WAL and buffer usage (must wait for workers to finish) */
	for (int i = 0; i < pcxt->nworkers_launched; i++)
		InstrAccumParallelQuery(&bufferusage[i], &walusage[i]);

	/* Free last reference to MVCC snapshot, if one was used */
	if (IsMVCCSnapshot(snapshot))
		UnregisterSnapshot(snapshot);
	DestroyParallelContext(pcxt);
	ExitParallelMode();
}

static inline void
ParallelIndexBuildLockModes(ParallelIndexBuildShared * shared,
							LOCKMODE *heapLockmode, LOCKMODE *indexLockmode)
{
	/* Use lock modes known to be obtained by index.c */
	if (!shared->isconcurrent)
	{
		*heapLockmode = ShareLock;
		*indexLockmode = AccessExclusiveLock;
	}
	else
	{
		*heapLockmode = ShareUpdateExclusiveLock;
		*indexLockmode = RowExclusiveLock;
	}
}

static inline void
ParallelIndexBuildOpenRelations(ParallelIndexBuildShared * shared,
								Relation *heapRel, Relation *indexRel)
{
	LOCKMODE	heapLockmode;
	LOCKMODE	indexLockmode;

	ParallelIndexBuildLockModes(shared, &heapLockmode, &indexLockmode);

#if PG_VERSION_NUM >= 140000
	/* Track query ID */
	pgstat_report_query_id(shared->queryid, false);
#endif

	*heapRel = table_open(shared->heaprelid, heapLockmode);
	*indexRel = index_open(shared->indexrelid, indexLockmode);
}

static inline void
ParallelIndexBuildCloseRelations(ParallelIndexBuildShared * shared,
								 Relation heapRel, Relation indexRel)
{
	LOCKMODE	heapLockmode;
	LOCKMODE	indexLockmode;

	ParallelIndexBuildLockModes(shared, &heapLockmode, &indexLockmode);

	index_close(indexRel, indexLockmode);
	table_close(heapRel, heapLockmode);
}

static inline double
ParallelIndexBuildScan(ParallelIndexBuildShared * shared, shm_toc *toc,
					   uint64 pscan_key, Relation heap, Relation index,
					   bool progress, IndexBuildCallback callback,
					   void *callback_state)
{
	ParallelTableScanDesc pscan;
	IndexInfo  *indexInfo;
	TableScanDesc scan;
	double		reltuples;

	indexInfo = BuildIndexInfo(index);
	indexInfo->ii_Concurrent = shared->isconcurrent;

	pscan = shm_toc_lookup(toc, pscan_key, false);
	scan = table_beginscan_parallel(heap, pscan
#if PG_VERSION_NUM >= 190000
									,SO_NONE
#endif
		);
	reltuples = table_index_build_scan(heap, index, indexInfo, true, progress,
									   callback, callback_state, scan);

	/* Record a broken HOT chain, so the leader can mark indcheckxmin */
	if (indexInfo->ii_BrokenHotChain)
	{
		SpinLockAcquire(&shared->mutex);
		shared->brokenhotchain = true;
		SpinLockRelease(&shared->mutex);
	}

	return reltuples;
}

static inline void
ParallelIndexBuildReportScanDone(ParallelIndexBuildShared * shared,
								 double reltuples, double indtuples,
								 bool havedead)
{
	SpinLockAcquire(&shared->mutex);
	shared->nparticipantsdone++;
	shared->reltuples += reltuples;
	shared->indtuples += indtuples;
	if (havedead)
		shared->havedead = true;
	SpinLockRelease(&shared->mutex);

	/* Notify leader */
	ConditionVariableSignal(&shared->workersdonecv);
}

static inline int
EstimateParallelQueryText(ParallelContext *pcxt)
{
	int			querylen;

	if (!debug_query_string)
		return 0;

	querylen = strlen(debug_query_string);
	shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
	shm_toc_estimate_keys(&pcxt->estimator, 1);

	return querylen;
}

static inline void
StoreParallelQueryText(ParallelContext *pcxt, uint64 key, int querylen)
{
	char	   *sharedquery;

	if (!debug_query_string)
		return;

	sharedquery = (char *) shm_toc_allocate(pcxt->toc, querylen + 1);
	memcpy(sharedquery, debug_query_string, querylen + 1);
	shm_toc_insert(pcxt->toc, key, sharedquery);
}

static inline void
RestoreParallelQueryText(shm_toc *toc, uint64 key)
{
	debug_query_string = shm_toc_lookup(toc, key, true);
	pgstat_report_activity(STATE_RUNNING, debug_query_string);
}

#endif							/* PG_VERSION_NUM >= 200000 */

#endif							/* PARALLELBUILD_H */
