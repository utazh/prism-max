"""LMCache adapter for sparse layer plans followed by an uncached query tail.

The upstream experimental layer-plan path reports either every aligned chunk
as a hit or zero chunks.  A score request includes the shared prefix plus new
query tokens, so the last query chunk is naturally absent from the warmed
cache.  This adapter preserves the original LMCache source and changes the
server process only: it returns the longest contiguous prefix whose selected
layer objects are present, letting vLLM compute the query tail normally.
"""

from __future__ import annotations

from functools import wraps
from typing import Sequence


def contiguous_prefix_hits(chunk_handle_results: Sequence[Sequence[bool]]) -> int:
    """Count leading logical chunks for which every selected object was found.

    An empty inner sequence represents a dropped-all-layer chunk.  The layer
    retrieval path zero-fills that chunk, so it is a valid sparse-cache hit.
    """

    hits = 0
    for chunk_results in chunk_handle_results:
        if not all(chunk_results):
            break
        hits += 1
    return hits


def install_layer_plan_partial_prefix_patch() -> None:
    """Patch only the LMCache server class used by this reproduction runner."""

    from lmcache.v1.multiprocess import server

    engine_cls = server.MPCacheEngine
    if getattr(engine_cls, "_contiguous_fuxian_partial_prefix_patch", False):
        return

    original_lookup = engine_cls.lookup
    original_status = engine_cls.query_prefetch_status

    @wraps(original_lookup)
    def lookup_with_partial_prefix(self, key, tp_size):
        if not server.layer_group_plan_enabled():
            return original_lookup(self, key, tp_size)

        model_name, world_size = key.model_name, key.world_size
        server.log_telemetry(server.make_start_event("lookup_and_prefetch", key.request_id))
        layout_desc = self._find_layout_desc(model_name, world_size)
        if layout_desc is None:
            return original_lookup(self, key, tp_size)

        extra_count = server.compute_extra_count(tp_size, world_size)
        chunk_hashes = self.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            return original_lookup(self, key, tp_size)

        total_chunks = len(chunk_hashes)
        num_layers = int(layout_desc.shapes[0][1])
        layer_plan = server.build_external_layer_plan_tiers(
            total_chunks,
            num_layers,
            request_id=key.request_id,
            sensitive_layers=server.int8_layers_from_env(),
        )
        specs = server._layer_object_specs(key, chunk_hashes, layer_plan)
        specs_by_chunk = [[] for _ in range(total_chunks)]
        for spec in specs:
            specs_by_chunk[spec.chunk_index].append(spec)

        chunk_jobs = []
        all_spans = []
        started = server.time.perf_counter()
        for chunk_specs in specs_by_chunk:
            grouped = {}
            for spec in chunk_specs:
                grouped.setdefault((spec.codec, spec.layer_count), []).append(spec)
            handles = []
            for (codec, layer_count), codec_specs in grouped.items():
                keys = [spec.object_key for spec in codec_specs]
                handle = self.storage_manager.submit_prefetch_task(
                    keys,
                    server.build_layer_object_layout_desc(
                        layout_desc, codec, layer_count=layer_count
                    ),
                    extra_count=extra_count,
                )
                span = server._PrefetchSpanJob(handle=handle, chunk_count=1, keys=keys)
                handles.append(span)
                all_spans.append(span)
            chunk_jobs.append(tuple(handles))

        server._write_precision_timeline_event(
            {
                "event": "prefetch_start",
                "request_id": key.request_id,
                "policy": server.precision_policy_from_env(),
                "chunks": total_chunks,
                "layer_objects": len(specs),
                "partial_prefix_mode": True,
                "elapsed_ms": 0.0,
            }
        )
        job = server._PrefetchJob(
            spans=tuple(all_spans),
            world_size=key.world_size,
            request_id=key.request_id,
            started_perf=started,
            total_chunks=total_chunks,
            policy=server.precision_policy_from_env(),
            layer_plan_mode=True,
        )
        job.contiguous_fuxian_chunk_jobs = tuple(chunk_jobs)
        return self._register_prefetch_job(job)

    @wraps(original_status)
    def status_with_partial_prefix(self, prefetch_job_id):
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(prefetch_job_id)
        chunk_jobs = getattr(job, "contiguous_fuxian_chunk_jobs", None) if job else None
        if chunk_jobs is None:
            return original_status(self, prefetch_job_id)

        pipeline = server._pipeline_prefetch_enabled()
        for handles in chunk_jobs:
            for span in handles:
                if span.found_count is not None:
                    continue
                found_count = (
                    self.storage_manager.query_prefetch_expected(span.handle)
                    if pipeline
                    else self.storage_manager.query_prefetch_status(span.handle)
                )
                if found_count is None:
                    return None
                span.found_count = 1 if found_count >= len(span.keys) else 0

        found_chunks = contiguous_prefix_hits(
            [[span.found_count == 1 for span in handles] for handles in chunk_jobs]
        )
        server._write_precision_timeline_event(
            {
                "event": "prefetch_end",
                "request_id": job.request_id,
                "policy": job.policy,
                "chunks": job.total_chunks,
                "found_chunks": found_chunks,
                "layer_plan_mode": True,
                "partial_prefix_mode": True,
                "prefetch_ms": server._elapsed_ms(job.started_perf),
                "elapsed_ms": server._elapsed_ms(job.started_perf),
            }
        )
        server.logger.info(
            "CONTIG_LAYER_PREFIX_MATCH chunks=%d found_chunks=%d layer_objects=%d",
            job.total_chunks,
            found_chunks,
            len(job.spans),
        )
        server.log_telemetry(
            server.make_end_event(
                "lookup_and_prefetch", job.request_id, found_count=found_chunks
            )
        )
        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(prefetch_job_id, None)
        return found_chunks

    engine_cls.lookup = lookup_with_partial_prefix
    engine_cls.query_prefetch_status = status_with_partial_prefix
    engine_cls._contiguous_fuxian_partial_prefix_patch = True
