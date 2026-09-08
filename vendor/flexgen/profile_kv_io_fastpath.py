import argparse
import json
import statistics
import time

import numpy as np
import torch


DEFAULT_PREFIX_FILE = (
    "cache/prefix/opt-13b/rte-single-prefix937-suffix50_16/"
    "facebook_opt-13b_attnid20_-2061317838873865398.npy"
)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_bytes(*tensors):
    total = 0
    for tensor in tensors:
        if tensor is not None:
            total += tensor.element_size() * tensor.nelement()
    return total


def new_cpu_buffer(shape, dtype):
    try:
        return torch.empty(shape, dtype=dtype, pin_memory=torch.cuda.is_available())
    except RuntimeError:
        return torch.empty(shape, dtype=dtype)


def split_chunks(tensor, chunk_size):
    chunks = []
    for start in range(0, tensor.shape[0], chunk_size):
        # Clone to mimic pcache's per-chunk CPU tensors instead of one big view.
        chunks.append(tensor[start:start + chunk_size].clone())
    return chunks


def make_selected_ids(seq_len, selected_count, mode, seed):
    if mode == "first":
        ids = torch.arange(selected_count, dtype=torch.long)
    elif mode == "stride":
        ids = torch.linspace(0, seq_len - 1, selected_count).long()
        ids = torch.unique(ids)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        ids = torch.randperm(seq_len, generator=generator)[:selected_count]
        ids, _ = torch.sort(ids)
    return ids


def median_time(fn, repeat):
    costs = []
    for _ in range(repeat):
        cuda_sync()
        start = time.monotonic()
        fn()
        cuda_sync()
        costs.append(time.monotonic() - start)
    return statistics.median(costs)


def chunk_loop_full(key_chunks, value_chunks, shape, per_chunk_sync, device):
    key_gpu = torch.empty(shape, dtype=key_chunks[0].dtype, device=device)
    value_gpu = torch.empty(shape, dtype=value_chunks[0].dtype, device=device)

    offset = 0
    for key_chunk, value_chunk in zip(key_chunks, value_chunks):
        if per_chunk_sync:
            cuda_sync()
        end = offset + key_chunk.shape[0]
        key_gpu[offset:end] = key_chunk.to(device, non_blocking=True)
        value_gpu[offset:end] = value_chunk.to(device, non_blocking=True)
        offset = end

    return key_gpu, value_gpu


def promote_then_evict(chunk, device):
    # Mimic gpu_size=0 cache update: CPU chunk is promoted to GPU, then evicted
    # back to CPU before the real get() result is copied to GPU.
    promoted = chunk.to(device, non_blocking=True)
    evicted = promoted.to("cpu")
    return evicted


def chunk_loop_full_promote_evict(key_chunks, value_chunks, shape, device):
    key_gpu = torch.empty(shape, dtype=key_chunks[0].dtype, device=device)
    value_gpu = torch.empty(shape, dtype=value_chunks[0].dtype, device=device)

    offset = 0
    for key_chunk, value_chunk in zip(key_chunks, value_chunks):
        cuda_sync()
        promote_then_evict(key_chunk, device)
        promote_then_evict(value_chunk, device)
        cuda_sync()

        end = offset + key_chunk.shape[0]
        key_gpu[offset:end] = key_chunk.to(device, non_blocking=True)
        value_gpu[offset:end] = value_chunk.to(device, non_blocking=True)
        offset = end

    return key_gpu, value_gpu


def pinned_gather_full(key_chunks, value_chunks, shape, device):
    key_cpu = new_cpu_buffer(shape, key_chunks[0].dtype)
    value_cpu = new_cpu_buffer(shape, value_chunks[0].dtype)

    offset = 0
    for key_chunk, value_chunk in zip(key_chunks, value_chunks):
        end = offset + key_chunk.shape[0]
        key_cpu[offset:end] = key_chunk
        value_cpu[offset:end] = value_chunk
        offset = end

    key_gpu = torch.empty(shape, dtype=key_chunks[0].dtype, device=device)
    value_gpu = torch.empty(shape, dtype=value_chunks[0].dtype, device=device)
    key_gpu.copy_(key_cpu, non_blocking=True)
    value_gpu.copy_(value_cpu, non_blocking=True)
    return key_gpu, value_gpu


def chunk_loop_selected(key_chunks, value_chunks, selected_ids, chunk_size, shape, per_chunk_sync, device):
    key_gpu = torch.empty(shape, dtype=key_chunks[0].dtype, device=device)
    value_gpu = torch.empty(shape, dtype=value_chunks[0].dtype, device=device)

    pos2chunk = selected_ids // chunk_size
    token_ids = selected_ids % chunk_size
    chunk_ids = torch.unique(pos2chunk)

    for chunk_id in chunk_ids:
        chunk_index = int(chunk_id.item())
        mask = pos2chunk == chunk_id
        token_id = token_ids[mask]
        if per_chunk_sync:
            cuda_sync()
        key_gpu[mask] = key_chunks[chunk_index][token_id].to(device, non_blocking=True)
        value_gpu[mask] = value_chunks[chunk_index][token_id].to(device, non_blocking=True)

    return key_gpu, value_gpu


def chunk_loop_selected_promote_evict(key_chunks, value_chunks, selected_ids, chunk_size, shape, device):
    key_gpu = torch.empty(shape, dtype=key_chunks[0].dtype, device=device)
    value_gpu = torch.empty(shape, dtype=value_chunks[0].dtype, device=device)

    pos2chunk = selected_ids // chunk_size
    token_ids = selected_ids % chunk_size
    chunk_ids = torch.unique(pos2chunk)

    for chunk_id in chunk_ids:
        chunk_index = int(chunk_id.item())
        mask = pos2chunk == chunk_id
        token_id = token_ids[mask]

        cuda_sync()
        promote_then_evict(key_chunks[chunk_index], device)
        promote_then_evict(value_chunks[chunk_index], device)
        cuda_sync()

        key_gpu[mask] = key_chunks[chunk_index][token_id].to(device, non_blocking=True)
        value_gpu[mask] = value_chunks[chunk_index][token_id].to(device, non_blocking=True)

    return key_gpu, value_gpu


def pinned_gather_selected(key_chunks, value_chunks, selected_ids, chunk_size, shape, device):
    key_cpu = new_cpu_buffer(shape, key_chunks[0].dtype)
    value_cpu = new_cpu_buffer(shape, value_chunks[0].dtype)

    pos2chunk = selected_ids // chunk_size
    token_ids = selected_ids % chunk_size
    chunk_ids = torch.unique(pos2chunk)

    for chunk_id in chunk_ids:
        chunk_index = int(chunk_id.item())
        mask = pos2chunk == chunk_id
        token_id = token_ids[mask]
        key_cpu[mask] = key_chunks[chunk_index][token_id]
        value_cpu[mask] = value_chunks[chunk_index][token_id]

    key_gpu = torch.empty(shape, dtype=key_chunks[0].dtype, device=device)
    value_gpu = torch.empty(shape, dtype=value_chunks[0].dtype, device=device)
    key_gpu.copy_(key_cpu, non_blocking=True)
    value_gpu.copy_(value_cpu, non_blocking=True)
    return key_gpu, value_gpu


def run_case(name, fn, moved_bytes, repeat):
    seconds = median_time(fn, repeat)
    gb = moved_bytes / 1e9
    return {
        "name": name,
        "bytes": moved_bytes,
        "gb": gb,
        "seconds": seconds,
        "bandwidth_gbps": gb / seconds if seconds > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-file", default=DEFAULT_PREFIX_FILE)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--select-percent", type=float, default=25.0)
    parser.add_argument("--select-mode", choices=["random", "first", "stride"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data = np.load(args.prefix_file, mmap_mode="r")
    if data.shape[0] != 2:
        raise ValueError(f"Expected prefix KV shape [2, seq, heads, dim], got {data.shape}")

    key_cpu = torch.from_numpy(np.array(data[0]))
    value_cpu = torch.from_numpy(np.array(data[1]))
    seq_len, num_heads, head_dim = key_cpu.shape

    key_chunks = split_chunks(key_cpu, args.chunk_size)
    value_chunks = split_chunks(value_cpu, args.chunk_size)

    selected_count = int(seq_len * args.select_percent / 100.0)
    selected_ids = make_selected_ids(seq_len, selected_count, args.select_mode, args.seed)

    full_shape = key_cpu.shape
    selected_shape = (selected_ids.shape[0], num_heads, head_dim)
    full_bytes = tensor_bytes(key_cpu, value_cpu)
    selected_bytes = selected_ids.shape[0] * num_heads * head_dim * key_cpu.element_size() * 2

    # Warm up CUDA and allocator.
    chunk_loop_full(key_chunks[:1], value_chunks[:1], key_chunks[0].shape, False, device)
    cuda_sync()

    results = []
    results.append(run_case(
        "full_chunk_loop_with_sync",
        lambda: chunk_loop_full(key_chunks, value_chunks, full_shape, True, device),
        full_bytes,
        args.repeat,
    ))
    results.append(run_case(
        "full_chunk_loop_promote_evict",
        lambda: chunk_loop_full_promote_evict(key_chunks, value_chunks, full_shape, device),
        full_bytes,
        args.repeat,
    ))
    results.append(run_case(
        "full_chunk_loop_no_sync",
        lambda: chunk_loop_full(key_chunks, value_chunks, full_shape, False, device),
        full_bytes,
        args.repeat,
    ))
    results.append(run_case(
        "full_pinned_gather_copy",
        lambda: pinned_gather_full(key_chunks, value_chunks, full_shape, device),
        full_bytes,
        args.repeat,
    ))
    results.append(run_case(
        "selected_chunk_loop_with_sync",
        lambda: chunk_loop_selected(key_chunks, value_chunks, selected_ids, args.chunk_size, selected_shape, True, device),
        selected_bytes,
        args.repeat,
    ))
    results.append(run_case(
        "selected_chunk_loop_promote_evict",
        lambda: chunk_loop_selected_promote_evict(key_chunks, value_chunks, selected_ids, args.chunk_size, selected_shape, device),
        selected_bytes,
        args.repeat,
    ))
    results.append(run_case(
        "selected_chunk_loop_no_sync",
        lambda: chunk_loop_selected(key_chunks, value_chunks, selected_ids, args.chunk_size, selected_shape, False, device),
        selected_bytes,
        args.repeat,
    ))
    results.append(run_case(
        "selected_pinned_gather_copy",
        lambda: pinned_gather_selected(key_chunks, value_chunks, selected_ids, args.chunk_size, selected_shape, device),
        selected_bytes,
        args.repeat,
    ))

    output = {
        "prefix_file": args.prefix_file,
        "shape": list(data.shape),
        "chunk_size": args.chunk_size,
        "select_percent": args.select_percent,
        "selected_count": int(selected_ids.shape[0]),
        "select_mode": args.select_mode,
        "device": device,
        "results": results,
    }

    print(json.dumps(output, indent=2))
    print("\nname\tGB\tseconds\tGB/s")
    for row in results:
        print(f"{row['name']}\t{row['gb']:.6f}\t{row['seconds']:.6f}\t{row['bandwidth_gbps']:.3f}")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
