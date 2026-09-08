"""Read-only SSD diagnostic backend. No cache eviction or payload mutation.

O_DIRECT uses 512-byte aligned reads (the target ext4/SATA device's alignment).
All variants pay the same bounce-buffer/copy overhead. This is a diagnostic,
not a claim that the existing buffered production reader uses direct I/O.
"""
from __future__ import annotations

import os
import threading
import torch

_ALIGNMENT = 512
_LOCAL = threading.local()
_INSTALLED = False


def aligned_range(offset: int, size: int, alignment: int = _ALIGNMENT):
    if offset < 0 or size < 0 or alignment <= 0:
        raise ValueError("invalid read range")
    start = offset // alignment * alignment
    end = (offset + size + alignment - 1) // alignment * alignment
    return start, end - start, offset - start


def direct_pread_into_pinned(fd, *, nbytes, offset, out=None):
    if out is None:
        out = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    if out.dtype != torch.uint8 or out.numel() != nbytes or not out.is_contiguous():
        raise ValueError("expected contiguous byte output of requested length")
    if nbytes == 0:
        return out
    start, length, skip = aligned_range(int(offset), int(nbytes))
    storage = getattr(_LOCAL, "storage", None)
    if storage is None or storage.numel() < length + _ALIGNMENT:
        storage = torch.empty(length + _ALIGNMENT, dtype=torch.uint8, pin_memory=True)
        _LOCAL.storage = storage
    alignment_skip = (-storage.data_ptr()) % _ALIGNMENT
    bounce = storage[alignment_skip:alignment_skip + length]
    view = memoryview(bounce.numpy()).cast("B")
    count = os.preadv(fd, [view], start)
    # A rounded final read may legitimately stop at EOF.
    if count < skip + nbytes:
        raise EOFError(f"short direct read: {count}, need {skip + nbytes}")
    out.copy_(bounce[skip:skip + nbytes])
    return out


def install_direct_io():
    global _INSTALLED
    if _INSTALLED:
        return
    if not hasattr(os, "O_DIRECT"):
        raise RuntimeError("direct I/O diagnostic requires Linux O_DIRECT")
    from . import mixed_precision_reader as reader_module

    def direct_fd(self, path):
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            self._fds[path] = fd
        return fd

    reader_module.MixedPrecisionPayloadReader._fd = direct_fd
    reader_module.pread_into_pinned = direct_pread_into_pinned
    _INSTALLED = True


def process_io():
    with open("/proc/self/io", encoding="ascii") as stream:
        return {key: int(value) for key, value in
                (line.split(":") for line in stream)}


def close_reader(reader):
    for fd in reader._fds.values():
        os.close(fd)
    reader._fds.clear()
