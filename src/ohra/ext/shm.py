"""Zero-copy numpy arrays between processes. Optional: needs numpy.

A ``SharedArray`` is a small, picklable handle: segment name, shape and dtype. The
array itself lives in ``multiprocessing.shared_memory``. Ownership rule: whoever holds
the handle owns the segment. Sending the handle transfers ownership, and the owner
calls ``unlink()`` exactly once when the data is no longer needed. Readers that do
not own it use ``with handle.open() as arr:`` and leave the segment alone.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

# Keep the resource tracker out of it. Ownership is explicit here, and the tracker would
# otherwise unlink a segment when a process that merely attached exits, or warn about a
# segment that another process unlinked. 3.13 has track=False; before it, each segment
# created or opened is unregistered again (bpo-38119). Only POSIX has the tracker.
_KW: dict[str, Any] = {"track": False} if sys.version_info >= (3, 13) else {}
_UNTRACK = sys.version_info < (3, 13) and os.name == "posix"


def _untracked(seg: SharedMemory) -> SharedMemory:
    if _UNTRACK:
        resource_tracker.unregister("/" + seg.name, "shared_memory")
    return seg


@dataclass(frozen=True, slots=True)
class SharedArray:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @staticmethod
    def empty(shape: tuple[int, ...], dtype: DTypeLike = np.float64) -> SharedArray:
        """Allocate a segment. Fill it through ``open()``; nothing is copied."""
        dt = np.dtype(dtype)
        size = max(1, int(np.prod(shape)) * dt.itemsize)
        seg = _untracked(SharedMemory(create=True, size=size, **_KW))
        try:
            return SharedArray(seg.name, tuple(shape), dt.str)
        finally:
            seg.close()

    @staticmethod
    def create(arr: NDArray[Any]) -> SharedArray:
        """Copy ``arr`` into a new segment. One copy here, none afterwards."""
        handle = SharedArray.empty(arr.shape, arr.dtype)
        with handle.open(writeable=True) as view:
            view[...] = arr
        return handle

    def open(self, writeable: bool = False) -> _View:
        """``with handle.open() as arr:`` maps the segment as an ndarray for the block.
        Read-only by default, matching the immutability rule for messages."""
        return _View(self, writeable)

    def unlink(self) -> None:
        """Free the segment. Only the owner calls this, exactly once."""
        SharedMemory(self.name, **_KW).unlink()  # unlink() unregisters what this registers


class _View:
    __slots__ = ("_handle", "_writeable", "_seg", "_arr")

    def __init__(self, handle: SharedArray, writeable: bool) -> None:
        self._handle = handle
        self._writeable = writeable

    def __enter__(self) -> NDArray[Any]:
        self._seg = _untracked(SharedMemory(self._handle.name, **_KW))
        self._arr: NDArray[Any] = np.ndarray(
            self._handle.shape, np.dtype(self._handle.dtype), buffer=self._seg.buf
        )
        self._arr.flags.writeable = self._writeable
        return self._arr

    def __exit__(
        self, et: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None
    ) -> None:
        del self._arr  # release the buffer export before closing the mapping
        self._seg.close()
