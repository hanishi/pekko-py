"""Shared memory handles. Skipped without numpy."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pytest

np = pytest.importorskip("numpy")

from ohra import ActorContext, ActorRef, ActorSystem, Behavior, Behaviors, Success, Try  # noqa: E402
from ohra.ext.shm import SharedArray  # noqa: E402


def total_in_worker(handle: SharedArray) -> float:
    """Runs in another process: maps the segment, reads it, does not own it."""
    with handle.open() as arr:
        return float(arr.sum())


def double_in_worker(handle: SharedArray) -> SharedArray:
    """Runs in another process: owns the input, returns a new segment it hands back."""
    out = SharedArray.empty(handle.shape, handle.dtype)
    with handle.open() as src, out.open(writeable=True) as dst:
        np.multiply(src, 2, out=dst)
    handle.unlink()
    return out


def test_handle_is_small_and_views_are_read_only():
    arr = np.arange(1000, dtype=np.float64)
    handle = SharedArray.create(arr)
    try:
        assert len(repr(handle)) < 200
        with handle.open() as view:
            assert view.sum() == arr.sum()
            with pytest.raises(ValueError):
                view[0] = 1.0
    finally:
        handle.unlink()


def test_another_process_reads_the_same_bytes():
    handle = SharedArray.create(np.ones(10_000))
    try:
        with ProcessPoolExecutor(1) as pool:
            assert pool.submit(total_in_worker, handle).result() == 10_000.0
    finally:
        handle.unlink()


def test_a_process_that_only_reads_leaves_the_segment_to_its_owner():
    # a separate interpreter has a resource tracker of its own, which on 3.12 unlinked every
    # segment the process attached when it exited
    handle = SharedArray.create(np.ones(10))
    try:
        reader = (
            "import sys; from ohra.ext.shm import SharedArray\n"
            f"with SharedArray({handle.name!r}, (10,), {handle.dtype!r}).open() as arr:\n"
            "    sys.exit(0 if arr.sum() == 10 else 1)"
        )
        done = subprocess.run([sys.executable, "-c", reader], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "leaked" not in done.stderr
        with handle.open() as arr:
            assert arr.sum() == 10
    finally:
        handle.unlink()


@dataclass(frozen=True)
class Scale:
    data: SharedArray
    reply_to: ActorRef[float]


@dataclass(frozen=True)
class Scaled:
    result: Try[SharedArray]
    reply_to: ActorRef[float]


type Cmd = Scale | Scaled


def scaler(pool: ProcessPoolExecutor) -> Behavior[Cmd]:
    def setup(ctx: ActorContext[Cmd]) -> Behavior[Cmd]:
        loop = asyncio.get_running_loop()

        def on_message(msg: Cmd) -> Behavior[Cmd]:
            match msg:
                case Scale(data, reply_to):
                    # ownership moves to the worker
                    fut = loop.run_in_executor(pool, double_in_worker, data)
                    ctx.pipe_to_self(fut, lambda r: Scaled(r, reply_to))
                case Scaled(Success(out), reply_to):
                    with out.open() as arr:  # the reply handed ownership back to us
                        reply_to.tell(float(arr.sum()))
                    out.unlink()
                case Scaled(_, reply_to):
                    reply_to.tell(float("nan"))
            return Behaviors.same

        return Behaviors.receive_message(on_message)

    return Behaviors.setup(setup)


async def test_actor_hands_an_array_to_a_process_and_gets_one_back():
    with ProcessPoolExecutor(1) as pool:
        async with ActorSystem(scaler(pool), "t") as system:
            handle = SharedArray.create(np.arange(100, dtype=np.float64))
            assert await system.ask(lambda r: Scale(handle, r), timeout=30) == 2 * sum(range(100))
    with pytest.raises(FileNotFoundError):  # both segments were unlinked
        handle.unlink()
