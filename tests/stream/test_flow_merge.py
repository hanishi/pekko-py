"""merge, after Apache Pekko's FlowMergeSpec."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from ohra.stream import DONE, AbruptStageTerminationError, Done, Flow, Keep, Sink, Source
from ohra.testkit import ActorTestKit

from ..util import suspending


def _released(release: asyncio.Event, error: Exception | None = None) -> Source[int, object]:
    """Completes, or fails with ``error``, once ``release`` is set."""

    async def gen() -> AsyncGenerator[int, None]:
        await release.wait()
        if error is not None:
            raise error
        return
        yield

    return Source.from_async_iterable(gen)


async def test_merge_emits_all_elements():
    async with ActorTestKit() as kit:
        merged = (
            Source.from_iterable(range(0, 10, 3))
            .merge(Source.from_iterable(range(1, 10, 3)))
            .merge(Source.from_iterable(range(2, 10, 3)))
        )
        assert sorted(await merged.run_with(Sink.seq(), kit.system)) == list(range(10))


async def test_merge_completes_when_both_complete():
    async with ActorTestKit() as kit:
        merged = Source.empty().merge(Source.from_iterable([1, 2, 3, 4]))
        assert await merged.run_with(Sink.seq(), kit.system) == [1, 2, 3, 4]


async def test_merge_eager_complete():
    log: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    async with ActorTestKit() as kit:
        merged = _released(release).merge(
            Source.from_async_iterable(suspending(log, started, 1)), eager_complete=True
        )
        out = merged.run_with(Sink.seq(), kit.system)
        await started.wait()
        release.set()
        assert await asyncio.wait_for(out, 1) == [1]  # bounded: a regression hangs here
        assert log == ["closed"]


async def test_merge_eager_complete_discards_the_other_inputs_pending_element():
    # Pekko's FlowMergeSpec "not try to grab from a closed input previously enqueued": the
    # completion cancels the other input, and the element it had pending is not emitted
    finished = asyncio.Event()

    async def other() -> AsyncGenerator[str, None]:
        yield "b1"
        finished.set()  # the first input completes, behind b2
        yield "b2"  # pulled as soon as b1 is taken, and pending while nobody pulls
        await asyncio.Event().wait()

    completed = asyncio.Event()

    async def first() -> AsyncGenerator[str, None]:
        try:
            await finished.wait()
            return
            yield
        finally:
            completed.set()  # in the step that hands the merge its completion

    async with ActorTestKit() as kit:
        out = (
            Source.from_async_iterable(first)
            .merge(Source.from_async_iterable(other), eager_complete=True)
            .run_with(Sink.queue(), kit.system)
        )
        assert await out.pull() == "b1"
        await completed.wait()  # b2, then the completion, are waiting in the merge
        assert [x async for x in out] == []


async def test_merge_failure_cancels_other():
    log: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()
    async with ActorTestKit() as kit:
        merged = _released(release, ValueError("boom")).merge(
            Source.from_async_iterable(suspending(log, started))
        )
        out = merged.run_with(Sink.seq(), kit.system)
        await started.wait()
        release.set()
        with pytest.raises(ValueError, match="boom"):
            await out
        assert log == ["closed"]


async def test_merge_fails_with_an_immediately_failed_input():
    async with ActorTestKit() as kit:
        merged = Source.failed(ValueError("boom")).merge(Source.from_iterable([1, 2, 3]))
        with pytest.raises(ValueError, match="boom"):
            await merged.run_with(Sink.seq(), kit.system)


async def test_merge_passes_along_early_cancellation():
    log: list[str] = []
    started = (asyncio.Event(), asyncio.Event())
    async with ActorTestKit() as kit:
        merged = Source.from_async_iterable(suspending(log, started[0], 1)).merge(
            Source.from_async_iterable(suspending(log, started[1]))
        )
        q = merged.run_with(Sink.queue(), kit.system)
        assert await q.pull() == 1
        pull = asyncio.ensure_future(q.pull())
        await started[0].wait()
        await started[1].wait()  # both inputs suspended inside a pull
        q.cancel()
        assert await asyncio.wait_for(pull, 1) is None  # answered once upstream is closed
        assert log == ["closed", "closed"]


async def test_merge_other_input_sees_the_run_aborted():
    stash: list[asyncio.Future[Done]] = []
    log: list[str] = []
    started = asyncio.Event()
    kit = ActorTestKit()
    async with kit:
        other = Source.from_async_iterable(suspending(log, started)).watch_termination(
            lambda _, done: stash.append(done)
        )
        out = Source.from_async_iterable(suspending([], asyncio.Event())).merge(other)
        seq = out.run_with(Sink.seq(), kit.system)
        await started.wait()
        kit.system.terminate()
        await kit.system.when_terminated()
    with pytest.raises(AbruptStageTerminationError):
        stash[0].result()
    with pytest.raises(AbruptStageTerminationError):
        await seq


async def test_merge_closes_the_other_when_cancelled_before_the_first_pull():
    stash: list[asyncio.Future[Done]] = []
    async with ActorTestKit() as kit:
        other = Source.from_iterable([2]).watch_termination(lambda _, done: stash.append(done))
        merged = Source.single(1).merge(other).watch_termination(Keep.right)
        assert await merged.to_mat(Sink.cancelled(), Keep.left).run(kit.system) is DONE
        assert stash[0].result() is DONE


async def test_merge_on_flow():
    async with ActorTestKit() as kit:
        flow = Flow.of(int).merge(Source.from_iterable([10, 20]))
        out = Source.from_iterable([1, 2]).via(flow).run_with(Sink.seq(), kit.system)
        assert sorted(await out) == [1, 2, 10, 20]
