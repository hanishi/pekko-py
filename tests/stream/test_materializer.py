"""Where streams run, after Apache Pekko's MaterializerForTypedSpec."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from ohra.stream import (
    AbruptStageTerminationError,
    Keep,
    Materializer,
    OverflowStrategy,
    QueueOfferResult,
    Sink,
    Source,
)
from ohra.stream import materializer as materializer_module
from ohra.testkit import ActorTestKit
from ..util import eventually


async def test_terminate_fails_running_streams():
    kit = ActorTestKit()
    async with kit:
        fut = Source.repeat(1).run_with(Sink.seq(), kit.system)
        kit.system.terminate()
        await kit.system.when_terminated()
    with pytest.raises(AbruptStageTerminationError):
        await fut


async def test_terminate_waits_for_a_stream_to_release_what_it_holds():
    log: list[str] = []
    opened = asyncio.Event()

    async def connection() -> AsyncGenerator[int, None]:
        opened.set()
        try:
            while True:
                yield 1
                await asyncio.sleep(0.01)
        finally:
            await asyncio.sleep(0.05)  # an orderly close, such as a client's
            log.append("closed")

    kit = ActorTestKit()
    async with kit:
        Source.from_async_iterable(connection).run_with(Sink.ignore(), kit.system)
        await opened.wait()
    assert log == ["closed"]  # when_terminated() came after the close, not before it


async def test_terminate_cancels_a_stream_cleanup_that_does_not_finish(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(materializer_module, "TEARDOWN_TIMEOUT", 0.05)
    log: list[str] = []
    opened = asyncio.Event()

    async def stuck() -> AsyncGenerator[int, None]:
        opened.set()
        try:
            while True:
                yield 1
                await asyncio.sleep(0.01)
        finally:
            try:
                await asyncio.Event().wait()  # a close that never returns
            except asyncio.CancelledError:
                log.append("cancelled")
                raise

    kit = ActorTestKit()
    async with kit:
        Source.from_async_iterable(stuck).run_with(Sink.ignore(), kit.system)
        await opened.wait()
    assert log == ["cancelled"]  # terminate() returned, and nothing outlives it


async def test_run_after_terminate_raises():
    kit = ActorTestKit()
    async with kit:
        pass
    with pytest.raises(RuntimeError, match="shut down"):
        Source.single(1).run_with(Sink.seq(), kit.system)


async def test_one_materializer_per_system():
    async with ActorTestKit() as kit:
        assert Materializer(kit.system) is Materializer(kit.system)


async def test_each_run_is_a_child_of_the_stream_supervisor():
    async with ActorTestKit() as kit:
        supervisor = (
            kit.system.path.removesuffix("/user") + "/system/Materializers/StreamSupervisor"
        )
        never: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        out = Source.future(never).run_with(Sink.seq(), kit.system)

        def running() -> bool:
            try:
                kit.system.resolve(f"{supervisor}/$0")
            except LookupError:
                return False
            return True

        assert running()
        never.set_result(1)
        assert await out == [1]
        await eventually(lambda: not running())


def _stop_all_runs(kit: ActorTestKit) -> None:
    """Stop every run's actor right after run(), before its drive task takes a first step."""
    root = kit.system._system_root  # pyright: ignore[reportPrivateUsage]
    supervisor = root.children["Materializers"].children["StreamSupervisor"]
    for child in list(supervisor.children.values()):
        supervisor.stop_child(child.ref)


async def test_run_stopped_before_the_drive_starts_is_aborted():
    # no finally block of the drive or the source runs: the drive coroutine is closed unstarted
    async with ActorTestKit() as kit:
        q, fut = (
            Source.queue(1, OverflowStrategy.backpressure)
            .to_mat(Sink.seq(), Keep.both)
            .run(kit.system)
        )
        q2, sq = (
            Source.queue(1, OverflowStrategy.backpressure)
            .to_mat(Sink.queue(), Keep.both)
            .run(kit.system)
        )
        _stop_all_runs(kit)
        with pytest.raises(AbruptStageTerminationError):
            await asyncio.wait_for(fut, 1)
        assert await asyncio.wait_for(q.offer(1), 1) == QueueOfferResult.QueueClosed
        with pytest.raises(AbruptStageTerminationError):
            await asyncio.wait_for(q.watch_completion(), 1)
        with pytest.raises(AbruptStageTerminationError):
            await asyncio.wait_for(sq.pull(), 1)
        assert await asyncio.wait_for(q2.offer(1), 1) == QueueOfferResult.QueueClosed
        with pytest.raises(AbruptStageTerminationError):
            await asyncio.wait_for(q2.watch_completion(), 1)


async def test_callback_registered_after_abort_runs_at_once():
    from ohra.stream.dsl import _Ctx  # pyright: ignore[reportPrivateUsage]

    async with ActorTestKit() as kit:
        ctx = _Ctx(kit.system)
        calls: list[str] = []
        ctx.on_abort(lambda: calls.append("early"))
        ctx.abort()
        ctx.on_abort(lambda: calls.append("late"))
        assert calls == ["early", "late"]
