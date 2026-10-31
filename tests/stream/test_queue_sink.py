"""Sink.queue, after Apache Pekko's QueueSinkSpec."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from ohra.stream import AbruptStageTerminationError, Sink, Source
from ohra.testkit import ActorTestKit

from ..util import eventually


async def test_sink_queue_pull_returns_none_at_end():
    async with ActorTestKit() as kit:
        q = Source.from_iterable([1, 2]).run_with(Sink.queue(), kit.system)
        assert await q.pull() == 1
        assert await q.pull() == 2
        assert await q.pull() is None
        assert await q.pull() is None


async def test_sink_queue_async_for_collects_all():
    async with ActorTestKit() as kit:
        q = Source.from_iterable([1, 2, 3]).run_with(Sink.queue(), kit.system)
        assert [x async for x in q] == [1, 2, 3]


async def test_iterating_a_sink_queue_keeps_none_elements_apart_from_the_end():
    async with ActorTestKit() as kit:
        q = Source.from_iterable([1, None]).run_with(Sink.queue(), kit.system)
        assert [x async for x in q] == [1, None]


async def test_anext_with_a_default_tells_a_none_element_from_the_end():
    # pull() returns None for both, as head_option does; anext(q, default) does not
    end = object()
    async with ActorTestKit() as kit:
        q = Source.from_iterable([None]).run_with(Sink.queue(), kit.system)
        assert await anext(q, end) is None
        assert await anext(q, end) is end


async def test_sink_queue_pull_raises_on_failure():
    async def gen() -> AsyncGenerator[int, None]:
        yield 1
        raise ValueError("boom")

    async with ActorTestKit() as kit:
        q = Source.from_async_iterable(gen).run_with(Sink.queue(), kit.system)
        assert await q.pull() == 1
        with pytest.raises(ValueError, match="boom"):
            await q.pull()
        with pytest.raises(ValueError, match="boom"):
            await q.pull()


async def test_sink_queue_async_for_raises_on_failure():
    async def gen() -> AsyncGenerator[int, None]:
        yield 1
        raise ValueError("boom")

    async with ActorTestKit() as kit:
        q = Source.from_async_iterable(gen).run_with(Sink.queue(), kit.system)
        got: list[int] = []
        with pytest.raises(ValueError, match="boom"):
            async for x in q:
                got.append(x)
        assert got == [1]


async def test_sink_queue_pulls_only_on_demand():
    yielded: list[int] = []

    async def gen() -> AsyncGenerator[int, None]:
        for i in range(10):
            yielded.append(i)
            yield i

    async with ActorTestKit() as kit:
        q = Source.from_async_iterable(gen).run_with(Sink.queue(), kit.system)
        for _ in range(5):
            await asyncio.sleep(0)
        assert len(yielded) <= 1
        assert await q.pull() == 0
        for _ in range(5):
            await asyncio.sleep(0)
        assert len(yielded) <= 2
        q.cancel()


async def test_sink_queue_cancel_cancels_upstream():
    log: list[str] = []

    async def gen() -> AsyncGenerator[int, None]:
        try:
            for i in range(10):
                yield i
        finally:
            log.append("closed")

    async with ActorTestKit() as kit:
        q = Source.from_async_iterable(gen).run_with(Sink.queue(), kit.system)
        assert await q.pull() == 0
        q.cancel()
        await eventually(lambda: log == ["closed"])
        assert await q.pull() is None


async def test_sink_queue_cancel_with_suspended_upstream():
    log: list[str] = []
    suspended = asyncio.Event()

    async def gen() -> AsyncGenerator[int, None]:
        try:
            yield 0
            suspended.set()
            await asyncio.Event().wait()  # never set: suspended mid-pull
            yield 1
        finally:
            log.append("closed")

    async with ActorTestKit() as kit:
        q = Source.from_async_iterable(gen).run_with(Sink.queue(), kit.system)
        assert await q.pull() == 0
        pending = asyncio.ensure_future(q.pull())
        await suspended.wait()  # the drive's pull task is inside the generator's __anext__
        q.cancel()
        await eventually(lambda: log == ["closed"])
        assert await pending is None


async def test_sink_queue_concurrent_pull_fails():
    async with ActorTestKit() as kit:
        gate = asyncio.Event()

        async def gen() -> AsyncGenerator[int, None]:
            await gate.wait()
            yield 1

        q = Source.from_async_iterable(gen).run_with(Sink.queue(), kit.system)
        first = asyncio.ensure_future(q.pull())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="concurrent pulls"):
            await q.pull()
        gate.set()
        assert await first == 1
        q.cancel()


async def test_cancelled_pull_does_not_lose_the_element():
    gate = asyncio.Event()

    async def gen() -> AsyncGenerator[int, None]:
        await gate.wait()
        yield 1

    async with ActorTestKit() as kit:
        q = Source.from_async_iterable(gen).run_with(Sink.queue(), kit.system)
        first = asyncio.ensure_future(q.pull())
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        gate.set()
        assert await q.pull() == 1
        q.cancel()


async def test_terminate_while_pull_is_pending():
    suspended = asyncio.Event()
    log: list[str] = []

    async def gen() -> AsyncGenerator[int, None]:
        try:
            suspended.set()
            await asyncio.Event().wait()
            yield 1
        finally:
            log.append("closed")

    async with ActorTestKit() as kit:
        q = Source.from_async_iterable(gen).run_with(Sink.queue(), kit.system)
        pending = asyncio.ensure_future(q.pull())
        await suspended.wait()
        kit.system.terminate()
        await kit.system.when_terminated()
    with pytest.raises(AbruptStageTerminationError):
        await asyncio.wait_for(pending, 1)
    with pytest.raises(AbruptStageTerminationError):
        await q.pull()
    assert log == ["closed"]
