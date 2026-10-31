"""``Source.queue`` and ``Sink.queue``, after Apache Pekko's ``QueueSource`` and ``QueueSink``.

Both queues are created when the graph is built, before the stream runs, so their state is
settled by the iterator's ``aclose()`` (a plain method that runs even if the stream never pulled)
and not only by generator ``finally`` blocks. A stream torn down answers a pending ``offer`` with
``QueueClosed`` and a pending ``pull`` with an exception or ``None``; it never hangs them."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, final

from .actor_source import BufferOverflowError, OverflowStrategy
from .ops import new_done_future, run_drive

if TYPE_CHECKING:
    from .dsl import Done, _Ctx  # pyright: ignore[reportPrivateUsage]

_log = logging.getLogger(__name__)


class QueueOfferResult:
    """What ``SourceQueue.offer`` reports: ``Enqueued``, ``Dropped``, ``QueueClosed`` or
    ``Failure(cause)``."""

    Enqueued: ClassVar[QueueOfferResult]
    Dropped: ClassVar[QueueOfferResult]
    QueueClosed: ClassVar[QueueOfferResult]
    Failure: ClassVar[type[Failure]]


@final
@dataclass(frozen=True)
class _Enqueued(QueueOfferResult):
    pass


@final
@dataclass(frozen=True)
class _Dropped(QueueOfferResult):
    pass


@final
@dataclass(frozen=True)
class _QueueClosed(QueueOfferResult):
    pass


@final
@dataclass(frozen=True)
class Failure(QueueOfferResult):
    cause: BaseException


QueueOfferResult.Enqueued = _Enqueued()
QueueOfferResult.Dropped = _Dropped()
QueueOfferResult.QueueClosed = _QueueClosed()
QueueOfferResult.Failure = Failure

_ENQUEUED = QueueOfferResult.Enqueued
_DROPPED = QueueOfferResult.Dropped
_CLOSED = QueueOfferResult.QueueClosed

_MAX_CONCURRENT_OFFERS = 1  # Pekko's default; not configurable here


class SourceQueue[T]:
    """Pekko's ``SourceQueueWithComplete``: feed a stream from outside with ``offer``."""

    def __init__(self, buffer_size: int, overflow_strategy: OverflowStrategy, ctx: _Ctx) -> None:
        self._ctx = ctx
        ctx.on_abort(self._abort)
        self._max = buffer_size
        self._strategy = overflow_strategy
        self._buffer: deque[T] = deque()
        self._pending: deque[tuple[T, asyncio.Future[QueueOfferResult]]] = deque()
        self._waiter: asyncio.Future[tuple[T] | None] | None = None  # the consumer's demand
        self._terminating = False
        self._finished = False
        self._failure: BaseException | None = None
        self._completion = new_done_future()

    # --- the public side -----------------------------------------------------------------

    def watch_completion(self) -> asyncio.Future[Done]:
        return self._completion

    async def offer(self, elem: T) -> QueueOfferResult:
        """``Enqueued`` once the element is in the buffer or delivered, ``Dropped`` when the
        strategy drops it or the queue is completing, ``QueueClosed`` after the stream ended,
        ``Failure`` on ``OverflowStrategy.fail`` overflow. Under ``backpressure`` it waits
        while the buffer is full; a second offer waiting at the same time raises
        ``RuntimeError``, as Pekko's ``IllegalStateException``."""
        if self._finished:
            return _CLOSED
        if self._terminating:
            return _DROPPED
        fut: asyncio.Future[QueueOfferResult] = asyncio.get_running_loop().create_future()
        if self._max != 0:
            self._buffer_elem(elem, fut)
            if self._waiter is not None and self._buffer:
                self._deliver(self._buffer.popleft())
        elif self._waiter is not None:
            self._deliver(elem)
            fut.set_result(_ENQUEUED)
        elif len(self._pending) < _MAX_CONCURRENT_OFFERS:
            self._pending.append((elem, fut))
        else:
            self._no_buffer_overflow(elem, fut)
        entry = next((e for e in self._pending if e[1] is fut), None)
        try:
            return await fut
        except asyncio.CancelledError:
            if entry is not None and entry in self._pending:
                self._pending.remove(entry)
            raise

    def complete(self) -> None:
        if self._finished or self._terminating:
            return
        if self._buffer or self._pending:
            self._terminating = True
        else:
            self._finish(None)

    def fail(self, exc: BaseException) -> None:
        if not self._finished:
            self._finish(exc)

    # --- the stream's side ---------------------------------------------------------------

    def _buffer_elem(self, elem: T, fut: asyncio.Future[QueueOfferResult]) -> None:
        if len(self._buffer) < self._max:
            self._buffer.append(elem)
            fut.set_result(_ENQUEUED)
            return
        match self._strategy:
            case OverflowStrategy.drop_head:
                _log.debug("Dropping the head element because buffer is full [DropHead]")
                self._buffer.popleft()
            case OverflowStrategy.drop_tail:
                _log.debug("Dropping the tail element because buffer is full [DropTail]")
                self._buffer.pop()
            case OverflowStrategy.drop_buffer:
                _log.debug(
                    "Dropping all the buffered elements because buffer is full [DropBuffer]"
                )
                self._buffer.clear()
            case OverflowStrategy.fail:
                self._overflow_failure(fut)
                return
            case OverflowStrategy.backpressure:
                _log.debug(
                    "Backpressuring because buffer is full and overflowStrategy is: [Backpressure]"
                )
                if len(self._pending) >= _MAX_CONCURRENT_OFFERS:
                    fut.set_exception(_too_many_offers())
                else:
                    self._pending.append((elem, fut))
                return
        self._buffer.append(elem)
        fut.set_result(_ENQUEUED)

    def _no_buffer_overflow(self, elem: T, fut: asyncio.Future[QueueOfferResult]) -> None:
        match self._strategy:
            case OverflowStrategy.drop_head | OverflowStrategy.drop_buffer:
                _log.debug(
                    "Dropping element because buffer is full and overflowStrategy is: [%s]",
                    self._strategy.name,
                )
                _, old = self._pending.popleft()
                old.set_result(_DROPPED)
                self._pending.append((elem, fut))
            case OverflowStrategy.drop_tail:
                _log.debug(
                    "Dropping element because buffer is full and overflowStrategy is: [DropTail]"
                )
                fut.set_result(_DROPPED)
            case OverflowStrategy.fail:
                self._overflow_failure(fut)
            case OverflowStrategy.backpressure:
                fut.set_exception(
                    RuntimeError(
                        "You have to wait for previous offer to be resolved to send another one"
                    )
                )

    def _overflow_failure(self, fut: asyncio.Future[QueueOfferResult]) -> None:
        _log.debug("Failing because buffer is full and overflowStrategy is: [Fail]")
        exc = BufferOverflowError(f"Buffer overflow (max capacity was: {self._max})!")
        fut.set_result(Failure(exc))
        self._finish(exc)

    def _deliver(self, elem: T) -> None:
        waiter, self._waiter = self._waiter, None
        if waiter is not None:
            waiter.set_result((elem,))

    def _finish(self, exc: BaseException | None, *, abrupt: bool = False) -> None:
        """The stream is over: complete (``exc`` None), fail, or detach (downstream cancel)."""
        self._finished = True
        self._failure = exc
        self._buffer.clear()
        while self._pending:
            _, fut = self._pending.popleft()
            if not fut.done():
                fut.set_result(_CLOSED)
        if not self._completion.done():
            if exc is not None:
                self._completion.set_exception(exc)
            else:
                from .dsl import DONE, AbruptStageTerminationError

                if abrupt:
                    self._completion.set_exception(
                        AbruptStageTerminationError("stream terminated abruptly")
                    )
                else:
                    self._completion.set_result(DONE)
        waiter, self._waiter = self._waiter, None
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    async def _next(self) -> T:
        """The consumer's pull: Pekko's ``onPull``."""
        while True:
            if self._failure is not None:
                raise self._failure
            if self._buffer:
                elem = self._buffer.popleft()
                while self._pending and len(self._buffer) < self._max:
                    e, fut = self._pending.popleft()
                    self._buffer.append(e)
                    fut.set_result(_ENQUEUED)
                if self._terminating and not self._buffer:
                    self._finish(None)
                return elem
            if self._max == 0 and self._pending:
                elem, fut = self._pending.popleft()
                fut.set_result(_ENQUEUED)
                if self._terminating and not self._pending:
                    self._finish(None)
                return elem
            if self._finished:
                raise StopAsyncIteration
            if self._waiter is not None:
                raise RuntimeError("a SourceQueue has one consumer")
            waiter = self._waiter = asyncio.get_running_loop().create_future()
            try:
                got = await waiter
            finally:
                if self._waiter is waiter:
                    self._waiter = None
            if got is not None:
                return got[0]

    def _abort(self) -> None:
        if not self._finished:
            self._finish(None, abrupt=True)

    async def _detach(self) -> None:
        """Downstream finished: pending offers see ``QueueClosed``, completion is ``DONE``, or
        fails if the run was aborted (``terminate()``)."""
        if not self._finished:
            self._finish(None, abrupt=self._ctx.aborted)


class _SourceQueueIterator[T]:
    """What the stream pulls. ``aclose()`` is a plain method, so it works before a first pull."""

    __slots__ = ("_queue",)

    def __init__(self, queue: SourceQueue[T]) -> None:
        self._queue = queue

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        return await self._queue._next()  # pyright: ignore[reportPrivateUsage]

    async def aclose(self) -> None:
        await self._queue._detach()  # pyright: ignore[reportPrivateUsage]


def _too_many_offers() -> RuntimeError:
    return RuntimeError(
        f"Too many concurrent offers. Specified maximum is {_MAX_CONCURRENT_OFFERS}. "
        "You have to wait for one previous future to be resolved to send another request"
    )


def source_queue_build[T](
    buffer_size: int, overflow_strategy: OverflowStrategy, ctx: _Ctx
) -> tuple[AsyncIterator[T], SourceQueue[T]]:
    queue = SourceQueue[T](buffer_size, overflow_strategy, ctx)
    return _SourceQueueIterator(queue), queue


_END: Any = object()


class SinkQueue[T]:
    """Pekko's ``SinkQueueWithCancel``: pull the stream's elements from outside, one ``pull``
    per element. Upstream is pulled only on demand."""

    def __init__(self, ctx: _Ctx) -> None:
        self._ctx = ctx
        ctx.on_abort(self._abort)
        self._sig: asyncio.Future[None] | None = None  # what the drive sleeps on
        self._want = False  # a pull is waiting
        self._cancelled = False
        self._waiter: asyncio.Future[Any] | None = None  # the pending pull
        self._ready: deque[Any] = deque()  # pulled for a caller that gave up
        self._end: BaseException | None = None  # set once the stream is over
        self._over = False

    async def pull(self) -> T | None:
        """The next element; ``None`` at completion; raises the stream's failure. A ``None``
        element looks like the end here: read such a stream with ``async for`` or
        ``anext(queue, default)``."""
        elem = await self._next()
        return None if elem is _END else elem

    async def _next(self) -> Any:
        if self._ready:
            return self._ready.popleft()
        if self._over:
            if self._end is not None:
                raise self._end
            return _END
        if self._waiter is not None and not self._waiter.done():
            raise RuntimeError(
                "Too many concurrent pulls. You have to wait for the previous pull to complete"
            )
        waiter = self._waiter = asyncio.get_running_loop().create_future()
        self._want = True
        self._poke()
        return await waiter

    def cancel(self) -> None:
        """Cancel the stream: upstream is cancelled and pulls return ``None``."""
        self._cancelled = True
        self._poke()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        elem = await self._next()
        if elem is _END:
            raise StopAsyncIteration
        return elem

    # --- the drive ------------------------------------------------------------------------

    def _settle(self, outcome: Any, exc: BaseException | None) -> None:
        """Deliver an element, or the end of the stream, to the pending pull."""
        waiter = self._waiter
        if outcome is _END:
            if self._over:
                return
            self._over = True
            self._end = exc
        if waiter is None or waiter.done():
            if outcome is not _END:
                self._ready.append(outcome)
        elif exc is not None:
            waiter.set_exception(exc)
        else:
            waiter.set_result(outcome)
        self._waiter = None

    def _abort(self) -> None:
        from .dsl import AbruptStageTerminationError

        self._settle(_END, AbruptStageTerminationError("stream terminated abruptly"))

    def _poke(self) -> None:
        sig = self._sig
        if sig is not None and not sig.done():
            sig.set_result(None)

    async def _sleep(self) -> None:
        sig = self._sig = asyncio.get_running_loop().create_future()
        await sig

    async def drive(self, upstream: AsyncIterator[T]) -> None:
        await run_drive(self._ctx, upstream, self._loop)
        if not self._over:
            self._settle(_END, None)

    async def _loop(self, up: AsyncIterator[T]) -> None:
        pulling: asyncio.Future[T] | None = None
        try:
            while True:
                while not self._want and not self._cancelled:
                    await self._sleep()
                if self._cancelled:
                    break
                pulling = asyncio.ensure_future(anext(up))
                pulling.add_done_callback(lambda _: self._poke())
                while not pulling.done() and not self._cancelled:
                    await self._sleep()
                if not pulling.done():
                    break
                self._want = False
                try:
                    self._settle(pulling.result(), None)
                except StopAsyncIteration:
                    self._settle(_END, None)
                    break
                except Exception as e:
                    self._settle(_END, e)
                    break
                pulling = None
        finally:
            # upstream is closed only once no task is suspended inside its __anext__
            if pulling is not None and not pulling.done():
                pulling.cancel()
                await asyncio.gather(pulling, return_exceptions=True)


def sink_queue_build[T](
    upstream: AsyncIterator[T], ctx: _Ctx
) -> tuple[Coroutine[Any, Any, None], SinkQueue[T]]:
    queue = SinkQueue[T](ctx)
    return queue.drive(upstream), queue
