"""Operators shared by ``Source`` and ``Flow``. Each takes the upstream iterator and the run's
context and returns the downstream iterator, an async generator that holds its upstream in
``contextlib.aclosing``, so closing it cancels upstream at once."""

from __future__ import annotations

import contextlib
import asyncio
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from ..ref import ActorRef
    from ..scheduler import Cancellable
    from .dsl import Done, Source, _Ctx  # pyright: ignore[reportPrivateUsage]


def closing[I: AsyncIterator[Any]](it: I) -> contextlib.AbstractAsyncContextManager[I]:
    """``contextlib.aclosing`` for a stage's upstream. Every iterator in a stream has ``aclose``
    (sources and operators are async generators, or a ``Stage``), though ``AsyncIterator``
    doesn't declare it."""
    return contextlib.aclosing(cast(Any, it))


async def run_drive[I, R](
    ctx: _Ctx, upstream: AsyncIterator[I], body: Callable[[AsyncIterator[I]], Awaitable[R]]
) -> R:
    """The shared shape of every sink drive: ``body`` pulls ``upstream`` in a child task, and the
    drive waits for it inside ``closing(upstream)``. When the drive task is cancelled (the run's
    actor stopped) it calls ``ctx.abort()`` first, so that the stages unwinding under the cancel
    see the run aborted, and only then cancels the child and waits for it, so that upstream is
    closed with no task suspended inside it. Waiting with ``asyncio.wait`` keeps the cancel from
    reaching the child before the abort."""
    async with closing(upstream) as up:
        child = asyncio.ensure_future(body(up))
        try:
            await asyncio.wait((child,))
        except asyncio.CancelledError:
            ctx.abort()
            raise
        finally:
            if not child.done():
                child.cancel()
                await asyncio.gather(child, return_exceptions=True)
        return child.result()


@runtime_checkable
class FailsEagerly(Protocol):
    """A source that can fail outside a pull (``ActorSource``): it calls each listener with the
    failure at once, so the stage right below can settle without waiting for the next pull."""

    def on_failure(self, listener: Callable[[BaseException], None]) -> None: ...


def watch_ref(ref: ActorRef[Any], on_terminated: Callable[[], None]) -> Callable[[], None]:
    """Run ``on_terminated`` once when the local actor ``ref`` stops (on the next loop step if it
    already has); return the unwatch. A ref that cannot be watched (a remote one) is not."""
    from ..cell import _cell_of, is_local_actor  # pyright: ignore[reportPrivateUsage]

    if not is_local_actor(ref):
        return lambda: None
    done = _cell_of(ref).done

    def notify(_: object) -> None:
        on_terminated()

    def unwatch() -> None:
        done.remove_done_callback(notify)

    done.add_done_callback(notify)
    return unwatch


class Stage[T]:
    """An operator's iterator, as ``_via_op`` hands it downstream. Closing it closes the
    operator, then its upstream: an async generator closed before its first pull never enters
    its ``aclosing``, so on its own it would leave a source's materialized resources (a queue,
    an actor) open when a stream is cancelled before it pulls (``Sink.cancelled``)."""

    __slots__ = ("_gen", "_on_close", "_upstream")

    def __init__(
        self,
        gen: AsyncIterator[T],
        upstream: AsyncIterator[Any],
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._gen = gen
        self._upstream = upstream
        self._on_close = on_close  # runs even if the operator never started

    def __aiter__(self) -> Stage[T]:
        return self

    def __anext__(self) -> Awaitable[T]:
        return self._gen.__anext__()

    async def aclose(self) -> None:
        try:
            async with closing(self._upstream), closing(self._gen):  # exits the operator first
                pass
        finally:
            if self._on_close is not None:
                self._on_close()


def map_op[I, O](f: Callable[[I], O]) -> Callable[[AsyncIterator[I], _Ctx], AsyncIterator[O]]:
    async def run(upstream: AsyncIterator[I], _ctx: _Ctx) -> AsyncGenerator[O, None]:
        async with closing(upstream) as up:
            async for x in up:
                yield f(x)

    return run


def filter_op[T](p: Callable[[T], bool]) -> Callable[[AsyncIterator[T], _Ctx], AsyncIterator[T]]:
    async def run(upstream: AsyncIterator[T], _ctx: _Ctx) -> AsyncGenerator[T, None]:
        async with closing(upstream) as up:
            async for x in up:
                if p(x):
                    yield x

    return run


type Op[I, O] = Callable[[AsyncIterator[I], _Ctx], AsyncIterator[O]]


def filter_not_op[T](p: Callable[[T], bool]) -> Op[T, T]:
    return filter_op(lambda x: not p(x))


def collect_op[I, O](pf: Callable[[I], O | None]) -> Op[I, O]:
    async def run(upstream: AsyncIterator[I], _ctx: _Ctx) -> AsyncGenerator[O, None]:
        async with closing(upstream) as up:
            async for x in up:
                y = pf(x)
                if y is not None:
                    yield y

    return run


def map_concat_op[I, O](f: Callable[[I], Iterable[O]]) -> Op[I, O]:
    async def run(upstream: AsyncIterator[I], _ctx: _Ctx) -> AsyncGenerator[O, None]:
        async with closing(upstream) as up:
            async for x in up:
                for y in f(x):
                    yield y

    return run


def take_op[T](n: int) -> Op[T, T]:
    async def run(upstream: AsyncIterator[T], ctx: _Ctx) -> AsyncGenerator[T, None]:
        # Pekko's Take pushes the nth element and completes the stage at once: the close of
        # upstream starts with the nth element, which goes down without waiting for it
        closer: asyncio.Future[None] | None = None
        try:
            count = 0
            if n > 0:
                async for x in upstream:
                    count += 1
                    if count >= n:
                        closer = close = asyncio.ensure_future(cast(Any, upstream).aclose())
                        ctx.on_abort(close.cancel)  # a run torn down does not wait on the close
                        yield x
                        return
                    yield x
        finally:
            await (closer if closer is not None else cast(Any, upstream).aclose())

    return run


def take_while_op[T](p: Callable[[T], bool], inclusive: bool) -> Op[T, T]:
    async def run(upstream: AsyncIterator[T], _ctx: _Ctx) -> AsyncGenerator[T, None]:
        async with closing(upstream) as up:
            async for x in up:
                if p(x):
                    yield x
                else:
                    if inclusive:
                        yield x
                    return

    return run


def drop_op[T](n: int) -> Op[T, T]:
    async def run(upstream: AsyncIterator[T], _ctx: _Ctx) -> AsyncGenerator[T, None]:
        async with closing(upstream) as up:
            left = n
            async for x in up:
                if left > 0:
                    left -= 1
                else:
                    yield x

    return run


def drop_while_op[T](p: Callable[[T], bool]) -> Op[T, T]:
    async def run(upstream: AsyncIterator[T], _ctx: _Ctx) -> AsyncGenerator[T, None]:
        async with closing(upstream) as up:
            dropping = True
            async for x in up:
                if dropping and p(x):
                    continue
                dropping = False
                yield x

    return run


def grouped_op[T](n: int) -> Op[T, list[T]]:
    if n <= 0:
        raise ValueError("n must be greater than 0")

    async def run(upstream: AsyncIterator[T], _ctx: _Ctx) -> AsyncGenerator[list[T], None]:
        async with closing(upstream) as up:
            buf: list[T] = []
            async for x in up:
                buf.append(x)
                if len(buf) == n:
                    out, buf = buf, []
                    yield out
            if buf:
                yield buf

    return run


def scan_op[I, S](zero: S, f: Callable[[S, I], S]) -> Op[I, S]:
    async def run(upstream: AsyncIterator[I], _ctx: _Ctx) -> AsyncGenerator[S, None]:
        async with closing(upstream) as up:
            acc = zero
            yield acc
            async for x in up:
                acc = f(acc, x)
                yield acc

    return run


def fold_op[I, S](zero: S, f: Callable[[S, I], S]) -> Op[I, S]:
    async def run(upstream: AsyncIterator[I], _ctx: _Ctx) -> AsyncGenerator[S, None]:
        async with closing(upstream) as up:
            acc = zero
            async for x in up:
                acc = f(acc, x)
            yield acc

    return run


def recover_op[T](f: Callable[[Exception], T | None]) -> Op[T, T]:
    async def run(upstream: AsyncIterator[T], _ctx: _Ctx) -> AsyncGenerator[T, None]:
        async with closing(upstream) as up:
            while True:
                try:
                    x = await anext(up)
                except StopAsyncIteration:
                    return
                except Exception as e:
                    last = f(e)
                    if last is None:
                        raise
                    yield last  # the recovery element, then completion
                    return
                yield x

    return run


def recover_with_retries_op[T](
    attempts: int, f: Callable[[Exception], Source[T, Any] | None]
) -> Op[T, T]:
    async def run(upstream: AsyncIterator[T], ctx: _Ctx) -> AsyncGenerator[T, None]:
        current: AsyncIterator[T] = upstream
        attempt = 0
        while True:
            try:
                async with closing(current) as up:
                    async for x in up:
                        yield x
                return
            except Exception as e:
                if attempts >= 0 and attempt >= attempts:
                    raise
                source = f(e)
                if source is None:
                    raise
                attempt += 1
                current, _ = source._build(ctx)  # pyright: ignore[reportPrivateUsage]

    return run


def watch_stage[T](up: AsyncIterator[T], ctx: _Ctx, fut: asyncio.Future[Done]) -> Stage[T]:
    """Settles ``fut`` with ``DONE`` on completion and on downstream cancel, with the upstream
    exception on failure (at once, unpulled, for an upstream with ``on_failure``:
    ``ActorSource``), and with ``AbruptStageTerminationError`` when the run is aborted
    (``terminate()``). A ``CancelledError`` passing through is not settled here: it is the
    downstream cancelling a pull (``DONE``, from the close) or the run's abort."""
    from .dsl import DONE, AbruptStageTerminationError

    def settle(error: BaseException | None) -> None:
        if fut.done():
            return
        if error is not None:
            fut.set_exception(error)
        elif ctx.aborted:
            fut.set_exception(AbruptStageTerminationError("stream terminated abruptly"))
        else:
            fut.set_result(DONE)

    ctx.on_abort(lambda: settle(None))
    if isinstance(up, FailsEagerly):
        up.on_failure(settle)

    async def run() -> AsyncGenerator[T, None]:
        try:
            async with closing(up) as u:
                async for x in u:
                    yield x
        except Exception as e:
            settle(e)
            raise
        settle(None)

    return Stage(run(), up, lambda: settle(None))  # the close: downstream cancel, even unstarted


def new_done_future() -> asyncio.Future[Done]:
    fut: asyncio.Future[Done] = asyncio.get_running_loop().create_future()
    # a failure nobody retrieved must not log "exception was never retrieved"
    fut.add_done_callback(lambda f: f.cancelled() or f.exception())
    return fut


def map_async_op[I, O](
    parallelism: int,
    f: Callable[[I], Awaitable[O]],
    ordered: bool,
    stop: asyncio.Future[Any] | None = None,
) -> Op[I, O]:
    """After Pekko's ``MapAsync`` (``ordered``) and ``MapAsyncUnordered``: pulls upstream only
    while fewer than ``parallelism`` calls are in flight, fails with the first failure, and
    cancels what is still running when the stage ends for any reason. A failure of ``stop`` fails
    the stage at once, as an upstream failure does in Pekko, whether a pull is waiting or not
    (``ActorFlow.ask``'s watch). ``stop`` may only fail: done, it stays in the wait set, so a
    result would make the stage spin; that raises ``RuntimeError`` instead."""
    if parallelism < 1:
        raise ValueError("parallelism must be at least 1")

    async def run(upstream: AsyncIterator[I], _ctx: _Ctx) -> AsyncGenerator[O, None]:
        pending: deque[asyncio.Future[O]] = deque()
        pull: asyncio.Future[I] | None = None
        done = False
        async with closing(upstream) as up:
            try:
                while True:
                    if not done and pull is None and len(pending) < parallelism:
                        pull = asyncio.ensure_future(anext(up))
                    if pull is None and not pending:
                        return
                    waiting: set[asyncio.Future[Any]] = set(pending)
                    if pull is not None:
                        waiting.add(pull)
                    if stop is not None:
                        waiting.add(stop)
                    await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                    for t in pending:
                        if t.done() and not t.cancelled() and t.exception() is not None:
                            raise cast(BaseException, t.exception())
                    if stop is not None and stop.done():
                        stop.result()  # raises its failure; a result is a caller's bug:
                        raise RuntimeError("map_async's stop future completed without failing")
                    if pull is not None and pull.done():
                        got, pull = pull, None
                        try:
                            pending.append(asyncio.ensure_future(f(got.result())))
                        except StopAsyncIteration:
                            done = True
                        continue
                    if ordered:
                        while pending and pending[0].done():
                            yield pending.popleft().result()
                    else:
                        for t in [t for t in pending if t.done()]:
                            pending.remove(t)
                            yield t.result()
            finally:
                # before the upstream closes: a pull still suspended inside it would make
                # its aclose() fail with "already running"
                tasks: list[asyncio.Future[Any]] = [*pending]
                if pull is not None:
                    tasks.append(pull)
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    return run


def _timer(ctx: _Ctx, delay: float) -> tuple[asyncio.Future[None], Cancellable]:
    """A future the system scheduler resolves after ``delay``, so ``ManualTime`` controls it."""
    fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def fire() -> None:
        if not fut.done():
            fut.set_result(None)

    return fut, ctx.system.scheduler.schedule_once(delay, fire)


def throttle_op[T](elements: int, per: float, maximum_burst: int | None) -> Op[T, T]:
    """After Pekko's ``Throttle`` in shaping mode, with ``NanoTimeTokenBucket`` on the scheduler's
    clock: an element that finds no token waits for the delay the bucket names."""
    if elements <= 0:
        raise ValueError("elements must be > 0")
    if per <= 0:
        raise ValueError("per time must be > 0")
    per_ns = round(per * 1e9)
    if per_ns < elements:
        raise ValueError("Rates larger than 1 unit / nanosecond are not supported")
    between = per_ns // elements  # nanosBetweenTokens
    if maximum_burst is None:  # Pekko's AutomaticMaximumBurst
        capacity = max(1, 100_000_000 // between)
    elif maximum_burst < 0:
        raise ValueError("maximum_burst must be >= 0")
    else:
        capacity = maximum_burst

    async def run(upstream: AsyncIterator[T], ctx: _Ctx) -> AsyncGenerator[T, None]:
        scheduler = ctx.system.scheduler

        def now_ns() -> int:
            return round(scheduler.current_time * 1e9)

        # Pekko's TokenBucket, in integer nanoseconds: init()
        available = capacity
        last_update = now_ns()
        timer: Cancellable | None = None
        try:
            async with closing(upstream) as up:
                async for x in up:
                    # TokenBucket.offer(1)
                    now = now_ns()
                    elapsed = now - last_update
                    arrived = 0
                    if elapsed >= between:
                        arrived = elapsed // between
                        last_update += arrived * between
                    available = min(available + arrived, capacity)
                    if available >= 1:
                        available -= 1
                    else:
                        remaining = 1 - available
                        delay = remaining * between - (now - last_update)
                        available = 0
                        last_update = now + delay
                        fut, timer = _timer(ctx, delay / 1e9)
                        await fut
                    yield x
        finally:
            if timer is not None:
                timer.cancel()

    return run


def grouped_within_op[T](n: int, d: float) -> Op[T, list[T]]:
    """After Pekko's ``GroupedWeightedWithin`` with every weight 1: a group closes at ``n``
    elements, or when the timer fires with elements buffered. The timer restarts after each group
    and runs on while the buffer is empty, so no group is ever empty. Upstream is pulled in a
    task that races the timer."""
    if n <= 0:
        raise ValueError("max number must be > 0")
    if d <= 0:
        raise ValueError("interval must be > 0")

    async def run(upstream: AsyncIterator[T], ctx: _Ctx) -> AsyncGenerator[list[T], None]:
        buf: list[T] = []
        pull: asyncio.Future[T] | None = None
        timer, cancellable = _timer(ctx, d)
        async with closing(upstream) as up:
            try:
                while True:
                    if pull is None:
                        pull = asyncio.ensure_future(anext(up))
                    await asyncio.wait((pull, timer), return_when=asyncio.FIRST_COMPLETED)
                    if pull.done():
                        got, pull = pull, None
                        try:
                            buf.append(got.result())
                        except StopAsyncIteration:
                            break
                        if len(buf) == n:
                            cancellable.cancel()
                            group, buf = buf, []
                            timer, cancellable = _timer(ctx, d)
                            yield group
                    else:  # the timer fired
                        timer, cancellable = _timer(ctx, d)
                        if buf:
                            group, buf = buf, []
                            yield group
            finally:
                # before the upstream closes: a pull still suspended inside it would make
                # its aclose() fail with "already running"
                cancellable.cancel()
                if pull is not None:
                    pull.cancel()
                    await asyncio.gather(pull, return_exceptions=True)
            if buf:
                yield buf

    return run


class TickSource[T]:
    """After Pekko's ``TickSource``: the iterator and the materialized ``Cancellable`` in one.
    The fixed-delay scheduler job starts when the stream is materialized; a tick that finds no
    pull waiting is dropped, and cancelling completes the stream."""

    def __init__(self, ctx: _Ctx, initial_delay: float, interval: float, tick: T) -> None:
        self._tick = tick
        self._waiter: asyncio.Future[None] | None = None
        self._stopped = False
        self._job = ctx.system.scheduler.schedule_with_fixed_delay(
            initial_delay, interval, self._fire
        )
        ctx.on_abort(self.cancel)

    def _fire(self) -> None:
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)

    def cancel(self) -> bool:
        if self._stopped:
            return False
        self._stopped = True
        self._job.cancel()
        self._fire()
        return True

    @property
    def is_cancelled(self) -> bool:
        return self._stopped

    def __aiter__(self) -> TickSource[T]:
        return self

    async def __anext__(self) -> T:
        if self._stopped:
            raise StopAsyncIteration
        self._waiter = asyncio.get_running_loop().create_future()
        try:
            await self._waiter
        finally:
            self._waiter = None
        if self._stopped:
            raise StopAsyncIteration
        return self._tick

    async def aclose(self) -> None:
        self.cancel()


def merge_op[T](other: Source[T, Any], eager_complete: bool) -> Op[T, T]:
    """After Pekko's ``Merge`` of two inputs. One pump task per input pulls an element, hands it
    over and pulls the next only once the merge has taken it, so each input holds at most one
    element, as Pekko's inlets do. Elements go out in arrival order. Completes when both inputs
    have completed, or the first time one does with ``eager_complete``; a failure fails it at
    once. ``other`` is built with the run, its materialized value dropped."""

    def op(upstream: AsyncIterator[T], ctx: _Ctx) -> AsyncIterator[T]:
        second, _ = other._build(ctx)  # pyright: ignore[reportPrivateUsage]

        async def run() -> AsyncGenerator[T, None]:
            # an element with the future its pump waits on, a failure, or None for completion
            arrived: deque[tuple[T, asyncio.Future[None]] | Exception | None] = deque()
            wake = asyncio.Event()

            async def pump(it: AsyncIterator[T]) -> None:
                try:
                    async for x in it:
                        taken = asyncio.get_running_loop().create_future()
                        arrived.append((x, taken))
                        wake.set()
                        await taken
                except Exception as e:
                    arrived.appendleft(e)  # ahead of pending elements: Pekko fails at once
                else:
                    arrived.append(None)
                wake.set()

            async with closing(upstream) as up, closing(second) as other_up:
                pumps = [asyncio.ensure_future(pump(up)), asyncio.ensure_future(pump(other_up))]
                try:
                    running = len(pumps)
                    while running:
                        while not arrived:
                            wake.clear()
                            await wake.wait()
                        item = arrived.popleft()
                        if isinstance(item, Exception):
                            raise item
                        if eager_complete and (item is None or None in arrived):
                            # as Pekko's Merge: the completion cancels the other input, and the
                            # element it had pending is discarded (an input completes only
                            # once its own element is taken)
                            return
                        if item is None:
                            running -= 1
                        else:
                            x, taken = item
                            taken.set_result(None)
                            yield x
                finally:
                    # before the inputs close: a pump suspended inside one would make its
                    # aclose() fail with "already running"
                    for p in pumps:
                        p.cancel()
                    await asyncio.gather(*pumps, return_exceptions=True)

        return Stage(run(), second)  # closes ``second`` even if never pulled

    return op


def zip_with_op[A, B, R](other: Source[B, Any], f: Callable[[A, B], R]) -> Op[A, R]:
    """After Pekko's ``ZipWith2``: pulls both inputs for each element and completes as soon as
    either completes. ``other`` is built with the run, its materialized value dropped."""

    def op(upstream: AsyncIterator[A], ctx: _Ctx) -> AsyncIterator[R]:
        second, _ = other._build(ctx)  # pyright: ignore[reportPrivateUsage]

        async def run() -> AsyncGenerator[R, None]:
            async with closing(upstream) as up, closing(second) as other_up:
                pulls: list[asyncio.Future[Any]] = []
                try:
                    while True:
                        pull_a = asyncio.ensure_future(anext(up))
                        pull_b = asyncio.ensure_future(anext(other_up))
                        pulls = [pull_a, pull_b]
                        try:
                            a, b = await asyncio.gather(pull_a, pull_b)
                        except StopAsyncIteration:
                            return
                        pulls = []
                        yield f(a, b)
                finally:
                    # before the inputs close: the other pull may still be suspended in one
                    for p in pulls:
                        p.cancel()
                    await asyncio.gather(*pulls, return_exceptions=True)

        return Stage(run(), second)

    return op


def concat_op[T](other: Source[T, Any]) -> Op[T, T]:
    """After Pekko's ``Concat`` of two inputs: ``other`` is built with the run, its materialized
    value dropped, and first pulled once upstream has completed."""

    def op(upstream: AsyncIterator[T], ctx: _Ctx) -> AsyncIterator[T]:
        second, _ = other._build(ctx)  # pyright: ignore[reportPrivateUsage]

        async def run() -> AsyncGenerator[T, None]:
            async with closing(upstream) as up:
                async for x in up:
                    yield x
            async with closing(second) as other_up:
                async for x in other_up:
                    yield x

        return Stage(run(), second)  # closes ``second`` when the stream ends before it

    return op
