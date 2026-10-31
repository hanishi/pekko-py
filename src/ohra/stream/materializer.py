"""Where streams run: Apache Pekko's ``Materializer``. Each system has one, which spawns Pekko's
``/system/Materializers/StreamSupervisor`` on first use and one child of it per run. The child
drives the sink with ``pipe_to_self`` and stops when the stream ends; when the system terminates,
stopping the child cancels the drive, which aborts the run (``_Ctx.abort``) so the sink fails its
pending materialized value with ``AbruptStageTerminationError``. The child's stop waits for the
stream to tear down, up to ``TEARDOWN_TIMEOUT``, so ``terminate()`` returns after every source has
run its cleanup. If the child stops before the drive's first step, the materializer aborts the run
itself."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from ..behavior import Behavior, Behaviors
from ..signals import PostStop, Signal
from .dsl import _Ctx  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from ..cell import ActorCell
    from ..context import ActorContext
    from ..system import ActorSystem
    from .dsl import RunnableGraph

# How long a run's actor, when stopped, waits for the stream's cleanup (a source's ``finally``,
# say) before cancelling it a second time; as remoting's shutdown-flush-timeout
TEARDOWN_TIMEOUT = 1.0

_materializers: weakref.WeakKeyDictionary[ActorSystem[Any], Materializer] = (
    weakref.WeakKeyDictionary()
)


class Materializer:
    """``Materializer(system)`` is the system's one materializer, as Pekko's
    ``SystemMaterializer``. It holds its system weakly, so a terminated system can be collected."""

    _system: weakref.ref[ActorSystem[Any]]

    def __new__(cls, system: ActorSystem[Any]) -> Materializer:
        m = _materializers.get(system)
        if m is None:
            m = _materializers[system] = super().__new__(cls)
            m._system = weakref.ref(system)
        return m

    def materialize[M](self, graph: RunnableGraph[M]) -> M:
        """Build the graph and start it in a new child of the stream supervisor; return its
        materialized value. ``RuntimeError`` once the system is terminating."""
        system = self._system()
        if system is None or system._system_root.stopping:  # pyright: ignore[reportPrivateUsage]
            raise RuntimeError("stream materializer has been shut down")
        root = system._system_root  # pyright: ignore[reportPrivateUsage]
        ctx = _Ctx(system)
        try:
            drive, mat = graph._build(ctx)  # pyright: ignore[reportPrivateUsage]
        except BaseException:
            ctx.abort()  # stages built before the failure (a tick's job) release what they hold
            raise
        started = False

        async def start() -> None:
            nonlocal started
            started = True
            await drive

        def ended(_: object) -> None:
            if not started:  # stopped before the drive's first step: nothing else settles it
                drive.close()
                ctx.abort()

        child = _child(_child(root, "Materializers"), "StreamSupervisor").spawn_child(
            _run(start), None, None
        )
        child.done.add_done_callback(ended)
        return mat


class _Stop:
    """The stream ended: its actor stops."""


def _child(parent: ActorCell[Any], name: str) -> ActorCell[Any]:
    child = parent.children.get(name)
    return child if child is not None else parent.spawn_child(Behaviors.empty, name, None)


def _run(start: Callable[[], Coroutine[Any, Any, None]]) -> Behavior[_Stop]:
    def setup(ctx: ActorContext[_Stop]) -> Behavior[_Stop]:
        drive = asyncio.ensure_future(start())
        ctx.pipe_to_self(drive, lambda _: _Stop())

        async def on_signal(ctx: ActorContext[_Stop], sig: Signal) -> Behavior[_Stop]:
            if sig is PostStop:  # stopping cancelled the drive: wait until it has torn down
                done, _ = await asyncio.wait((drive,), timeout=TEARDOWN_TIMEOUT)
                if not done:
                    ctx.log.warning(
                        "stream cleanup still running after %s s: cancelling it again",
                        TEARDOWN_TIMEOUT,
                    )
                    drive.cancel()
                    await asyncio.wait((drive,))
            return Behaviors.same

        return Behaviors.receive_message(lambda _: Behaviors.stopped()).receive_signal(on_signal)

    return Behaviors.setup(setup)
