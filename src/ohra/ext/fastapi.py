"""FastAPI integration: an ActorSystem owned by the app's lifespan, reached through a typed
dependency.

    actors = Actors(guardian, "app")
    app = FastAPI(lifespan=actors.lifespan, exception_handlers=actors.exception_handlers)

    @app.get("/x")
    async def x(system: ActorSystem[Cmd] = actors.depends()) -> ...:
        return await system.ask(..., timeout=1)

With ``actors.exception_handlers``, an ``AskTimeoutError`` escaping a route becomes a 504.
``actors.websocket`` gives a WebSocket route an actor per connection, as Play's
``ActorFlow.actorRef`` does: the route owns the socket and the actor only gets ``out``. Requires
the ``web`` extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import JSONResponse

from ..behavior import Behavior, Behaviors
from ..context import ActorContext
from ..patterns import AskTimeoutError
from ..ref import ActorRef
from ..signals import Signal, Terminated
from ..stream import (
    AbruptStageTerminationError,
    ActorSource,
    BufferOverflowError,
    Keep,
    OverflowStrategy,
    Sink,
    Source,
)
from ..system import ActorSystem

_log = logging.getLogger(__name__)


class Actors[T]:
    def __init__(self, guardian: Behavior[T], name: str = "app", **settings: Any) -> None:
        """``settings`` go to ``ActorSystem``: ``address``, ``cluster``, ``shutdown``,
        ``management``, ``mailbox``."""
        self._guardian = guardian
        self._name = name
        self._settings = settings
        self._system: ActorSystem[T] | None = None
        self._connections: dict[ActorRef[Any], asyncio.Task[None]] = {}  # wrapper: its send
        self._draining = False
        # Starlette copies the handlers when it builds the app, before the lifespan runs, so
        # they go to FastAPI(...) rather than being added at startup
        self.exception_handlers: dict[Any, Any] = {AskTimeoutError: _ask_timeout_to_504}

    @property
    def system(self) -> ActorSystem[T]:
        """The running system. Only valid between startup and shutdown."""
        if self._system is None:
            raise RuntimeError(
                "ActorSystem is not running: pass actors.lifespan to FastAPI(lifespan=...)"
            )
        return self._system

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None]:
        if AskTimeoutError not in app.exception_handlers:
            _log.warning(
                "an AskTimeoutError will be a 500, not a 504: pass "
                "FastAPI(lifespan=actors.lifespan, exception_handlers=actors.exception_handlers)"
            )
        try:
            async with ActorSystem(self._guardian, self._name, **self._settings) as system:
                self._system = system
                self._draining = False
                system.coordinated_shutdown.add_task(
                    "service-stop", "websockets-drain", self._drain
                )
                yield
        finally:
            self._system = None

    async def _drain(self) -> None:
        """The coordinated shutdown's service-stop phase, as Pekko HTTP's graceful termination:
        every open connection's actor stops behind the frames it has, and the phase waits, up to
        its timeout, for the frames told ``out`` to go out, while the outbox streams still run."""
        self._draining = True
        connections = list(self._connections.items())
        for flow, _ in connections:
            flow.tell(_CLOSE)
        if connections:
            await asyncio.wait([sending for _, sending in connections])

    def depends(self) -> ActorSystem[T]:
        """Default value for a route parameter: ``system: ActorSystem[Cmd] = actors.depends()``."""
        return cast(ActorSystem[T], Depends(lambda: self.system))

    async def websocket[In, Out](
        self,
        ws: WebSocket,
        props: Callable[[ActorRef[Out]], Behavior[In]],
        decode: Callable[[str], In],
        buffer_size: int = 16,
        overflow_strategy: OverflowStrategy = OverflowStrategy.fail,
    ) -> None:
        """Run one WebSocket connection through an actor, as Play's ``ActorFlow.actorRef``:
        accept the socket, spawn ``props(out)``, tell it ``decode(text)`` for every frame that
        comes in, and send every message it tells ``out`` as JSON. Returns when the connection is
        over. A closed socket stops the actor after the frames it already has; an actor that
        stops, or fails, closes the socket after the frames it told ``out``, and so does the
        system terminating, whose coordinated shutdown drains every connection in its
        service-stop phase. With ``run_by_actor_system_terminate=False`` there is no such phase:
        terminating is abrupt, frames not yet sent are dropped and the socket closes with 1001,
        as does a connection that arrives during the shutdown. A client more than
        ``buffer_size`` frames behind is closed with 1008 under ``OverflowStrategy.fail``, and a
        frame ``decode`` rejects closes it with 1003."""
        await ws.accept()
        root = self.system._root  # pyright: ignore[reportPrivateUsage]
        if self._draining or root.stopping:  # the system is shutting down
            await ws.close(code=1001, reason="going away")
            return
        source: Source[Out | _Done, ActorRef[Out | _Done]] = ActorSource.actor_ref(
            completion_matcher=lambda m: m is _DONE,
            failure_matcher=lambda _: None,
            buffer_size=buffer_size,
            overflow_strategy=overflow_strategy,
        )
        # the 1008 close needs watch_termination right on the source: only there does an
        # overflow fail ``terminated`` at once, without waiting for the next pull
        (out, terminated), outbox = (
            source.watch_termination(Keep.both).to_mat(Sink.queue(), Keep.both).run(self.system)
        )
        cell = root.spawn_child(_flow(props, out), None, None)
        # Play's sink watches the wrapper: however it ends, the frames told so far go out first
        cell.done.add_done_callback(lambda _: out.tell(_DONE))
        flow = cell.ref

        async def receive() -> None:
            with contextlib.suppress(WebSocketDisconnect):
                while ws.application_state == WebSocketState.CONNECTED:
                    text = await ws.receive_text()
                    try:
                        msg = decode(text)
                    except Exception:
                        if ws.application_state != WebSocketState.CONNECTED:
                            return  # the server closed meanwhile: the frame comes too late
                        _log.warning("closing a WebSocket: decode rejected a frame", exc_info=True)
                        await ws.close(code=1003, reason="unsupported data")
                        return
                    flow.tell(msg)

        async def send() -> None:
            # the client may be gone already; an overflow is handled through ``terminated``
            with contextlib.suppress(WebSocketDisconnect, BufferOverflowError):
                try:
                    async for frame in outbox:
                        await ws.send_json(frame)
                except AbruptStageTerminationError:  # terminated with no service-stop drain
                    await ws.close(code=1001, reason="going away")
                    return
                await ws.close()  # the actor stopped and its frames are sent

        receiving = asyncio.create_task(receive())
        sending = asyncio.create_task(send())
        self._connections[flow] = sending

        def overflowed() -> bool:
            return (
                terminated.done()
                and not terminated.cancelled()
                and isinstance(terminated.exception(), BufferOverflowError)
            )

        def overflow(_: object) -> None:  # Pekko fails downstream at once, even mid-send
            if overflowed():
                sending.cancel()

        terminated.add_done_callback(overflow)
        try:
            done, _ = await asyncio.wait({receiving, sending}, return_when=asyncio.FIRST_COMPLETED)
            if overflowed():
                if ws.application_state == WebSocketState.CONNECTED:  # else the client is gone
                    with contextlib.suppress(WebSocketDisconnect):
                        await ws.close(code=1008, reason="too slow")
                if receiving in done:  # it failed in the same tick: that is still an error
                    receiving.result()
            else:
                for task in done:
                    task.result()
        finally:  # everything before the await: in a cancelled scope (anyio) every await raises
            del self._connections[flow]
            flow.tell(_CLOSE)
            outbox.cancel()
            receiving.cancel()
            sending.cancel()
            await asyncio.wait({receiving, sending})


def _ask_timeout_to_504(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": f"actor ask timed out: {exc}"}, status_code=504)


class _Done:
    """Told to the outbox when the connection's wrapper has stopped."""


_DONE = _Done()


class _Close:
    """Told to the wrapper when the connection is over, as Play's wrapper gets Status.Success."""


_CLOSE = _Close()


def _flow[In, Out](
    props: Callable[[ActorRef[Out]], Behavior[In]], out: ActorRef[Out | _Done]
) -> Behavior[In | _Close]:
    """Play's wrapper actor: spawns the connection's actor, forwards frames to it, and stops it
    behind those frames (Play's PoisonPill) when the connection is over."""

    def setup(ctx: ActorContext[In | _Close]) -> Behavior[In | _Close]:
        actor = ctx.spawn(props(out), "flowActor")
        ctx.watch(actor)

        def on_message(msg: In | _Close) -> Behavior[In | _Close]:
            if isinstance(msg, _Close):
                ctx.stop(actor, after_pending=True)
            else:
                actor.tell(msg)
            return Behaviors.same

        def on_signal(_ctx: ActorContext[In | _Close], sig: Signal) -> Behavior[In | _Close]:
            if isinstance(sig, Terminated):
                return Behaviors.stopped()
            return Behaviors.unhandled

        return Behaviors.receive_message(on_message).receive_signal(on_signal)

    return Behaviors.setup(setup)
