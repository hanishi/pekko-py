"""ohra.ext.fastapi: an ActorSystem owned by the app's lifespan, reached through a typed
dependency."""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import importlib.util
import inspect
import logging
from dataclasses import dataclass
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi import FastAPI, WebSocket  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from ohra import ActorRef, ActorSystem, Behavior, Behaviors, PostStop, Signal  # noqa: E402
from ohra.ext.fastapi import Actors  # noqa: E402

from .util import eventually  # noqa: E402


@dataclass(frozen=True)
class Greet:
    name: str
    reply_to: ActorRef[str]


@dataclass(frozen=True)
class Stall:
    reply_to: ActorRef[str]


type Cmd = Greet | Stall


@Behaviors.receive_message
async def greeter(msg: Cmd) -> Behavior[Cmd]:
    match msg:
        case Greet(name, reply_to):
            reply_to.tell(f"hello {name}")
        case Stall():
            await asyncio.sleep(0.2)
    return Behaviors.same


def make_app() -> tuple[FastAPI, Actors[Cmd]]:
    actors = Actors(greeter, "web")
    app = FastAPI(lifespan=actors.lifespan, exception_handlers=actors.exception_handlers)

    @app.get("/greet/{name}")
    async def greet(name: str, system: ActorSystem[Cmd] = actors.depends()) -> dict[str, str]:
        return {"text": await system.ask(lambda r: Greet(name, r), timeout=1)}

    @app.get("/stall")
    async def stall(system: ActorSystem[Cmd] = actors.depends()) -> dict[str, str]:
        return {"text": await system.ask(Stall, timeout=0.01)}

    return app, actors


async def test_route_asks_the_system_through_the_dependency():
    app, actors = make_app()
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.get("/greet/pekko")
        assert r.status_code == 200
        assert r.json() == {"text": "hello pekko"}
        system = actors.system
        assert system.is_terminated is False
    assert system.is_terminated is True


@asynccontextmanager
async def served(app: FastAPI) -> AsyncGenerator[None]:
    """Run the lifespan through the app, as a server does: Starlette builds the app's
    middleware, exception handlers included, on that first call."""
    events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    sent: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def send(message: Any) -> None:
        await sent.put(message)

    scope = {"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}
    running = asyncio.create_task(app(scope, events.get, send))
    await events.put({"type": "lifespan.startup"})
    try:
        assert (await sent.get())["type"] == "lifespan.startup.complete"
        yield
    finally:
        await events.put({"type": "lifespan.shutdown"})
        await running


async def test_ask_timeout_is_a_504():
    app, _ = make_app()
    async with served(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.get("/stall")
    assert r.status_code == 504
    assert "timed out" in r.json()["detail"]


async def test_an_app_without_the_exception_handlers_is_warned_about(
    caplog: pytest.LogCaptureFixture,
):
    actors = Actors(greeter, "web")
    with caplog.at_level(logging.WARNING, logger="ohra.ext.fastapi"):
        async with served(FastAPI(lifespan=actors.lifespan)):
            pass
    assert any("exception_handlers=actors.exception_handlers" in r.message for r in caplog.records)


async def test_the_system_is_an_error_after_shutdown():
    app, actors = make_app()
    async with app.router.lifespan_context(app):
        pass
    with pytest.raises(RuntimeError):
        _ = actors.system


def test_system_before_startup_is_an_error():
    actors = Actors(greeter, "web")
    with pytest.raises(RuntimeError):
        _ = actors.system


async def test_system_settings_reach_the_actor_system():
    from ohra import CoordinatedShutdownSettings

    settings = CoordinatedShutdownSettings(default_phase_timeout=1.5)
    # address=, cluster=, management= pass the same way
    actors = Actors(greeter, "web", shutdown=settings)
    app = FastAPI(lifespan=actors.lifespan, exception_handlers=actors.exception_handlers)
    async with app.router.lifespan_context(app):
        assert actors.system.coordinated_shutdown.settings is settings


def _chatroom() -> Any:
    """examples/fastapi_chatroom.py, loaded from its file as a reader runs it."""
    path = Path(__file__).resolve().parents[1] / "examples" / "fastapi_chatroom.py"
    spec = importlib.util.spec_from_file_location("fastapi_chatroom", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chatroom_relays_messages_and_rejects_a_taken_name():
    from fastapi.testclient import TestClient

    with TestClient(_chatroom().app) as client:
        with client.websocket_connect("/ws/lobby/alice") as alice:
            with client.websocket_connect("/ws/lobby/bob") as bob:
                assert alice.receive_json() == {"type": "notice", "text": "bob joined"}
                alice.send_text("hello bob")
                said = {"type": "message", "from": "alice", "text": "hello bob"}
                assert bob.receive_json() == said
                assert alice.receive_json() == said
                with client.websocket_connect("/ws/lobby/alice") as impostor:
                    assert impostor.receive_json() == {
                        "type": "denied",
                        "text": "'alice' is taken",
                    }
            assert alice.receive_json() == {"type": "notice", "text": "bob left"}


def test_no_chatroom_actor_holds_the_socket():
    # only the route touches the socket: no message carries it, no behavior factory takes it
    chatroom = _chatroom()
    routes = {getattr(route, "endpoint", None) for route in chatroom.app.routes}
    for name, value in vars(chatroom).items():
        if dataclasses.is_dataclass(value):
            fields = [str(f.type) for f in dataclasses.fields(value)]
            assert not any("WebSocket" in t for t in fields), name
        elif inspect.isfunction(value) and value not in routes:
            params = [str(p.annotation) for p in inspect.signature(value).parameters.values()]
            assert not any("WebSocket" in t for t in params), name


class FakeSocket:
    """Just enough of a WebSocket for a route: frames in from a queue, frames out to ``sent``,
    or a send that fails with ``send_error``."""

    def __init__(
        self,
        send_error: BaseException | None = None,
        close_error: BaseException | None = None,
        frame_on_close: str | None = None,
        sends_block_after: int | None = None,
    ) -> None:
        from starlette.websockets import WebSocketState

        self.incoming: asyncio.Queue[str | BaseException] = asyncio.Queue()
        self.send_error = send_error
        self.close_error = close_error
        self.frame_on_close = frame_on_close  # a frame that arrives in the tick the server closes
        self.sends_block_after = sends_block_after  # a client that stops reading
        self.blocked = False
        self.sends = 0
        self.sent: list[Any] = []
        self.closed: int | None = None
        self.application_state = WebSocketState.CONNECTED

    def socket(self) -> WebSocket:
        """This fake where a route expects Starlette's WebSocket."""
        return cast(WebSocket, self)

    async def accept(self) -> None:
        pass

    async def send_json(self, data: Any) -> None:
        if self.sends_block_after is not None and self.sends >= self.sends_block_after:
            self.blocked = True
            await asyncio.Event().wait()  # the TCP window is full: the send never returns
        self.sends += 1
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(data)

    async def receive_text(self) -> str:
        from starlette.websockets import WebSocketState

        if self.application_state != WebSocketState.CONNECTED:  # as Starlette's
            raise RuntimeError('WebSocket is not connected. Need to call "accept" first.')
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        from starlette.websockets import WebSocketState

        if self.application_state != WebSocketState.CONNECTED:  # as Starlette's
            raise RuntimeError('Cannot call "send" once a close message has been sent.')
        self.application_state = WebSocketState.DISCONNECTED
        if self.frame_on_close is not None:
            self.incoming.put_nowait(self.frame_on_close)
        if self.close_error is not None:
            raise self.close_error
        self.closed = code


# --- actors.websocket, after Play's ActorFlow.actorRef -------------------------------------------


@dataclass(frozen=True)
class Said:
    text: str


type Frame = dict[str, str]


def echo(out: ActorRef[Frame], stopped: asyncio.Event | None = None) -> Behavior[Said]:
    @Behaviors.receive_message
    def on_message(msg: Said) -> Behavior[Said]:
        out.tell({"echo": msg.text})
        return Behaviors.same

    def on_signal(_ctx: Any, sig: Signal) -> Behavior[Said]:
        if sig is PostStop and stopped is not None:
            stopped.set()
        return Behaviors.same

    return on_message.receive_signal(on_signal)


def talker(*frames: Frame) -> Callable[[ActorRef[Frame]], Behavior[Said]]:
    """An actor that tells ``frames`` when it starts, then stops."""

    def props(out: ActorRef[Frame]) -> Behavior[Said]:
        @Behaviors.setup
        def setup(_ctx: Any) -> Behavior[Said]:
            for frame in frames:
                out.tell(frame)
            return Behaviors.stopped()

        return setup

    return props


async def test_websocket_relays_frames_both_ways():
    from fastapi import WebSocketDisconnect

    actors = Actors(Behaviors.ignore, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()
        connection = asyncio.create_task(actors.websocket(ws.socket(), echo, Said))
        ws.incoming.put_nowait("hi")
        await eventually(lambda: ws.sent == [{"echo": "hi"}])
        ws.incoming.put_nowait(WebSocketDisconnect(1000))
        await connection


async def test_closing_the_socket_stops_the_actor():
    from fastapi import WebSocketDisconnect

    actors = Actors(Behaviors.ignore, "ws")
    stopped = asyncio.Event()
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()
        connection = asyncio.create_task(
            actors.websocket(ws.socket(), lambda out: echo(out, stopped), Said)
        )
        ws.incoming.put_nowait(WebSocketDisconnect(1000))
        await connection
        await eventually(stopped.is_set)


async def test_an_actor_that_stops_closes_the_socket_after_its_frames():
    actors = Actors(Behaviors.ignore, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()
        await actors.websocket(ws.socket(), talker({"n": "1"}, {"n": "2"}), Said)
        assert ws.sent == [{"n": "1"}, {"n": "2"}]
        assert ws.closed == 1000


async def test_overflow_closes_with_1008():
    actors = Actors(Behaviors.ignore, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()
        frames = [{"n": str(i)} for i in range(3)]
        await actors.websocket(ws.socket(), talker(*frames), Said, buffer_size=1)
        assert ws.closed == 1008


async def test_a_receive_failure_in_the_tick_of_an_overflow_is_raised():
    actors = Actors(Behaviors.ignore, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()
        ws.incoming.put_nowait(RuntimeError("receive broke"))  # before the route sees the overflow
        frames = [{"n": str(i)} for i in range(3)]
        with pytest.raises(RuntimeError, match="receive broke"):
            await actors.websocket(ws.socket(), talker(*frames), Said, buffer_size=1)
        assert ws.closed == 1008


async def test_a_send_that_fails_ends_the_connection_and_stops_the_actor():
    from fastapi import WebSocketDisconnect

    actors = Actors(Behaviors.ignore, "ws")
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    errors: list[str] = []
    loop.set_exception_handler(lambda _, context: errors.append(str(context["message"])))
    try:
        async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
            ws = FakeSocket(WebSocketDisconnect(1006))  # the client's TCP dropped mid-send
            connection = asyncio.create_task(
                actors.websocket(ws.socket(), lambda out: echo(out, stopped), Said)
            )
            ws.incoming.put_nowait("hi")
            await connection
            await eventually(stopped.is_set)
            del connection
            gc.collect()
    finally:
        loop.set_exception_handler(None)
    assert errors == []  # no task died with an exception nobody retrieved


async def test_a_route_cancelled_by_the_server_still_stops_the_actor():
    import anyio

    actors = Actors(Behaviors.ignore, "ws")
    stopped = asyncio.Event()
    scope = anyio.CancelScope()  # how Starlette cancels a route: every later await raises too
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()

        async def route() -> None:
            with scope:
                await actors.websocket(ws.socket(), lambda out: echo(out, stopped), Said)

        connection = asyncio.create_task(route())
        ws.incoming.put_nowait("hi")
        await eventually(lambda: ws.sent == [{"echo": "hi"}])
        scope.cancel()
        await connection
        await eventually(stopped.is_set)


async def test_frames_the_actor_already_has_are_handled_when_the_client_closes():
    from fastapi import WebSocketDisconnect

    actors = Actors(Behaviors.ignore, "ws")
    gate = asyncio.Event()
    got: list[str] = []

    def slow(_out: ActorRef[Frame]) -> Behavior[Said]:
        @Behaviors.receive_message
        async def on_message(msg: Said) -> Behavior[Said]:
            await gate.wait()
            got.append(msg.text)
            return Behaviors.same

        return on_message

    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()
        for frame in ("1", "2", "3"):
            ws.incoming.put_nowait(frame)
        ws.incoming.put_nowait(WebSocketDisconnect(1000))
        await actors.websocket(ws.socket(), slow, Said)
        gate.set()
        await eventually(lambda: got == ["1", "2", "3"])


async def test_terminating_the_system_closes_open_connections():
    actors = Actors(Behaviors.ignore, "ws")
    ws = FakeSocket()
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        connection = asyncio.create_task(actors.websocket(ws.socket(), echo, Said))
        ws.incoming.put_nowait("hi")
        await eventually(lambda: ws.sent == [{"echo": "hi"}])
    await asyncio.wait_for(connection, 1)
    assert ws.closed == 1000


async def test_an_actor_that_fails_to_start_closes_the_socket():
    actors = Actors(Behaviors.ignore, "ws")

    def broken(_out: ActorRef[Frame]) -> Behavior[Said]:
        raise ValueError("no such room")

    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()
        await asyncio.wait_for(actors.websocket(ws.socket(), broken, Said), 1)
        assert ws.closed == 1000


async def test_a_close_after_the_client_left_is_not_an_error():
    from fastapi import WebSocketDisconnect

    actors = Actors(Behaviors.ignore, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket(close_error=WebSocketDisconnect(1006))
        await actors.websocket(ws.socket(), talker({"n": "1"}), Said)
        assert ws.sent == [{"n": "1"}]


async def test_a_frame_in_the_tick_the_server_closes_is_not_an_error():
    actors = Actors(Behaviors.ignore, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket(frame_on_close="late")
        await actors.websocket(ws.socket(), talker({"n": "1"}), Said)
        assert ws.closed == 1000


async def test_a_frame_decode_rejects_closes_with_1003_and_stops_the_actor():
    actors = Actors(Behaviors.ignore, "ws")
    stopped = asyncio.Event()

    def strict(text: str) -> Said:
        if not text:
            raise ValueError("empty frame")
        return Said(text)

    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket()
        ws.incoming.put_nowait("")
        await actors.websocket(ws.socket(), lambda out: echo(out, stopped), strict)
        assert ws.closed == 1003
        await eventually(stopped.is_set)


async def test_a_client_that_stops_reading_is_closed_with_1008():
    actors = Actors(Behaviors.ignore, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket(sends_block_after=1)
        websocket = actors.websocket(ws.socket(), echo, Said, buffer_size=1)
        connection = asyncio.create_task(websocket)
        ws.incoming.put_nowait("a")
        await eventually(lambda: ws.sent == [{"echo": "a"}])
        ws.incoming.put_nowait("b")
        await eventually(lambda: ws.blocked)  # the send of "b" never returns
        ws.incoming.put_nowait("c")  # buffered
        ws.incoming.put_nowait("d")  # overflows
        await asyncio.wait_for(connection, 1)
        assert ws.closed == 1008


async def test_a_connection_while_the_system_shuts_down_is_closed_with_1001():
    started, gate = asyncio.Event(), asyncio.Event()

    def slow_to_stop() -> Behavior[None]:
        async def on_signal(_ctx: Any, sig: Signal) -> Behavior[None]:
            if sig is PostStop:
                started.set()
                await gate.wait()
            return Behaviors.same

        return Behaviors.ignore.receive_signal(on_signal)

    @Behaviors.setup
    def guardian(ctx: Any) -> Behavior[None]:
        ctx.spawn(slow_to_stop(), "slow")
        return Behaviors.ignore

    actors = Actors(guardian, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        actors.system.terminate()
        await started.wait()  # the guardian is stopping, waiting on its child
        try:
            ws = FakeSocket()
            await actors.websocket(ws.socket(), echo, Said)
            assert ws.closed == 1001
        finally:
            gate.set()


async def test_an_overflow_after_the_client_left_is_not_an_error():
    from starlette.websockets import WebSocketDisconnect, WebSocketState

    class GoneSocket(FakeSocket):
        """The client is gone: the first send fails as Starlette's does, and in the next tick
        the actor overflows ``out``."""

        out: ActorRef[Frame] | None = None

        async def send_json(self, data: Any) -> None:
            self.application_state = WebSocketState.DISCONNECTED
            out = self.out
            assert out is not None
            asyncio.get_running_loop().call_soon(lambda: (out.tell({}), out.tell({})))
            raise WebSocketDisconnect(1006)

    actors = Actors(Behaviors.ignore, "ws")
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = GoneSocket()

        def props(out: ActorRef[Frame]) -> Behavior[Said]:
            ws.out = out
            return echo(out)

        ws.incoming.put_nowait("a")
        await asyncio.wait_for(actors.websocket(ws.socket(), props, Said, buffer_size=1), 1)


async def test_a_rejected_frame_in_the_tick_the_server_closes_is_not_an_error():
    actors = Actors(Behaviors.ignore, "ws")

    def strict(text: str) -> Said:
        if not text:
            raise ValueError("empty frame")
        return Said(text)

    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        ws = FakeSocket(frame_on_close="")
        await actors.websocket(ws.socket(), talker({"n": "1"}), strict)
        assert ws.closed == 1000
