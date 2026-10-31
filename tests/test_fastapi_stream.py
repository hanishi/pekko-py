"""ohra.ext.fastapi's actors.websocket on an ohra.stream graph: what the stream adds. The outbox
runs under the system's materializer, and the coordinated shutdown's service-stop phase drains
every connection before the system stops it, as Apache Pekko HTTP's graceful termination does."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402

from ohra import ActorRef, Behavior, Behaviors, CoordinatedShutdownSettings  # noqa: E402
from ohra.ext.fastapi import Actors  # noqa: E402

from .test_fastapi import FakeSocket, Frame, Said  # noqa: E402


def chatty(out: ActorRef[Frame]) -> Behavior[Said]:
    """Tells five frames for every frame in."""

    @Behaviors.receive_message
    def on_message(_msg: Said) -> Behavior[Said]:
        for i in range(5):
            out.tell({"n": str(i)})
        return Behaviors.same

    return on_message


def slow_socket(gate: asyncio.Event | None = None) -> tuple[FakeSocket, asyncio.Event]:
    """A socket whose sends take a few loop steps, as a real write does, and an event set once
    the first frame is out. Sends after the first wait for ``gate``, if given."""
    ws = FakeSocket()
    first_sent = asyncio.Event()
    send_json = ws.send_json

    async def slow_send(data: Any) -> None:
        if gate is not None and first_sent.is_set():
            await gate.wait()
        for _ in range(3):
            await asyncio.sleep(0)
        await send_json(data)
        first_sent.set()

    ws.send_json = slow_send
    return ws, first_sent


FRAMES = [{"n": str(i)} for i in range(5)]


async def test_terminating_the_system_flushes_the_frames_told_then_closes_with_1000():
    actors = Actors(Behaviors.ignore, "ws")
    ws, first_sent = slow_socket()
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        connection = asyncio.create_task(
            actors.websocket(ws.socket(), chatty, Said, buffer_size=10)
        )
        ws.incoming.put_nowait("go")
        await first_sent.wait()  # four frames are still in the outbox
    await asyncio.wait_for(connection, 1)
    assert ws.sent == FRAMES
    assert ws.closed == 1000


@dataclass(frozen=True)
class Stop:
    pass


async def test_a_guardian_that_stops_flushes_the_frames_told_then_closes_with_1000():
    actors = Actors(Behaviors.receive_message(lambda _: Behaviors.stopped()), "ws")
    ws, first_sent = slow_socket()
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        connection = asyncio.create_task(
            actors.websocket(ws.socket(), chatty, Said, buffer_size=10)
        )
        ws.incoming.put_nowait("go")
        await first_sent.wait()
        actors.system.tell(Stop())
        await asyncio.wait_for(connection, 1)
    assert ws.sent == FRAMES
    assert ws.closed == 1000


async def test_terminating_without_the_coordinated_shutdown_drops_the_frames_with_1001():
    direct = CoordinatedShutdownSettings(run_by_actor_system_terminate=False)
    actors = Actors(Behaviors.ignore, "ws", shutdown=direct)
    gate = asyncio.Event()
    ws, first_sent = slow_socket(gate)
    async with actors.lifespan(FastAPI(exception_handlers=actors.exception_handlers)):
        connection = asyncio.create_task(
            actors.websocket(ws.socket(), chatty, Said, buffer_size=10)
        )
        ws.incoming.put_nowait("go")
        await first_sent.wait()
    gate.set()  # the system is gone: the stream was aborted under the second send
    await asyncio.wait_for(connection, 1)
    assert ws.sent == FRAMES[:2]
    assert ws.closed == 1001
