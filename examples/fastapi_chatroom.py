"""The chat room over WebSockets. ``actors.websocket`` owns each socket and gives it an actor,
as Play's ``ActorFlow.actorRef`` does: the actor gets the client's frames as messages and tells
the frames it sends to ``out``. Each room is an actor with one session child per user; rooms are
created on demand by the lobby. A room watches its clients' connection actors, so when a socket
closes and its actor stops, the room removes the session and tells the others.

    uv run --extra web uvicorn examples.fastapi_chatroom:app --port 8000

then open http://localhost:8000 in two browser tabs. Or run the file directly for a scripted
demo with three clients:

    uv run --extra web python examples/fastapi_chatroom.py
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohra import ActorContext, ActorRef, Behavior, Behaviors, Signal, StashBuffer, Terminated
from ohra.ext.fastapi import Actors

# --- room protocol (as in examples/chatroom.py, plus join/leave notices) ----------------------


@dataclass(frozen=True)
class GetSession:
    screen_name: str
    client: ActorRef["SessionEvent"]


@dataclass(frozen=True)
class PublishSessionMessage:
    screen_name: str
    message: str


type RoomCmd = GetSession | PublishSessionMessage


@dataclass(frozen=True)
class SessionGranted:
    handle: ActorRef["PostMessage"]


@dataclass(frozen=True)
class SessionDenied:
    reason: str


@dataclass(frozen=True)
class MessagePosted:
    screen_name: str
    message: str


@dataclass(frozen=True)
class Notice:
    text: str


type SessionEvent = SessionGranted | SessionDenied | MessagePosted | Notice


@dataclass(frozen=True)
class PostMessage:
    message: str


@dataclass(frozen=True)
class NotifyClient:
    event: MessagePosted | Notice


type SessionCmd = PostMessage | NotifyClient


@dataclass(frozen=True)
class Member:
    screen_name: str
    session: ActorRef[SessionCmd]
    client: ActorRef[SessionEvent]


def chat_room(members: tuple[Member, ...] = ()) -> Behavior[RoomCmd]:
    def broadcast(event: MessagePosted | Notice, to: tuple[Member, ...]) -> None:
        for m in to:
            m.session.tell(NotifyClient(event))

    @Behaviors.receive
    def on_message(ctx: ActorContext[RoomCmd], msg: RoomCmd) -> Behavior[RoomCmd]:
        match msg:
            case GetSession(screen_name, client):
                if any(m.screen_name == screen_name for m in members):
                    client.tell(SessionDenied(f"{screen_name!r} is taken"))
                    return Behaviors.same
                ses = ctx.spawn(session(ctx.self, screen_name, client), screen_name)
                ctx.watch(client)  # the connection actor stops when the socket closes
                client.tell(SessionGranted(ses))
                broadcast(Notice(f"{screen_name} joined"), members)
                return chat_room((*members, Member(screen_name, ses, client)))
            case PublishSessionMessage(screen_name, message):
                broadcast(MessagePosted(screen_name, message), members)
                return Behaviors.same

    def on_signal(ctx: ActorContext[RoomCmd], sig: Signal) -> Behavior[RoomCmd]:
        if isinstance(sig, Terminated):
            gone = next((m for m in members if m.client is sig.ref), None)
            if gone is None:
                return Behaviors.same
            ctx.stop(gone.session)
            rest = tuple(m for m in members if m is not gone)
            broadcast(Notice(f"{gone.screen_name} left"), rest)
            return chat_room(rest)
        return Behaviors.unhandled

    return on_message.receive_signal(on_signal)


def session(
    room: ActorRef[PublishSessionMessage], screen_name: str, client: ActorRef[SessionEvent]
) -> Behavior[SessionCmd]:
    @Behaviors.receive_message
    def on_message(msg: SessionCmd) -> Behavior[SessionCmd]:
        match msg:
            case PostMessage(message):
                room.tell(PublishSessionMessage(screen_name, message))
            case NotifyClient(event):
                client.tell(event)
        return Behaviors.same

    return on_message


# --- one actor per WebSocket connection, spawned by actors.websocket ---------------------------


@dataclass(frozen=True)
class Incoming:
    """A frame from the client."""

    text: str


type ConnectionMsg = Incoming | SessionEvent

type Frame = dict[str, str]


def connection(
    lobby: ActorRef["Join"], room: str, screen_name: str, out: ActorRef[Frame]
) -> Behavior[ConnectionMsg]:
    """Joins the room, then turns the client's frames into posts and room events into frames for
    ``out``. Telling ``out`` never waits, so the actor is free for its next message, a stop
    included, however slow the socket is. Stopping closes the socket."""

    def with_stash(buffer: StashBuffer[ConnectionMsg]) -> Behavior[ConnectionMsg]:
        @Behaviors.receive_message
        def joining(msg: ConnectionMsg) -> Behavior[ConnectionMsg]:
            match msg:
                case SessionGranted(handle):
                    return buffer.unstash_all(joined(handle))
                case SessionDenied(reason):
                    out.tell({"type": "denied", "text": reason})
                    return Behaviors.stopped()
                case _:  # a frame that came before the room answered
                    buffer.stash(msg)
                    return Behaviors.same

        def joined(handle: ActorRef[PostMessage]) -> Behavior[ConnectionMsg]:
            @Behaviors.receive_message
            def on_message(msg: ConnectionMsg) -> Behavior[ConnectionMsg]:
                match msg:
                    case Incoming(text):
                        handle.tell(PostMessage(text))
                    case MessagePosted(name, message):
                        out.tell({"type": "message", "from": name, "text": message})
                    case Notice(text):
                        out.tell({"type": "notice", "text": text})
                    case SessionGranted() | SessionDenied():
                        pass
                return Behaviors.same

            return on_message

        return joining

    @Behaviors.setup
    def setup(ctx: ActorContext[ConnectionMsg]) -> Behavior[ConnectionMsg]:
        lobby.tell(Join(room, screen_name, ctx.self))
        # frames from before the room answers: room for as many as the route buffers out
        return Behaviors.with_stash(256, with_stash)

    return setup


# --- lobby: rooms by name ------------------------------------------------------------------------


@dataclass(frozen=True)
class Join:
    room: str
    screen_name: str
    client: ActorRef[SessionEvent]


def lobby() -> Behavior[Join]:
    @Behaviors.receive
    def on_message(ctx: ActorContext[Join], msg: Join) -> Behavior[Join]:
        room: ActorRef[RoomCmd] = ctx.child(msg.room) or ctx.spawn(chat_room(), msg.room)
        room.tell(GetSession(msg.screen_name, msg.client))
        return Behaviors.same

    return on_message


# --- HTTP and WebSocket --------------------------------------------------------------------------


actors = Actors(lobby(), "chat")
app = FastAPI(
    title="chat",
    lifespan=actors.lifespan,
    exception_handlers=actors.exception_handlers,
)


@app.websocket("/ws/{room}/{screen_name}")
async def chat(ws: WebSocket, room: str, screen_name: str) -> None:
    await actors.websocket(
        ws,
        lambda out: connection(actors.system, room, screen_name, out),
        Incoming,
        buffer_size=256,
    )


PAGE = """<!doctype html><meta charset=utf-8><title>chat</title>
<style>
body{font:15px system-ui;max-width:40em;margin:2em auto;color:#222}
#join,#chat{border:1px solid #ddd;border-radius:6px;padding:1em}
#chat{display:none}
#log{height:22em;overflow:auto;border:1px solid #ddd;border-radius:4px;padding:.5em;margin:.6em 0;background:#fafafa}
#log div{margin:.15em 0}.n{color:#888;font-style:italic}.me b{color:#276}.d{color:#b33}
input{font:inherit;padding:.3em}button{font:inherit;padding:.3em .8em}
#m{width:78%}#status{color:#777;margin-left:.5em}
</style>
<form id=join><b>Join a room</b><br><br>
room <input id=room value=lobby size=12> &nbsp; name <input id=name placeholder="your name" size=12 autofocus>
<button>join</button><div class=n>Open this page in another tab with a different name to talk to yourself.</div></form>
<div id=chat><b id=title></b><span id=status></span><div id=log></div>
<form id=f><input id=m autocomplete=off placeholder="say something"> <button>send</button></form></div>
<script>
const $=id=>document.getElementById(id), log=$("log");
const add=(html,cls)=>{const d=document.createElement("div");d.innerHTML=html;if(cls)d.className=cls;log.append(d);log.scrollTop=1e9};
const esc=s=>s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
$("join").onsubmit=e=>{e.preventDefault();const room=$("room").value.trim()||"lobby",name=$("name").value.trim();if(!name)return;
 const ws=new WebSocket(`ws://${location.host}/ws/${encodeURIComponent(room)}/${encodeURIComponent(name)}`);
 $("join").style.display="none";$("chat").style.display="block";$("title").textContent=`#${room} as ${name}`;$("m").focus();
 ws.onopen=()=>$("status").textContent="connected";
 ws.onmessage=e=>{const m=JSON.parse(e.data);
  if(m.type==="message")add(`<b>${esc(m.from)}</b>: ${esc(m.text)}`,m.from===name?"me":"");
  else if(m.type==="denied")add(esc(m.text)+" — reload to pick another name","d");
  else add(esc(m.text),"n")};
 ws.onclose=()=>{$("status").textContent="disconnected";add("connection closed","n")};
 $("f").onsubmit=e=>{e.preventDefault();const t=$("m").value.trim();if(t&&ws.readyState===1){ws.send(t);$("m").value=""}}};
</script>"""  # noqa: E501 (a minified page)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


# --- scripted demo: three clients in one room ----------------------------------------------------


async def demo() -> None:
    import uvicorn
    import websockets

    server = uvicorn.Server(uvicorn.Config(app, port=8765, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    url = "ws://127.0.0.1:8765/ws/lobby/"

    async def show(name: str, ws: Any) -> None:
        async for raw in ws:
            m = json.loads(raw)
            print(
                f"[{name}] "
                + (f"{m['from']}: {m['text']}" if m["type"] == "message" else f"({m['text']})")
            )

    try:
        async with (
            websockets.connect(url + "alice") as alice,
            websockets.connect(url + "bob") as bob,
        ):
            readers = [
                asyncio.create_task(show("alice", alice)),
                asyncio.create_task(show("bob", bob)),
            ]
            await asyncio.sleep(0.05)
            await alice.send("hello bob")
            await bob.send("hi alice")
            await asyncio.sleep(0.05)
            async with websockets.connect(url + "alice") as impostor:  # name taken
                print("[alice#2]", json.loads(await impostor.recv())["text"])
            await asyncio.sleep(0.05)
            await bob.close()  # bob leaves: alice is told
            await asyncio.sleep(0.1)
            for r in readers:
                r.cancel()
    finally:
        server.should_exit = True
        await task


if __name__ == "__main__":
    asyncio.run(demo())
