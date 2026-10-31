# 9. FastAPI

## The problem

Most Python services are web apps, and the web framework already runs the event loop. The actor system has to start with the app and stop with it, and every route has to reach it.

## The idea

`ohra.ext.fastapi.Actors` ties an actor system to a FastAPI app. The system starts when the app starts and shuts down, through the coordinated shutdown of [chapter 2](02-actor-system.md), when the app stops, and each route receives it as a dependency.

## A first example

Chapter 1's cart behind two routes. It needs the `web` extra (`pip install 'ohra[web]'`):

```python
from fastapi import FastAPI

from cart import AddItem, CartCmd, GetCart, cart
from ohra import ActorSystem
from ohra.ext.fastapi import Actors

actors = Actors(cart({}), "shop")  # the actor system lives as long as the app
app = FastAPI(lifespan=actors.lifespan, exception_handlers=actors.exception_handlers)


@app.post("/items/{sku}")
async def add_item(
    sku: str, qty: int = 1, system: ActorSystem[CartCmd] = actors.depends()
) -> dict[str, int]:
    return await system.ask(lambda r: AddItem(sku, qty, r), timeout=1)


@app.get("/cart")
async def get_cart(system: ActorSystem[CartCmd] = actors.depends()) -> dict[str, int]:
    return await system.ask(GetCart, timeout=1)
```

Start it in one terminal, and call it from another:

```
$ uvicorn web_cart:app
```

```
$ curl -X POST localhost:8000/items/book
{"book":1}
$ curl -X POST 'localhost:8000/items/pen?qty=2'
{"book":1,"pen":2}
$ curl localhost:8000/cart
{"book":1,"pen":2}
```

- **`Actors(cart({}), "shop")`** takes the same arguments as `ActorSystem`.
- **`lifespan=actors.lifespan`** lets FastAPI start the system and stop it.
- **`exception_handlers=actors.exception_handlers`** maps an ask timeout to a 504. Starlette reads the handlers when it builds the app, before the lifespan runs, so they go to `FastAPI(...)`; the lifespan logs a warning when they are missing.
- **`actors.depends()`** hands each route the system, with its type: `system.ask(GetCart, ...)` returns a `dict[str, int]`, and pyright checks that against the route's return type.
- **Each request is a message.** Two requests at the same moment still reach the cart one at a time, as in chapter 1.
- **A cart that does not answer within the timeout** turns into a 504 response.

> **Coming from Apache Pekko**
>
> - FastAPI routes play the part of Pekko HTTP routes that ask actors. The integration itself has no Pekko counterpart.
> - Under uvicorn the server keeps its signals. Its shutdown runs `system.terminate()` through the lifespan, instead of the system's `run_on_signals` hook, as [appendix B](appendix-deviations.md) explains.

## How it works

This integration is not part of Apache Pekko. It is what a Python service over actors otherwise writes by hand.

`ohra.ext.fastapi.Actors` owns an `ActorSystem` for the life of a FastAPI app and hands it to routes as a typed dependency, so `ask` replies are type-checked all the way to the response. With `actors.exception_handlers` passed to `FastAPI`, an `AskTimeoutError` that escapes a route becomes a 504. Keyword arguments after the name go to `ActorSystem` (`address`, `cluster`, `shutdown`, `management`), so a cluster node can be owned by the app. Needs the `web` extra. To stop taking requests before a cluster node leaves, add a task to the `service-unbind` phase of the coordinated shutdown, and let uvicorn keep its signals: its shutdown should call `system.terminate()` rather than `run_on_signals` taking SIGTERM.

```python
from fastapi import FastAPI
from ohra.ext.fastapi import Actors

actors = Actors(counter(), "web")
app = FastAPI(lifespan=actors.lifespan, exception_handlers=actors.exception_handlers)


@app.get("/total")
async def total(system: ActorSystem[Command] = actors.depends()) -> dict[str, int]:
    return {"total": await system.ask(GetTotal, timeout=1)}
```

## WebSockets: the route owns the connection

A WebSocket route takes the same dependency. What it shouldn't do is hand the socket to an actor. Writing to a socket is an await, and an actor that awaits in its handler is held until the await returns: no other message gets through, not even a stop. One slow client is then enough to wedge its actor.

The route keeps the socket instead, and `actors.websocket` gives each connection an actor of its own, as Play Framework's `ActorFlow.actorRef` does:

```python
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket

from ohra import ActorRef, Behavior, Behaviors
from ohra.ext.fastapi import Actors

type Frame = dict[str, str]


@dataclass(frozen=True)
class Said:
    text: str


def echo(out: ActorRef[Frame]) -> Behavior[Said]:
    @Behaviors.receive_message
    def on_message(msg: Said) -> Behavior[Said]:
        out.tell({"echo": msg.text})  # a tell: the actor never waits on the socket
        return Behaviors.same

    return on_message


actors = Actors(Behaviors.ignore, "echo")
app = FastAPI(lifespan=actors.lifespan, exception_handlers=actors.exception_handlers)


@app.websocket("/ws")
async def chat(ws: WebSocket) -> None:
    await actors.websocket(ws, echo, Said)  # one actor per connection, for its lifetime
```

- **One actor per connection.** `actors.websocket(ws, echo, Said)` accepts the socket, spawns `echo(out)`, tells it `Said(text)` for every frame that comes in, and sends every message it tells `out` as JSON. It returns when the connection is over.
- **The two lives are tied.** A closed socket stops the actor, after the frames it already has. An actor that stops, by returning `Behaviors.stopped()` or by failing, closes the socket once the frames it told `out` are sent, and so does the system shutting down: `terminate()` drains every open connection in the coordinated shutdown's `service-stop` phase, up to the phase's timeout, and closes it with 1000.
- **`out` holds 16 frames** (`buffer_size`). A client that falls further behind is closed with code 1008 under the default `OverflowStrategy.fail`, even one that has stopped reading so that a send never returns. The drop strategies drop frames instead.
- **A connection that arrives while the system shuts down** is closed with 1001 (going away).
- **`decode` is the trust boundary.** A frame it rejects by raising closes the socket with 1003 (unsupported data), logged with its traceback on `ohra.ext.fastapi`, and stops the actor.

`examples/fastapi_chatroom.py` is the full version: a connection actor per socket that joins a room through a lobby, turns the client's frames into posts and room events into frames for `out`, and a room that watches its connection actors to tell the others when someone leaves.

### How it works: an `ActorSource` into a `Sink.queue`

`out` is the materialized ref of a stream from [chapter 13](13-streams.md): an `ActorSource` into a `Sink.queue`, which `actors.websocket` runs for each connection. Actors tell the ref, which never waits, and the route pulls from the queue with `async for` and sends. A route can wire it by hand, for a connection that doesn't fit one actor:

```python
import asyncio
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ohra import ActorRef, ActorSystem, Behavior, Behaviors
from ohra.ext.fastapi import Actors
from ohra.stream import ActorSource, BufferOverflowError, Keep, OverflowStrategy, Sink, Source

type Frame = dict[str, str]


@dataclass(frozen=True)
class Say:
    text: str
    outbox: ActorRef[Frame]


def echo() -> Behavior[Say]:
    @Behaviors.receive_message
    def on_message(msg: Say) -> Behavior[Say]:
        msg.outbox.tell({"echo": msg.text})  # a tell: the actor never waits on the socket
        return Behaviors.same

    return on_message


actors = Actors(echo(), "echo")
app = FastAPI(lifespan=actors.lifespan, exception_handlers=actors.exception_handlers)


@app.websocket("/ws")
async def chat(ws: WebSocket, system: ActorSystem[Say] = actors.depends()) -> None:
    await ws.accept()
    frames: Source[Frame, ActorRef[Frame]] = ActorSource.actor_ref(
        completion_matcher=lambda _: False,
        failure_matcher=lambda _: None,
        buffer_size=256,
        overflow_strategy=OverflowStrategy.fail,
    )
    outbox, queue = frames.to_mat(Sink.queue(), Keep.both).run(system)

    async def send() -> None:  # the route, not an actor, writes to the socket
        try:
            async for frame in queue:
                await ws.send_json(frame)
        except BufferOverflowError:
            await ws.close(code=1008, reason="too slow")

    sending = asyncio.create_task(send())
    try:
        while True:
            system.tell(Say(await ws.receive_text(), outbox))
    except WebSocketDisconnect:
        pass
    finally:
        queue.cancel()
        sending.cancel()
        await asyncio.wait([sending])
```

- **`outbox`** is an ordinary `ActorRef[Frame]`, so it travels in messages like any other ref. Each run of `frames` makes a new one.
- **The buffer holds 256 frames.** A client that falls that far behind fails the stream with `BufferOverflowError` under `OverflowStrategy.fail`, and the route closes the socket with code 1008. The loop sees the error at its next pull, so a send that never returns hides it; `actors.websocket` puts `watch_termination` right on the source, which sees the overflow at once, and cancels such a send. `drop_head`, `drop_tail` and `drop_buffer` drop frames instead, and with `buffer_size=0` a frame is dropped unless the loop is already waiting for one.
- **`completion_matcher` and `failure_matcher`** let an actor end the stream: a message the first matches ends it once the buffer is drained, and an exception the second returns fails it at once. Here neither matches, and the stream ends when the route cancels the queue.
- **Once the stream has ended,** by `queue.cancel()` or in any other way, tells to `outbox` are dead letters, logged at debug on `ohra.dead_letters`.
- **Annotate the source,** as `frames: Source[Frame, ActorRef[Frame]] = ...`: the frame type comes from the annotation, since the lambdas don't name it.

> **Coming from Apache Pekko**
>
> - `actors.websocket` follows Play Framework's `ActorFlow.actorRef`, which builds the same thing from Pekko streams: a wrapper actor that spawns yours and watches it, a `Source.actorRef` for `out`, and a `Sink.actorRef` for the frames that come in. Pekko itself has no counterpart.
> - `ActorSource.actorRef` comes from Pekko's stream-typed module, as does the rest of [chapter 13](13-streams.md). [Appendix B](appendix-deviations.md) lists where it differs.

---

[← Management and Kubernetes](08-management-and-kubernetes.md) · [Contents](README.md) · [numpy and shared memory →](10-numpy-and-shared-memory.md)
