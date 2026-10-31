r"""A FastAPI service backed by actors: one cart entity per cart id, created by the shard region on
its first message, passivated after a period of inactivity, and persisted in SQLite so a cart
survives passivation and a server restart. Request handlers ask the entity directly; the actor
system lives for the lifetime of the app. CARTS_DB sets the database path (default: carts.db next
to this file).

    uv run --extra web uvicorn examples.fastapi_carts:app --port 8000

    curl -X POST localhost:8000/carts/alice/items -H 'content-type: application/json' \
        -d '{"sku": "book", "qty": 2}'
    curl localhost:8000/carts/alice
    curl -X DELETE localhost:8000/carts/alice
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohra import (
    ActorContext,
    ActorRef,
    ActorSystem,
    Behavior,
    Behaviors,
    Entity,
    EntityContext,
    EntityRef,
    EntityTypeKey,
    GetShardRegionState,
    Passivate,
    Sharding,
    ShardingEnvelope,
    SqliteStore,
    ask,
)
from ohra.ext.fastapi import Actors

IDLE_TIMEOUT = 30.0  # seconds without a request before a cart entity is passivated
store = SqliteStore[dict[str, int]](
    os.environ.get("CARTS_DB", str(Path(__file__).with_name("carts.db")))
)

# --- cart protocol -------------------------------------------------------------------


@dataclass(frozen=True)
class AddItem:
    sku: str
    qty: int
    reply_to: ActorRef[dict[str, int]]


@dataclass(frozen=True)
class GetCart:
    reply_to: ActorRef[dict[str, int]]


@dataclass(frozen=True)
class Checkout:
    reply_to: ActorRef[dict[str, int]]


type CartCmd = AddItem | GetCart | Checkout

Cart = EntityTypeKey[CartCmd]("Cart")

# --- cart entity: items as the behavior's state, written before every reply -----------------


def cart_entity(entity: EntityContext[CartCmd]) -> Behavior[CartCmd]:
    @Behaviors.persisted(store)
    def cart(items: dict[str, int]) -> Behavior[CartCmd]:
        @Behaviors.receive
        def on_message(ctx: ActorContext[CartCmd], msg: CartCmd) -> Behavior[CartCmd]:
            match msg:
                case AddItem(sku, qty, reply_to):
                    updated = {**items, sku: items.get(sku, 0) + qty}
                    return cart(updated).then_reply(reply_to, updated)  # written, then replied
                case GetCart(reply_to):
                    reply_to.tell(items)  # a read: nothing written
                    return Behaviors.same
                case Checkout(reply_to):
                    # An empty cart is written, the caller gets the items, then the region stops
                    # this entity after anything already queued; a request that raced the checkout
                    # starts a fresh cart.
                    passivate = lambda: entity.shard.tell(Passivate(ctx.self))  # noqa: E731
                    return cart({}).then_reply(reply_to, items).then_run(passivate)

        return on_message

    return cart.recover(entity.entity_id, empty={})  # one read; the first message creates the row


# --- HTTP ------------------------------------------------------------------------------------

type RegionCmd = ShardingEnvelope[CartCmd] | GetShardRegionState

actors = Actors(
    Sharding.init(Entity(Cart, cart_entity).with_passivate_after(IDLE_TIMEOUT)), "carts"
)
app = FastAPI(
    title="carts",
    lifespan=actors.lifespan,
    exception_handlers=actors.exception_handlers,
)


class Item(BaseModel):
    sku: str
    qty: int = 1


async def existing(system: ActorSystem[RegionCmd], cart_id: str) -> EntityRef[CartCmd]:
    """Asking an unknown id would create an empty cart, so read-only routes check the store first.
    The store, not the set of live entities, is the truth: a passivated or restarted cart is still
    a cart."""
    loaded = await store.load(cart_id)
    # never written, or checked out (an empty cart is written then)
    if loaded is None or not loaded[0]:
        raise HTTPException(404, "no such cart")
    return Sharding.entity_ref_for(system, cart_id)


@app.post("/carts/{cart_id}/items")
async def add_item(
    cart_id: str, item: Item, system: ActorSystem[RegionCmd] = actors.depends()
) -> dict[str, Any]:
    items = await ask(
        Sharding.entity_ref_for(system, cart_id),
        lambda r: AddItem(item.sku, item.qty, r),
        timeout=2,
    )
    return {"cart": cart_id, "items": items}


@app.get("/carts/{cart_id}")
async def get_cart(
    cart_id: str, system: ActorSystem[RegionCmd] = actors.depends()
) -> dict[str, Any]:
    items = await ask(await existing(system, cart_id), GetCart, timeout=2)
    if not items:
        raise HTTPException(404, "no such cart")
    return {"cart": cart_id, "items": items}


@app.delete("/carts/{cart_id}")
async def checkout(
    cart_id: str, system: ActorSystem[RegionCmd] = actors.depends()
) -> dict[str, Any]:
    items = await ask(await existing(system, cart_id), Checkout, timeout=2)
    if not items:
        raise HTTPException(404, "no such cart")
    return {"cart": cart_id, "checked_out": items}


@app.get("/carts")
async def count(system: ActorSystem[RegionCmd] = actors.depends()) -> dict[str, int]:
    state = await system.ask(GetShardRegionState, timeout=2)
    return {"active_carts": len(state.entity_ids)}
