"""Shopping carts on a Kubernetes cluster: examples/fastapi_carts.py with the carts spread over the
pods by cluster sharding. Every pod serves HTTP, any pod can reach any cart, and each cart lives on
one pod, persisted in Postgres.

Every response names two pods: `served_by` took the HTTP request, `hosted_on` runs the cart entity.
Through the Service the first changes from request to request while the second stays put; delete
the `hosted_on` pod and the next request recovers the cart, with its items, on another pod.

    # as examples/k8s/carts.yaml runs it
    uvicorn carts:app --app-dir examples/k8s --host 0.0.0.0 --port 8000

uvicorn owns SIGTERM: it stops taking requests, then the lifespan exits, which runs the coordinated
shutdown, so the pod hands its carts off and leaves the cluster after it has stopped serving.
DATABASE_URL is the Postgres database every pod reaches; sharding keeps one writer per cart, and
the store's revision check catches anything else. Carts are stored as JSON, so `SELECT
persistence_id, revision, convert_from(state_payload, 'UTF8') FROM durable_state` shows them.
"""

import json
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ohra import (  # noqa: E402
    ActorContext,
    ActorRef,
    Behavior,
    Behaviors,
    ClusterBootstrapSettings,
    ClusterSettings,
    ClusterSharding,
    Entity,
    EntityContext,
    EntityRef,
    EntityTypeKey,
    KubernetesApiServiceDiscovery,
    KubernetesApiSettings,
    Passivate,
    Management,
    ManagementSettings,
    SplitBrainResolverSettings,
    ask,
    remotable,
)
from ohra.ext.fastapi import Actors  # noqa: E402
from ohra.ext.postgres import PostgresStore  # noqa: E402

POD, IP = os.environ.get("POD_NAME", "local"), os.environ.get("POD_IP", "127.0.0.1")
IDLE_TIMEOUT = 30.0  # seconds without a request before a cart entity is passivated
store = PostgresStore[dict[str, int]](
    os.environ["DATABASE_URL"],
    codec=(lambda items: json.dumps(items).encode(), lambda raw: json.loads(raw)),
)  # opened by the lifespan

# --- cart protocol: remotable, since the HTTP pod and the cart's pod are usually different -------


@remotable
@dataclass(frozen=True)
class CartView:
    items: dict[str, int]
    hosted_on: str


@remotable
@dataclass(frozen=True)
class AddItem:
    sku: str
    qty: int
    reply_to: ActorRef[CartView]


@remotable
@dataclass(frozen=True)
class GetCart:
    reply_to: ActorRef[CartView]


@remotable
@dataclass(frozen=True)
class Checkout:
    reply_to: ActorRef[CartView]


type CartCmd = AddItem | GetCart | Checkout

CART = EntityTypeKey[CartCmd]("Cart")

# --- cart entity: items as the behavior's state, written before every reply ----------------------


def cart_entity(entity: EntityContext[CartCmd]) -> Behavior[CartCmd]:
    @Behaviors.persisted(store)
    def cart(items: dict[str, int]) -> Behavior[CartCmd]:
        @Behaviors.receive
        def on_message(ctx: ActorContext[CartCmd], msg: CartCmd) -> Behavior[CartCmd]:
            match msg:
                case AddItem(sku, qty, reply_to):
                    updated = {**items, sku: items.get(sku, 0) + qty}
                    # written, then replied
                    return cart(updated).then_reply(reply_to, CartView(updated, POD))
                case GetCart(reply_to):
                    reply_to.tell(CartView(items, POD))  # a read: nothing written
                    return Behaviors.same
                case Checkout(reply_to):
                    passivate = lambda: entity.shard.tell(Passivate(ctx.self))  # noqa: E731
                    return cart({}).then_reply(reply_to, CartView(items, POD)).then_run(passivate)

        return on_message

    # one read from Postgres, on whichever pod hosts it
    return cart.recover(entity.entity_id, empty={})


# --- the cluster node, owned by the app's lifespan -----------------------------------------------


@Behaviors.setup
def guardian(ctx: ActorContext[None]) -> Behavior[None]:
    # the same on every pod
    ClusterSharding(ctx.system).init(Entity(CART, cart_entity).with_passivate_after(IDLE_TIMEOUT))
    return Behaviors.empty


actors = Actors(
    guardian,
    "carts",
    address=(IP, 2551),
    cluster=ClusterSettings(downing=SplitBrainResolverSettings(stable_after=5.0)),
    management=ManagementSettings(
        hostname=IP,
        port=7626,
        bind_hostname="0.0.0.0",
        discovery=KubernetesApiServiceDiscovery(KubernetesApiSettings(use_raw_ip=True)),
        bootstrap=ClusterBootstrapSettings(
            service_name="carts", port_name="management", required_contact_point_nr=2
        ),
    ),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # the pool opens before the cluster node starts and closes after it stops
    async with store, actors.lifespan(app):
        # health checks for the probes, then the bootstrap
        await Management(actors.system).start()
        yield


app = FastAPI(title="carts", lifespan=lifespan, exception_handlers=actors.exception_handlers)

# --- HTTP ----------------------------------------------------------------------------------------


class Item(BaseModel):
    sku: str
    qty: int = 1


def ref(cart_id: str) -> EntityRef[CartCmd]:
    return ClusterSharding(actors.system).entity_ref_for(CART, cart_id)


async def existing(cart_id: str) -> EntityRef[CartCmd]:
    """Asking an unknown id would create an empty cart, so read-only routes check the store
    first."""
    loaded = await store.load(cart_id)
    if loaded is None or not loaded[0]:  # never written, or checked out
        raise HTTPException(404, "no such cart")
    return ref(cart_id)


def body(cart_id: str, view: CartView, key: str = "items") -> dict[str, Any]:
    return {"cart": cart_id, key: view.items, "served_by": POD, "hosted_on": view.hosted_on}


@app.post("/carts/{cart_id}/items")
async def add_item(cart_id: str, item: Item) -> dict[str, Any]:
    return body(
        cart_id, await ask(ref(cart_id), lambda r: AddItem(item.sku, item.qty, r), timeout=3)
    )


@app.get("/carts/{cart_id}")
async def get_cart(cart_id: str) -> dict[str, Any]:
    return body(cart_id, await ask(await existing(cart_id), GetCart, timeout=3))


@app.delete("/carts/{cart_id}")
async def checkout(cart_id: str) -> dict[str, Any]:
    return body(cart_id, await ask(await existing(cart_id), Checkout, timeout=3), "checked_out")
