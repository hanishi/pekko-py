r"""Online learning with PyTorch behind FastAPI: will this shopper buy this product?

A predictor actor micro-batches purchase predictions into one PyTorch call per batch. A
shoppers actor simulates a shop: shoppers with hidden tastes see products, and buy or skip
them. Every outcome goes back to a learner actor, which trains its own copy of the model in
a thread and hands the predictor fresh weights, swapped in between batches. The model starts
untrained and learns only from outcomes. Shift the shoppers' tastes and watch it catch up.

    # demo client: learn, shift tastes, learn again
    uv run --extra web --extra torch python examples/fastapi_purchase_predictor.py
    uv run --extra web --extra torch uvicorn examples.fastapi_purchase_predictor:app --port 8000

then open http://localhost:8000 for the page: what shoppers really want next to what the model
predicts, accuracy over time, each shopper's top five with buy and skip, and the shift button.

    curl -X POST localhost:8000/predict -H 'content-type: application/json' \
        -d '{"shopper": 3, "product": 7}'
    curl -X POST localhost:8000/feedback -H 'content-type: application/json' \
        -d '{"prediction_id": 1, "bought": true}'
    curl localhost:8000/recommend/3
    curl -X POST localhost:8000/shift
"""

import asyncio
import copy
import itertools
import json
import sys
import urllib.request
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohra import (
    ActorContext,
    ActorRef,
    ActorSystem,
    Behavior,
    Behaviors,
    Failure,
    Success,
    Try,
    batching,
)
from ohra.ext.fastapi import Actors

N_SHOPPERS, N_PRODUCTS = 30, 24
BATCH_MAX_N, BATCH_MAX_WAIT = 64, 0.005  # prediction micro-batches
STEP_EVERY, TRAIN_STEPS, TRAIN_BATCH = 0.25, 20, 256  # learner cadence
REPLAY = 2_000  # the learner trains on the most recent outcomes only, so it can follow a shift
PENDING_LIMIT = 10_000  # predictions remembered while waiting for an outcome
VISIT_EVERY, VISITS_PER_TICK = 0.05, 8  # simulated traffic: 160 visits a second

torch.set_num_threads(1)  # a tiny model: one thread each for predicting and training is plenty

# --- model ---------------------------------------------------------------------------------------


class PurchaseModel(nn.Module):
    """An embedding per shopper and per product, and a small network on how well they match."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.shoppers = nn.Embedding(N_SHOPPERS, dim)
        self.products = nn.Embedding(N_PRODUCTS, dim)
        self.mlp = nn.Sequential(nn.Linear(3 * dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, shopper: Tensor, product: Tensor) -> Tensor:
        s, p = self.shoppers(shopper), self.products(product)
        return self.mlp(torch.cat([s, p, s * p], dim=-1)).squeeze(-1)  # one logit per pair


def predict(model: PurchaseModel, shoppers: list[int], products: list[int]) -> list[float]:
    """The chance of a purchase for each pair. Runs in a thread, on the predictor's copy."""
    with torch.no_grad():
        logits = model(torch.tensor(shoppers), torch.tensor(products))
        return torch.sigmoid(logits).numpy().tolist()


def train(
    model: PurchaseModel, optimizer: torch.optim.Optimizer, samples: list[tuple[int, int, float]]
) -> tuple[dict[str, Tensor], float]:
    """Gradient steps on random minibatches of recent outcomes. Runs in a thread, on the
    learner's copy, and returns a copy of the new weights for the predictor."""
    shoppers = torch.tensor([s for s, _, _ in samples])
    products = torch.tensor([p for _, p, _ in samples])
    bought = torch.tensor([b for _, _, b in samples])
    loss = torch.tensor(0.0)
    for _ in range(TRAIN_STEPS):
        i = torch.randint(len(samples), (TRAIN_BATCH,))
        logits = model(shoppers[i], products[i])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, bought[i])
        optimizer.zero_grad()
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()
    return copy.deepcopy(model.state_dict()), loss.item()


# --- predictor: batches requests, one PyTorch call per batch, reports predictions to the learner -


@dataclass(frozen=True)
class Predict:
    shopper: int
    product: int
    reply_to: ActorRef["Prediction"]


@dataclass(frozen=True)
class Prediction:
    prediction_id: int
    probability: float
    model_version: int


@dataclass(frozen=True)
class GetGrid:
    """Every shopper's chance of buying every product, for the page."""

    reply_to: ActorRef["Grid"]


@dataclass(frozen=True)
class Grid:
    chances: list[list[float]]  # shopper by product
    model_version: int


@dataclass(frozen=True)
class UseWeights:
    version: int
    state: dict[str, Tensor]  # a copy made for this message, never changed afterwards


type PredictorCmd = Predict | GetGrid | UseWeights


def predictor(learner: ActorRef["Predicted"]) -> Behavior[PredictorCmd]:
    @Behaviors.setup
    def setup(_ctx: ActorContext[PredictorCmd]) -> Behavior[PredictorCmd]:
        # The predictor's own copy of the model. Only this actor touches it, and only between
        # batches, so the learner can train its copy at the same time.
        model = PurchaseModel().eval()
        version = [0]  # the batching handler is a closure, so the version lives in a cell
        ids = itertools.count(1)

        async def handle(batch: list[PredictorCmd]) -> None:
            requests: list[Predict | GetGrid] = []
            for msg in batch:
                match msg:
                    case UseWeights(new_version, state):
                        await flush(requests)  # requests queued before the swap use the old one
                        requests = []
                        model.load_state_dict(state)
                        version[0] = new_version
                    case Predict() | GetGrid():
                        requests.append(msg)
            await flush(requests)

        async def flush(requests: list[Predict | GetGrid]) -> None:
            if not requests:
                return
            shoppers: list[int] = []
            products: list[int] = []
            for r in requests:
                match r:
                    case Predict(s, p):
                        shoppers.append(s)
                        products.append(p)
                    case GetGrid():
                        shoppers += [s for s in range(N_SHOPPERS) for _ in range(N_PRODUCTS)]
                        products += list(range(N_PRODUCTS)) * N_SHOPPERS
            # one call for the whole batch, in a thread; the actor waits for it, so the model
            # cannot be swapped in the middle
            chances = iter(await asyncio.to_thread(predict, model, shoppers, products))
            for r in requests:
                match r:
                    case Predict(s, p, reply_to):
                        pid, chance = next(ids), next(chances)
                        reply_to.tell(Prediction(pid, chance, version[0]))
                        learner.tell(Predicted(pid, s, p, chance))
                    case GetGrid(reply_to):
                        rows = [
                            [next(chances) for _ in range(N_PRODUCTS)] for _ in range(N_SHOPPERS)
                        ]
                        reply_to.tell(Grid(rows, version[0]))

        return batching(BATCH_MAX_N, BATCH_MAX_WAIT, handle)

    return setup


# --- learner: trains on outcomes, hands the predictor new weights --------------------------------


@dataclass(frozen=True)
class Predicted:
    prediction_id: int
    shopper: int
    product: int
    probability: float


@dataclass(frozen=True)
class Feedback:
    prediction_id: int
    bought: bool
    reply_to: ActorRef["FeedbackResult"]


@dataclass(frozen=True)
class FeedbackResult:
    accepted: bool  # False when the id is unknown or already used
    accuracy: float
    model_version: int


@dataclass(frozen=True)
class GetModel:
    reply_to: ActorRef["ModelInfo"]


@dataclass(frozen=True)
class ModelInfo:
    version: int
    outcomes: int
    accuracy: float  # over the most recent outcomes
    loss: float


@dataclass(frozen=True)
class _Step:
    pass


@dataclass(frozen=True)
class _Trained:
    state: dict[str, Tensor]
    loss: float


@dataclass(frozen=True)
class _TrainingFailed:
    exception: Exception


type LearnerCmd = Predicted | Feedback | GetModel | _Step | _Trained | _TrainingFailed


def learner(predictor_ref: Callable[[], ActorRef[UseWeights]]) -> Behavior[LearnerCmd]:
    @Behaviors.setup
    def setup(ctx: ActorContext[LearnerCmd]) -> Behavior[LearnerCmd]:
        ctx.timers.start_timer_with_fixed_delay("step", _Step(), STEP_EVERY)
        # The learner's own copy of the model and its optimizer. Only the training thread
        # touches them, and the stepping flag below makes sure there is never more than one.
        model = PurchaseModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Per-message state that is updated in place; everything else is in behavior parameters.
        pending: OrderedDict[int, Predicted] = OrderedDict()
        replay: deque[tuple[int, int, float]] = deque(maxlen=REPLAY)
        recent: deque[bool] = deque(maxlen=500)  # was each recent prediction right?

        def accuracy() -> float:
            return sum(recent) / len(recent) if recent else 0.0

        def learning(
            version: int, outcomes: int, loss: float, stepping: bool
        ) -> Behavior[LearnerCmd]:
            @Behaviors.receive_message
            def on_message(msg: LearnerCmd) -> Behavior[LearnerCmd]:
                match msg:
                    case Predicted(pid):
                        pending[pid] = msg
                        if len(pending) > PENDING_LIMIT:
                            pending.popitem(last=False)
                    case Feedback(pid, bought, reply_to):
                        found = pending.pop(pid, None)
                        if found is None:
                            reply_to.tell(FeedbackResult(False, accuracy(), version))
                            return Behaviors.same
                        replay.append((found.shopper, found.product, float(bought)))
                        recent.append((found.probability > 0.5) == bought)
                        reply_to.tell(FeedbackResult(True, accuracy(), version))
                        return learning(version, outcomes + 1, loss, stepping)
                    case _Step() if not stepping and len(replay) >= TRAIN_BATCH:
                        # a snapshot: the replay buffer keeps filling while the thread trains
                        work = asyncio.to_thread(train, model, optimizer, list(replay))

                        @ctx.pipe_to_self(work)
                        def _(result: Try[tuple[dict[str, Tensor], float]]) -> LearnerCmd:
                            match result:
                                case Success((state, new_loss)):
                                    return _Trained(state, new_loss)
                                case Failure(e):
                                    return _TrainingFailed(e)

                        return learning(version, outcomes, loss, stepping=True)
                    case _Trained(state, new_loss):
                        predictor_ref().tell(UseWeights(version + 1, state))
                        return learning(version + 1, outcomes, new_loss, stepping=False)
                    case _TrainingFailed(e):
                        ctx.log.error("training failed: %r", e)
                        return learning(version, outcomes, loss, stepping=False)
                    case GetModel(reply_to):
                        reply_to.tell(ModelInfo(version, outcomes, accuracy(), loss))
                    case _Step():
                        pass
                return Behaviors.same

            return on_message

        return learning(0, 0, float("nan"), stepping=False)

    return setup


# --- shoppers: a simulated shop whose hidden tastes the model has to learn -----------------------


type Matrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class World:
    tastes: Matrix  # shopper by taste
    traits: Matrix  # product by taste
    popularity: Matrix  # per product
    chance: Matrix  # shopper by product: the true chance of a purchase

    @staticmethod
    def of(tastes: Matrix, traits: Matrix, popularity: Matrix) -> "World":
        logits = 3 * tastes @ traits.T / np.sqrt(tastes.shape[1]) + popularity
        return World(tastes, traits, popularity, 1 / (1 + np.exp(-logits)))


def new_world(rng: np.random.Generator) -> World:
    return World.of(
        rng.normal(size=(N_SHOPPERS, 3)),
        rng.normal(size=(N_PRODUCTS, 3)),
        rng.normal(scale=0.5, size=N_PRODUCTS),
    )


def _names(ids: npt.NDArray[np.int64]) -> str:
    return ", ".join(str(i) for i in sorted(ids.tolist()))


def shifted(world: World, rng: np.random.Generator) -> tuple[World, str]:
    """Change what shoppers want: a trend, some shoppers' tastes, or a new season."""
    tastes, traits, popularity = world.tastes.copy(), world.traits.copy(), world.popularity.copy()
    match int(rng.integers(3)):
        case 0:
            hot = rng.choice(N_PRODUCTS, size=4, replace=False)
            popularity[hot] += 4
            what = f"a trend: products {_names(hot)} are suddenly popular"
        case 1:
            changed = rng.choice(N_SHOPPERS, size=10, replace=False)
            tastes[changed] = rng.normal(size=(len(changed), 3))
            what = f"shoppers {_names(changed)} changed their tastes"
        case _:
            traits = rng.normal(size=traits.shape)
            what = "a new season: every product is different now"
    return World.of(tastes, traits, popularity), what


@dataclass(frozen=True)
class SetRunning:
    running: bool


@dataclass(frozen=True)
class Shift:
    reply_to: ActorRef[str]


@dataclass(frozen=True)
class GetTruth:
    reply_to: ActorRef["Truth"]


@dataclass(frozen=True)
class Truth:
    chances: list[list[float]]
    running: bool
    last_shift: str


@dataclass(frozen=True)
class _Tick:
    pass


@dataclass(frozen=True)
class _Visited:
    prediction_id: int
    shopper: int
    product: int


@dataclass(frozen=True)
class _Lost:
    pass


type ShoppersCmd = SetRunning | Shift | GetTruth | _Tick | _Visited | _Lost


def shoppers(predictor: ActorRef[Predict], learner: ActorRef[Feedback]) -> Behavior[ShoppersCmd]:
    @Behaviors.setup
    def setup(ctx: ActorContext[ShoppersCmd]) -> Behavior[ShoppersCmd]:
        rng = np.random.default_rng(0)
        ctx.timers.start_timer_with_fixed_delay("visit", _Tick(), VISIT_EVERY)

        def visit(shopper: int, product: int) -> None:
            @ctx.ask(predictor, lambda r: Predict(shopper, product, r), timeout=5)
            def _(result: Try[Prediction]) -> ShoppersCmd:
                match result:
                    case Success(p):
                        return _Visited(p.prediction_id, shopper, product)
                    case Failure():
                        return _Lost()

        def shopping(world: World, running: bool, last_shift: str) -> Behavior[ShoppersCmd]:
            @Behaviors.receive_message
            def on_message(msg: ShoppersCmd) -> Behavior[ShoppersCmd]:
                match msg:
                    case _Tick() if running:
                        for _ in range(VISITS_PER_TICK):
                            visit(int(rng.integers(N_SHOPPERS)), int(rng.integers(N_PRODUCTS)))
                    case _Visited(pid, shopper, product):
                        # the shopper buys or not by the true chance, which the model never sees
                        bought = bool(rng.random() < world.chance[shopper, product])
                        learner.tell(Feedback(pid, bought, ctx.system.ignore_ref))
                    case SetRunning(on):
                        return shopping(world, on, last_shift)
                    case Shift(reply_to):
                        new, what = shifted(world, rng)
                        reply_to.tell(what)
                        return shopping(new, running, what)
                    case GetTruth(reply_to):
                        reply_to.tell(Truth(world.chance.tolist(), running, last_shift))
                    case _Tick() | _Lost():
                        pass
                return Behaviors.same

            return on_message

        return shopping(new_world(rng), running=True, last_shift="")

    return setup


# --- guardian: wires the three actors, exposes one protocol to HTTP ------------------------------


type Cmd = Predict | GetGrid | Feedback | GetModel | SetRunning | Shift | GetTruth


def service() -> Behavior[Cmd]:
    @Behaviors.setup
    def setup(ctx: ActorContext[Cmd]) -> Behavior[Cmd]:
        refs: dict[str, ActorRef[Any]] = {}
        # the learner and the predictor refer to each other
        lrn = ctx.spawn(learner(lambda: refs["predictor"]), "learner")
        refs["predictor"] = ctx.spawn(predictor(lrn), "predictor")
        pred: ActorRef[Predict | GetGrid] = refs["predictor"]
        shop = ctx.spawn(shoppers(pred, lrn), "shoppers")

        @Behaviors.receive_message
        def on_message(msg: Cmd) -> Behavior[Cmd]:
            match msg:
                case Predict() | GetGrid():
                    pred.tell(msg)
                case Feedback() | GetModel():
                    lrn.tell(msg)
                case SetRunning() | Shift() | GetTruth():
                    shop.tell(msg)
            return Behaviors.same

        return on_message

    return setup


# --- HTTP ----------------------------------------------------------------------------------------


actors = Actors(service(), "shop")
app = FastAPI(
    title="purchase predictor",
    lifespan=actors.lifespan,
    exception_handlers=actors.exception_handlers,
)


class Visit(BaseModel):
    shopper: int = Field(ge=0, lt=N_SHOPPERS)
    product: int = Field(ge=0, lt=N_PRODUCTS)


class Outcome(BaseModel):
    prediction_id: int
    bought: bool


class Simulation(BaseModel):
    running: bool


@app.post("/predict")
async def predict_route(
    body: Visit, system: ActorSystem[Cmd] = actors.depends()
) -> dict[str, Any]:
    p = await system.ask(lambda r: Predict(body.shopper, body.product, r), timeout=5)
    return {
        "prediction_id": p.prediction_id,
        "probability": round(p.probability, 4),
        "model_version": p.model_version,
    }


@app.post("/feedback")
async def feedback(body: Outcome, system: ActorSystem[Cmd] = actors.depends()) -> dict[str, Any]:
    r = await system.ask(lambda r: Feedback(body.prediction_id, body.bought, r), timeout=5)
    return {
        "accepted": r.accepted,
        "accuracy": round(r.accuracy, 3),
        "model_version": r.model_version,
    }


@app.get("/recommend/{shopper}")
async def recommend(shopper: int, system: ActorSystem[Cmd] = actors.depends()) -> dict[str, Any]:
    grid = await system.ask(GetGrid, timeout=5)
    row = grid.chances[shopper % N_SHOPPERS]
    top = sorted(range(N_PRODUCTS), key=lambda p: -row[p])[:5]
    return {
        "shopper": shopper % N_SHOPPERS,
        "model_version": grid.model_version,
        "top": [{"product": p, "probability": round(row[p], 4)} for p in top],
    }


@app.get("/model")
async def model_route(system: ActorSystem[Cmd] = actors.depends()) -> dict[str, Any]:
    m = await system.ask(GetModel, timeout=5)
    return {
        "version": m.version,
        "outcomes": m.outcomes,
        "accuracy": round(m.accuracy, 3),
        "loss": None if np.isnan(m.loss) else round(m.loss, 4),
    }


@app.get("/state")
async def state(system: ActorSystem[Cmd] = actors.depends()) -> dict[str, Any]:
    """Everything the page draws, in one call."""
    grid, truth, m = await asyncio.gather(
        system.ask(GetGrid, timeout=5),
        system.ask(GetTruth, timeout=5),
        system.ask(GetModel, timeout=5),
    )
    error = float(np.abs(np.array(grid.chances) - np.array(truth.chances)).mean())
    return {
        "truth": truth.chances,
        "predicted": grid.chances,
        "error": round(error, 4),  # how far the model's grid is from the truth, on average
        "version": m.version,
        "outcomes": m.outcomes,
        "accuracy": round(m.accuracy, 3),
        "running": truth.running,
        "last_shift": truth.last_shift,
    }


@app.post("/simulation")
async def simulation(
    body: Simulation, system: ActorSystem[Cmd] = actors.depends()
) -> dict[str, bool]:
    system.tell(SetRunning(body.running))
    return {"running": body.running}


@app.post("/shift")
async def shift(system: ActorSystem[Cmd] = actors.depends()) -> dict[str, str]:
    return {"shift": await system.ask(Shift, timeout=5)}


PAGE = """<!doctype html><meta charset=utf-8><title>purchase predictor</title>
<style>
body{font:15px system-ui;max-width:52em;margin:2em auto;color:#222;padding:0 1em}
section{border:1px solid #ddd;border-radius:6px;padding:1em;margin-bottom:1em}
h2{font-size:1em;margin:0 0 .6em}
.maps{display:flex;gap:1.5em;flex-wrap:wrap}
.maps figure{margin:0}.maps figcaption{font-size:.85em;color:#555;margin-bottom:.3em}
canvas{cursor:pointer;border:1px solid #ccc;image-rendering:pixelated}
.muted{color:#777}.shift{color:#a50}
.rec{display:grid;grid-template-columns:5.5em 1fr 3.5em auto;gap:.5em;align-items:center;margin:.25em 0}
.track{height:10px;background:#eee;border-radius:5px}.fill{height:10px;border-radius:5px;background:#2a7;transition:width .3s}
button{margin-right:.3em}
</style>
<section><h2>What shoppers really want, and what the model predicts</h2>
<div class=muted>Each row is a shopper, each column a product; darker is a likelier purchase. The model sees only who bought what. Click a row to pick a shopper.</div>
<div class=maps><figure><figcaption>the truth (hidden from the model)</figcaption><canvas id=truth></canvas></figure>
<figure><figcaption>the model's predictions</figcaption><canvas id=pred></canvas></figure></div>
<p id=summary class=muted>loading</p></section>
<section><h2>Learning over time</h2>
<svg id=chart width=100% height=120 viewBox="0 0 600 120" preserveAspectRatio=none style="background:#fafafa;border:1px solid #eee"></svg>
<div class=muted><span style="color:#2a7">━ accuracy</span> on recent outcomes, 40% to 100% · <span style="color:#a50">━ error</span>, the average gap between the two maps, 0 to 0.4 · one point a second</div></section>
<section><h2>Change what shoppers want</h2>
<button id=shift>shift tastes</button> <label><input type=checkbox id=running checked> simulated shoppers</label>
<p id=lastshift class="muted shift"></p></section>
<section><h2>Recommendations for shopper <span id=who>0</span></h2><div id=recs></div>
<div class=muted>Buy or skip to teach the model yourself; it learns from your clicks at the next training step.</div></section>
<script>
const $=id=>document.getElementById(id),post=(p,b)=>fetch(p,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(b??{})}).then(r=>r.json());
const C=10;let shopper=0,hist=[];
function draw(id,m){const c=$(id),x=c.getContext("2d");c.width=m[0].length*C;c.height=m.length*C;
 m.forEach((row,i)=>row.forEach((p,j)=>{x.fillStyle=`hsl(150,50%,${(96-p*70).toFixed(0)}%)`;x.fillRect(j*C,i*C,C,C)}));
 x.strokeStyle="#c40";x.lineWidth=2;x.strokeRect(1,shopper*C+1,c.width-2,C-2)}
for(const id of["truth","pred"])$(id).onclick=e=>{shopper=Math.floor(e.offsetY/C);tick()};
function chart(){const w=600,h=120,n=Math.max(hist.length-1,1),y=v=>(h-Math.min(Math.max(v,0),1)*h).toFixed(1),
 line=(f,col)=>`<polyline fill=none stroke=${col} stroke-width=2 points="${hist.map((d,i)=>`${(i/n*w).toFixed(1)},${y(f(d))}`).join(" ")}"/>`;
 $("chart").innerHTML=line(d=>(d.accuracy-.4)/.6,"#2a7")+line(d=>d.error/.4,"#a50")}
async function recs(){const r=await (await fetch("/recommend/"+shopper)).json();$("who").textContent=shopper;
 $("recs").innerHTML=r.top.map(t=>`<div class=rec><span>product ${t.product}</span><div class=track><div class=fill style="width:${(t.probability*100).toFixed(0)}%"></div></div><span>${(t.probability*100).toFixed(0)}%</span><span><button data-p=${t.product} data-b=1>buy</button><button data-p=${t.product} data-b=0>skip</button></span></div>`).join("");
 $("recs").querySelectorAll("button").forEach(b=>b.onclick=async()=>{const p=await post("/predict",{shopper,product:+b.dataset.p});await post("/feedback",{prediction_id:p.prediction_id,bought:b.dataset.b=="1"});b.parentNode.textContent=b.dataset.b=="1"?"bought":"skipped"})}
async function tick(){const s=await (await fetch("/state")).json();draw("truth",s.truth);draw("pred",s.predicted);
 $("summary").textContent=`model v${s.version} · ${s.outcomes} outcomes seen · accuracy ${(s.accuracy*100).toFixed(1)}% · error ${s.error.toFixed(3)}`;
 $("running").checked=s.running;if(s.last_shift)$("lastshift").textContent="last shift: "+s.last_shift;
 hist.push({accuracy:s.accuracy,error:s.error});if(hist.length>180)hist.shift();chart()}
$("shift").onclick=async()=>{const r=await post("/shift");$("lastshift").textContent="last shift: "+r.shift};
$("running").onchange=e=>post("/simulation",{running:e.target.checked});
tick();recs();setInterval(tick,1000);setInterval(recs,3000);
</script>"""  # noqa: E501 (a minified page)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


# --- demo client: watch the model learn, shift the tastes, watch it learn again ------------------


def _call(base: str, method: str, path: str) -> Any:
    req = urllib.request.Request(
        base + path, data=b"{}" if method == "POST" else None, method=method
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


async def demo(seconds: int = 40, shift_at: int = 20) -> None:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning"))  # any free port
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    try:
        for t in range(0, seconds + 1, 2):
            if t == shift_at:
                print("   ", (await asyncio.to_thread(_call, base, "POST", "/shift"))["shift"])
            s = await asyncio.to_thread(_call, base, "GET", "/state")
            print(
                f"{t:3}s  model v{s['version']:<4} accuracy {s['accuracy']:.3f}"
                f"  error {s['error']:.3f}"
            )
            await asyncio.sleep(2)
    finally:
        server.should_exit = True
        await task


if __name__ == "__main__":
    asyncio.run(demo())
