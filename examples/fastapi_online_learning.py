r"""Online learning behind FastAPI: a feedback loop between predictions and outcomes.

POST /predict returns a prediction with an id. POST /feedback brings back the true label for
that id. A predictor actor micro-batches predictions into one numpy call per batch. A learner
actor keeps the recent predictions, scores incoming feedback, accumulates a minibatch, and
every few hundred milliseconds takes gradient steps on it in a thread, then hands the new
model to the predictor, which swaps it in between batches. The model starts untrained and
learns only from feedback.

    # server + demo client
    uv run --extra web --extra numpy python examples/fastapi_online_learning.py
    uv run --extra web --extra numpy uvicorn examples.fastapi_online_learning:app --port 8000

then open http://localhost:8000 for a page to play with: predict, report the outcome, watch
the weights move.

    curl -X POST localhost:8000/predict -H 'content-type: application/json' \
        -d '{"features": [1.2, -0.3, 0.8, 2.1]}'
    curl -X POST localhost:8000/feedback -H 'content-type: application/json' \
        -d '{"prediction_id": 1, "label": 1}'
    curl localhost:8000/model
"""

import asyncio
import itertools
import json
import sys
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ohra import (
    ActorContext,
    ActorRef,
    ActorSystem,
    Behavior,
    Behaviors,
    Success,
    Try,
    batching,
)
from ohra.ext.fastapi import Actors

N_FEATURES = 4
BATCH_MAX_N, BATCH_MAX_WAIT = 64, 0.005  # prediction micro-batches
STEP_EVERY, MIN_STEP_SAMPLES = 0.25, 8  # learner cadence
PENDING_LIMIT = 10_000  # predictions remembered while waiting for feedback

# --- model ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    version: int
    weights: np.ndarray  # frozen view: writeable=False
    bias: float


def untrained() -> Model:
    w = np.zeros(N_FEATURES)
    w.flags.writeable = False
    return Model(0, w, 0.0)


def sgd_steps(
    model: Model, x: np.ndarray, y: np.ndarray, steps: int = 20, lr: float = 0.5
) -> Model:
    """Logistic regression gradient steps on a minibatch. Runs in a thread."""
    w, b = model.weights.copy(), model.bias
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        w -= lr * x.T @ (p - y) / len(y)
        b -= lr * float((p - y).mean())
    w.flags.writeable = False
    return Model(model.version + 1, w, b)


# --- predictor: batches requests, one matmul per batch, reports each prediction to the learner --


@dataclass(frozen=True)
class Predict:
    features: tuple[float, ...]
    reply_to: ActorRef["Prediction"]


@dataclass(frozen=True)
class Prediction:
    prediction_id: int
    probability: float
    model_version: int


@dataclass(frozen=True)
class UseModel:
    model: Model


type PredictorCmd = Predict | UseModel


def predictor(learner: ActorRef["Predicted"]) -> Behavior[PredictorCmd]:
    @Behaviors.setup
    def setup(_ctx: ActorContext[PredictorCmd]) -> Behavior[PredictorCmd]:
        # the batching handler is a closure, so the live model lives in a cell
        current = [untrained()]
        ids = itertools.count(1)

        def handle(batch: list[PredictorCmd]) -> None:
            requests: list[Predict] = []
            for msg in batch:
                match msg:
                    case UseModel(model):
                        flush(requests)  # requests queued before the swap use the old model
                        requests = []
                        current[0] = model
                    case Predict():
                        requests.append(msg)
            flush(requests)

        def flush(requests: list[Predict]) -> None:
            if not requests:
                return
            model = current[0]
            x = np.array([r.features for r in requests])
            # one call for the whole batch
            probs = 1 / (1 + np.exp(-(x @ model.weights + model.bias)))
            for r, p in zip(requests, probs.tolist()):
                pid = next(ids)
                r.reply_to.tell(Prediction(pid, float(p), model.version))
                learner.tell(Predicted(pid, r.features, float(p)))

        return batching(BATCH_MAX_N, BATCH_MAX_WAIT, handle)

    return setup


# --- learner: closes the loop --------------------------------------------------------------------


@dataclass(frozen=True)
class Predicted:
    prediction_id: int
    features: tuple[float, ...]
    probability: float


@dataclass(frozen=True)
class Feedback:
    prediction_id: int
    label: int
    reply_to: ActorRef["FeedbackResult"]


@dataclass(frozen=True)
class FeedbackResult:
    accepted: bool  # False when the id is unknown or already used
    online_accuracy: float
    model_version: int


@dataclass(frozen=True)
class GetModel:
    reply_to: ActorRef["ModelInfo"]


@dataclass(frozen=True)
class ModelInfo:
    version: int
    feedback_count: int
    online_accuracy: float
    weights: tuple[float, ...]


@dataclass(frozen=True)
class _Step:
    pass


@dataclass(frozen=True)
class _Stepped:
    result: Try[Model]


type LearnerCmd = Predicted | Feedback | GetModel | _Step | _Stepped


@dataclass(frozen=True)
class Stats:
    feedback: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.feedback if self.feedback else 0.0


def learner(predict_ref: Callable[[], ActorRef[UseModel]]) -> Behavior[LearnerCmd]:
    @Behaviors.setup
    def setup(ctx: ActorContext[LearnerCmd]) -> Behavior[LearnerCmd]:
        ctx.timers.start_timer_with_fixed_delay("step", _Step(), STEP_EVERY)
        # Per-message state that is updated in place; everything else is in behavior parameters.
        pending: OrderedDict[int, tuple[tuple[float, ...], float]] = OrderedDict()
        minibatch: list[tuple[tuple[float, ...], int]] = []

        def learning(model: Model, stats: Stats, stepping: bool) -> Behavior[LearnerCmd]:
            @Behaviors.receive_message
            def on_message(msg: LearnerCmd) -> Behavior[LearnerCmd]:
                match msg:
                    case Predicted(pid, features, prob):
                        pending[pid] = (features, prob)
                        if len(pending) > PENDING_LIMIT:
                            pending.popitem(last=False)
                    case Feedback(pid, label, reply_to):
                        found = pending.pop(pid, None)
                        if found is None:
                            reply_to.tell(FeedbackResult(False, stats.accuracy, model.version))
                            return Behaviors.same
                        features, prob = found
                        minibatch.append((features, label))
                        new_stats = Stats(
                            stats.feedback + 1, stats.correct + ((prob > 0.5) == bool(label))
                        )
                        reply_to.tell(FeedbackResult(True, new_stats.accuracy, model.version))
                        return learning(model, new_stats, stepping)
                    case _Step() if not stepping and len(minibatch) >= MIN_STEP_SAMPLES:
                        x = np.array([f for f, _ in minibatch])
                        y = np.array([float(label) for _, label in minibatch], dtype=np.float64)
                        minibatch.clear()
                        ctx.pipe_to_self(asyncio.to_thread(sgd_steps, model, x, y), _Stepped)
                        return learning(model, stats, stepping=True)
                    case _Stepped(result):
                        if isinstance(result, Success):
                            predict_ref().tell(UseModel(result.value))
                            return learning(result.value, stats, stepping=False)
                        ctx.log.error("step failed: %r", result.exception)
                        return learning(model, stats, stepping=False)
                    case GetModel(reply_to):
                        reply_to.tell(
                            ModelInfo(
                                model.version,
                                stats.feedback,
                                stats.accuracy,
                                tuple(model.weights.tolist()),
                            )
                        )
                    case _Step():
                        pass
                return Behaviors.same

            return on_message

        return learning(untrained(), Stats(), stepping=False)

    return setup


# --- guardian: wires predictor and learner, exposes one protocol to HTTP -------------------------


type Cmd = Predict | Feedback | GetModel


def service() -> Behavior[Cmd]:
    @Behaviors.setup
    def setup(ctx: ActorContext[Cmd]) -> Behavior[Cmd]:
        refs: dict[str, ActorRef[Any]] = {}
        # the two refer to each other
        lrn = ctx.spawn(learner(lambda: refs["predictor"]), "learner")
        refs["predictor"] = ctx.spawn(predictor(lrn), "predictor")
        pred: ActorRef[Predict] = refs["predictor"]

        @Behaviors.receive_message
        def on_message(msg: Cmd) -> Behavior[Cmd]:
            match msg:
                case Predict():
                    pred.tell(msg)
                case Feedback() | GetModel():
                    lrn.tell(msg)
            return Behaviors.same

        return on_message

    return setup


# --- HTTP ----------------------------------------------------------------------------------------


actors = Actors(service(), "online")
app = FastAPI(
    title="online learning",
    lifespan=actors.lifespan,
    exception_handlers=actors.exception_handlers,
)


class Features(BaseModel):
    features: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


class Outcome(BaseModel):
    prediction_id: int
    label: int


@app.post("/predict")
async def predict(body: Features, system: ActorSystem[Cmd] = actors.depends()) -> dict[str, Any]:
    p = await system.ask(lambda r: Predict(tuple(body.features), r), timeout=5)
    return {
        "prediction_id": p.prediction_id,
        "probability": p.probability,
        "label": int(p.probability > 0.5),
        "model_version": p.model_version,
    }


@app.post("/feedback")
async def feedback(body: Outcome, system: ActorSystem[Cmd] = actors.depends()) -> dict[str, Any]:
    r = await system.ask(lambda r: Feedback(body.prediction_id, body.label, r), timeout=5)
    return {
        "accepted": r.accepted,
        "online_accuracy": round(r.online_accuracy, 3),
        "model_version": r.model_version,
    }


@app.get("/model")
async def model(system: ActorSystem[Cmd] = actors.depends()) -> dict[str, Any]:
    m = await system.ask(GetModel, timeout=5)
    return {
        "version": m.version,
        "feedback_count": m.feedback_count,
        "online_accuracy": round(m.online_accuracy, 3),
        "weights": m.weights,
    }


PAGE = """<!doctype html><meta charset=utf-8><title>online learning</title>
<style>
body{font:15px system-ui;max-width:44em;margin:2em auto;color:#222}
section{border:1px solid #ddd;border-radius:6px;padding:1em;margin-bottom:1em}
h2{font-size:1em;margin:0 0 .6em}
input[type=number]{width:5.5em}
button{margin:.2em .2em .2em 0}
.bar{height:10px;background:#4a7;border-radius:5px;transition:width .3s}
.w{display:grid;grid-template-columns:3em 1fr 4em;gap:.4em;align-items:center;margin:.2em 0}
.w .track{position:relative;height:10px;background:#eee;border-radius:5px}
.w .fill{position:absolute;top:0;height:10px;border-radius:5px}
.muted{color:#777}
code{background:#f4f4f4;padding:0 .3em}
</style>
<section><h2>1. Predict</h2>
features <span id=feat></span> <button id=rnd>randomize</button> <button id=predict>predict</button>
<div id=pred class=muted>no prediction yet</div><div class=bar id=pbar style="width:0"></div></section>
<section><h2>2. Report what actually happened</h2>
<button id=y1 disabled>outcome was 1</button> <button id=y0 disabled>outcome was 0</button>
<div id=fb class=muted>predict first</div></section>
<section><h2>3. Model</h2><div id=model class=muted>loading</div><div id=weights></div></section>
<section><h2>4. Or let the hidden rule play</h2>
<div class=muted>The page knows a secret rule, <code>1.5·a − 2.0·b + 0.5·c + 1.0·d + noise &gt; 0</code>, that the service never sees.
Each round predicts on random features and reports the rule's outcome. Watch the weights converge to it.</div>
<button id=run>run 100 rounds</button> <span id=prog class=muted></span></section>
<script>
const $=id=>document.getElementById(id), post=(p,b)=>fetch(p,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(b)}).then(r=>r.json());
const inputs=[...Array(4)].map((_,i)=>{const e=document.createElement("input");e.type="number";e.step="0.1";$("feat").append(e,i<3?" ":"");return e});
const gauss=()=>{let u=1-Math.random(),v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)};
const randomize=()=>inputs.forEach(e=>e.value=gauss().toFixed(2)); randomize();
let last=null;
async function predict(){const f=inputs.map(e=>+e.value);last=await post("/predict",{features:f});
 $("pred").textContent=`p(label=1) = ${last.probability.toFixed(3)} → label ${last.probability>0.5?1:0}   (prediction #${last.prediction_id}, model v${last.model_version})`;
 $("pbar").style.width=(last.probability*100).toFixed(0)+"%";$("y1").disabled=$("y0").disabled=false;$("fb").textContent="which was it?";return last}
async function feedback(label){if(!last)return;const r=await post("/feedback",{prediction_id:last.prediction_id,label});
 $("fb").textContent=`${r.accepted?"recorded":"ignored"} · running accuracy ${(r.online_accuracy*100).toFixed(1)}% · model v${r.model_version}`;
 $("y1").disabled=$("y0").disabled=true;last=null;model()}
async function model(){const m=await (await fetch("/model")).json();
 $("model").textContent=`version ${m.version} · ${m.feedback_count} outcomes seen · running accuracy ${(m.online_accuracy*100).toFixed(1)}%`;
 const max=Math.max(1,...m.weights.map(Math.abs));
 $("weights").innerHTML=m.weights.map((w,i)=>`<div class=w><span>${"abcd"[i]}</span><div class=track><div class=fill style="left:${w<0?50-Math.abs(w)/max*50:50}%;width:${Math.abs(w)/max*50}%;background:${w<0?"#d66":"#4a7"}"></div></div><span>${w.toFixed(2)}</span></div>`).join("")}
async function run(n){$("run").disabled=true;for(let i=1;i<=n;i++){randomize();const f=inputs.map(e=>+e.value);const p=await post("/predict",{features:f});
 const y=1.5*f[0]-2*f[1]+0.5*f[2]+f[3]+0.5*gauss()>0?1:0;const r=await post("/feedback",{prediction_id:p.prediction_id,label:y});
 if(i%10==0){$("prog").textContent=`round ${i}: accuracy ${(r.online_accuracy*100).toFixed(1)}%, model v${r.model_version}`;model()}}
 $("run").disabled=false;model()}
$("rnd").onclick=randomize;$("predict").onclick=predict;$("y1").onclick=()=>feedback(1);$("y0").onclick=()=>feedback(0);$("run").onclick=()=>run(100);model();
</script>"""  # noqa: E501 (a minified page)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


# --- demo client: the world has a hidden rule; the service learns it from feedback alone ---------


def _call(base: str, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


async def demo(rounds: int = 600) -> None:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, port=8765, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    base = "http://127.0.0.1:8765"
    rng = np.random.default_rng(0)
    hidden_rule = np.array([1.5, -2.0, 0.5, 1.0])
    try:
        for i in range(1, rounds + 1):
            x = rng.normal(size=N_FEATURES)
            p = await asyncio.to_thread(_call, base, "POST", "/predict", {"features": x.tolist()})
            # the outcome the world reveals
            label = int(x @ hidden_rule + rng.normal(scale=0.5) > 0)
            r = await asyncio.to_thread(
                _call,
                base,
                "POST",
                "/feedback",
                {"prediction_id": p["prediction_id"], "label": label},
            )
            if i % 100 == 0:
                print(
                    f"round {i:4}  model v{r['model_version']:<3}  online accuracy "
                    f"{r['online_accuracy']:.3f}"
                )
        print(await asyncio.to_thread(_call, base, "GET", "/model"))
    finally:
        server.should_exit = True
        await task


if __name__ == "__main__":
    asyncio.run(demo())
