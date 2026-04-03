"""
Predator-Prey Bayesian Optimisation tuner
==========================================
- Simulation  : CPU  (agent-based, hard to vectorise)
- Parallelism : multiprocessing.Pool — all CPU cores used per batch
- Optimiser   : BoTorch GP surrogate + qEI acquisition (GPU if available)

Run in Colab:
    !pip install botorch -q
    !git clone https://github.com/abidhaque/predator-prey-sim.git
    %cd predator-prey-sim
    !python3 tune_bo.py
"""

# ── 0. Install deps if missing ───────────────────────────────────────────────
import subprocess, sys
def pip(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])

try:
    import botorch
except ImportError:
    print("Installing botorch…")
    pip('botorch')

# ── 1. Imports ───────────────────────────────────────────────────────────────
import math, time, warnings
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.double
print(f"Device: {DEVICE}  |  CPU cores: {cpu_count()}")

# ── 2. Simulation ─────────────────────────────────────────────────────────────
W, H         = 1800, 1100
PREY_MAX_E   = 100.0
PRED_MAX_E   = 180.0
EAT_RADIUS   = 12.0
MAX_PREY     = 400
MAX_PRED     = 80
FRAMES       = 28000
SNAP_EVERY   = 120

def run_sim(P, seed=None):
    rng = np.random.default_rng(seed)

    killRadius      = P['killRadius']
    predSpeed       = P['predSpeed']
    preyFleeSpeed   = P['preyFleeSpeed']
    preySpeed       = P['preySpeed']
    predDetectRange = P['predDetectRange']
    predChaseRange  = P['predChaseRange']
    preyReproThresh = P['preyReproThreshold']
    predReproThresh = P['predReproThreshold']
    preyFoodEnergy  = P['preyFoodEnergy']
    predKillEnergy  = P['predPreyEnergy']
    preyDrain       = P['preyEnergyDrain']
    predDrain       = P['predEnergyDrain']
    foodRegenRate   = P['foodRegenRate']
    maxFood         = int(P['maxFood'])
    initPrey        = int(P['initPrey'])
    initPred        = int(P['initPred'])

    food_x = rng.uniform(0, W, maxFood).tolist()
    food_y = rng.uniform(0, H, maxFood).tolist()

    def vlimit(vx, vy, mx):
        m = math.hypot(vx, vy)
        return (vx/m*mx, vy/m*mx) if m > mx else (vx, vy)

    def vsetmag(vx, vy, mg):
        m = math.hypot(vx, vy) or 1.0
        return vx/m*mg, vy/m*mg

    prey = []
    for _ in range(initPrey):
        a = rng.uniform(0, 2*math.pi)
        prey.append({'x': rng.uniform(0,W), 'y': rng.uniform(0,H),
                     'vx': math.cos(a)*rng.uniform(1,2), 'vy': math.sin(a)*rng.uniform(1,2),
                     'e': rng.uniform(35,60), 'cd': int(rng.uniform(0,60))})

    pred = []
    for _ in range(initPred):
        pred.append({'x': rng.uniform(0,W), 'y': rng.uniform(0,H),
                     'vx': rng.uniform(-1,1), 'vy': rng.uniform(-1,1),
                     'e': rng.uniform(60,100), 'cd': int(rng.uniform(0,120))})

    food_timer = 0
    history = []

    for frame in range(FRAMES):
        food_timer += 1
        if food_timer >= foodRegenRate and len(food_x) < maxFood:
            food_x.append(rng.uniform(0,W))
            food_y.append(rng.uniform(0,H))
            food_timer = 0

        dead_prey = []; new_prey = []
        for i, p in enumerate(prey):
            p['cd'] = max(0, p['cd']-1)
            p['e'] -= preyDrain
            if p['e'] <= 0: dead_prey.append(i); continue
            for fi in range(len(food_x)-1, -1, -1):
                dx=food_x[fi]-p['x']; dy=food_y[fi]-p['y']
                if dx*dx+dy*dy < EAT_RADIUS*EAT_RADIUS:
                    p['e'] = min(PREY_MAX_E, p['e']+preyFoodEnergy)
                    food_x.pop(fi); food_y.pop(fi); break
            if p['e']>=preyReproThresh and p['cd']==0 and len(prey)+len(new_prey)<MAX_PREY:
                p['e'] -= preyReproThresh*0.48; p['cd'] = 180
                a = rng.uniform(0, 2*math.pi)
                new_prey.append({'x': p['x']+rng.uniform(-18,18), 'y': p['y']+rng.uniform(-18,18),
                                 'vx': math.cos(a)*rng.uniform(1,2), 'vy': math.sin(a)*rng.uniform(1,2),
                                 'e': preyReproThresh*0.28, 'cd': 60})
            fx=fy=0.0; fleeing=False
            for pr_ in pred:
                dx=p['x']-pr_['x']; dy=p['y']-pr_['y']; d=math.hypot(dx,dy)
                if 0<d<predDetectRange:
                    fleeing=True; s=1-d/predDetectRange
                    px2,py2=vsetmag(dx,dy,preyFleeSpeed); px2-=p['vx']; py2-=p['vy']
                    px2,py2=vlimit(px2,py2,0.35); fx+=px2*s*4; fy+=py2*s*4
            p['vx']+=fx; p['vy']+=fy
            p['vx'],p['vy']=vlimit(p['vx'],p['vy'],preyFleeSpeed if fleeing else preySpeed)
            p['x']=(p['x']+p['vx'])%W; p['y']=(p['y']+p['vy'])%H
        for i in reversed(dead_prey): prey.pop(i)
        prey.extend(new_prey)

        dead_pred = []; new_pred = []
        for i, pr_ in enumerate(pred):
            pr_['cd'] = max(0, pr_['cd']-1)
            pr_['e'] -= predDrain
            if pr_['e'] <= 0: dead_pred.append(i); continue
            fx=fy=0.0; target=None; targetDist=predChaseRange
            for p in prey:
                dx=p['x']-pr_['x']; dy=p['y']-pr_['y']; d=math.hypot(dx,dy)
                if d<targetDist: targetDist=d; target=p
            if target:
                dx=target['x']-pr_['x']; dy=target['y']-pr_['y']
                cx,cy=vsetmag(dx,dy,predSpeed); cx-=pr_['vx']; cy-=pr_['vy']
                cx,cy=vlimit(cx,cy,0.18); fx+=cx*1.4; fy+=cy*1.4
                if targetDist<killRadius:
                    target['e']=-999; pr_['e']=min(PRED_MAX_E, pr_['e']+predKillEnergy)
            else:
                fx+=rng.uniform(-0.2,0.2); fy+=rng.uniform(-0.2,0.2)
            if pr_['e']>=predReproThresh and pr_['cd']==0 and len(pred)+len(new_pred)<MAX_PRED:
                pr_['e']-=predReproThresh*0.52; pr_['cd']=400
                new_pred.append({'x': pr_['x']+rng.uniform(-25,25), 'y': pr_['y']+rng.uniform(-25,25),
                                 'vx': rng.uniform(-1,1), 'vy': rng.uniform(-1,1),
                                 'e': predReproThresh*0.3, 'cd': 200})
            pr_['vx']+=fx; pr_['vy']+=fy
            pr_['vx'],pr_['vy']=vlimit(pr_['vx'],pr_['vy'],predSpeed)
            pr_['x']=(pr_['x']+pr_['vx'])%W; pr_['y']=(pr_['y']+pr_['vy'])%H
        for i in reversed(dead_pred): pred.pop(i)
        pred.extend(new_pred)
        prey = [p for p in prey if p['e'] > -100]

        if frame % SNAP_EVERY == 0:
            history.append((len(prey), len(pred)))
        if not prey or not pred:
            break

    return history


def score_history(history):
    if len(history) < 60:
        return 0.0
    pr = [h[0] for h in history[30:]]
    pd = [h[1] for h in history[30:]]
    if pr[-1] == 0 or pd[-1] == 0:
        return 0.0

    pr_mean = sum(pr)/len(pr); pd_mean = sum(pd)/len(pd)
    pr_std  = math.sqrt(sum((x-pr_mean)**2 for x in pr)/len(pr))
    pd_std  = math.sqrt(sum((x-pd_mean)**2 for x in pd)/len(pd))

    peaks = sum(1 for i in range(1,len(pr)-1) if pr[i]>pr[i-1] and pr[i]>pr[i+1])
    peak_idxs = [i for i in range(1,len(pr)-1) if pr[i]>pr[i-1] and pr[i]>pr[i+1]]

    if len(peak_idxs) >= 3:
        ivs = [peak_idxs[i+1]-peak_idxs[i] for i in range(len(peak_idxs)-1)]
        iv_m = sum(ivs)/len(ivs)
        iv_s = math.sqrt(sum((x-iv_m)**2 for x in ivs)/len(ivs))
        regularity = 1.0 / (1.0 + iv_s/max(iv_m,1))
    else:
        regularity = 0.0

    cv = (pr_std/(pr_mean+1) + pd_std/(pd_mean+1)) / 2
    longevity     = min(len(history)/250, 1.0)
    min_viability = min(min(pr)/5.0, 1.0) * min(min(pd)/2.0, 1.0)
    pop_bonus     = min(pr_mean/40, 1.0) * min(pd_mean/6, 1.0)

    return cv * min(peaks/4, 1.0) * pop_bonus * longevity * min_viability * (0.4 + 0.6*regularity)


FIXED = {
    'preyFoodEnergy': 28, 'preyEnergyDrain': 0.08, 'preySpeed': 2.5,
    'predDetectRange': 90, 'predChaseRange': 130, 'initPrey': 100,
}

# Parameter space: (name, min, max, is_int)
PARAM_DEFS = [
    ('predSpeed',          3.5,  5.5,  False),
    ('preyFleeSpeed',      3.5,  5.5,  False),
    ('predEnergyDrain',    0.08, 0.30, False),
    ('predReproThreshold', 80,   200,  True),
    ('preyReproThreshold', 35,   90,   True),
    ('predPreyEnergy',     60,   140,  True),
    ('killRadius',         7,    18,   True),
    ('maxFood',            100,  250,  True),
    ('foodRegenRate',      6,    20,   True),
    ('initPred',           4,    10,   True),
]

BOUNDS = torch.tensor(
    [[lo for _,lo,_,_ in PARAM_DEFS],
     [hi for _,_,hi,_ in PARAM_DEFS]],
    dtype=DTYPE, device=DEVICE
)
DIM = len(PARAM_DEFS)

RUNS_PER_CFG = 4  # median over this many stochastic runs

def x_to_P(x_vec):
    """Convert a 1-D tensor/array from unit [0,1] space to parameter dict."""
    P = dict(FIXED)
    for i, (name, lo, hi, is_int) in enumerate(PARAM_DEFS):
        v = float(x_vec[i]) * (hi - lo) + lo
        P[name] = int(round(v)) if is_int else round(v, 3)
    return P

def P_to_x(P):
    """Convert param dict back to unit [0,1] vector."""
    return [(P[name]-lo)/(hi-lo) for name,lo,hi,_ in PARAM_DEFS]

def evaluate_P(P):
    seeds = [None]*RUNS_PER_CFG
    scores = sorted(score_history(run_sim(P, s)) for s in seeds)
    return scores[len(scores)//2]  # median

def _eval_worker(x_vec):
    return evaluate_P(x_to_P(x_vec))


# ── 3. Bayesian optimisation loop ────────────────────────────────────────────
INIT_SAMPLES   = 20   # random exploration before GP kicks in
BO_ITERATIONS  = 60   # GP-guided iterations
BATCH_SIZE     = 4    # candidates per BO step (parallelised on CPU)

def run_bo():
    print(f"\nInitial random exploration ({INIT_SAMPLES} samples)…")

    # Warm start: include the best-known config from prior search
    warm = {
        'predSpeed':4.4,'preyFleeSpeed':4.2,'predEnergyDrain':0.12,
        'predReproThreshold':130,'preyReproThreshold':50,'predPreyEnergy':100,
        'killRadius':10,'maxFood':160,'foodRegenRate':10,'initPred':6,
    }
    init_X = [P_to_x({**FIXED, **warm})]

    # Rest random
    rng = np.random.default_rng(42)
    while len(init_X) < INIT_SAMPLES:
        x = rng.uniform(0, 1, DIM).tolist()
        init_X.append(x)

    # Evaluate initial batch in parallel
    with Pool(cpu_count()) as pool:
        init_Y = pool.map(_eval_worker, init_X)

    train_X = torch.tensor(init_X, dtype=DTYPE, device=DEVICE)
    train_Y = torch.tensor(init_Y, dtype=DTYPE, device=DEVICE).unsqueeze(-1)

    best_idx = int(train_Y.argmax())
    best_score = float(train_Y[best_idx])
    best_x     = train_X[best_idx].tolist()
    print(f"  Init best score: {best_score:.4f}")

    # ── BO iterations ──
    print(f"\nBayesian optimisation ({BO_ITERATIONS} iters × batch {BATCH_SIZE})…")
    for it in range(BO_ITERATIONS):
        # Fit GP
        # Normalise Y for GP stability
        y_mean = train_Y.mean(); y_std = train_Y.std().clamp(min=1e-6)
        train_Y_norm = (train_Y - y_mean) / y_std

        gp  = SingleTaskGP(train_X, train_Y_norm).to(DEVICE)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Acquisition: qEI
        best_f = train_Y_norm.max()
        acqf   = qExpectedImprovement(model=gp, best_f=best_f)

        candidates, _ = optimize_acqf(
            acqf, bounds=torch.zeros(2, DIM, dtype=DTYPE, device=DEVICE).fill_(0).
                index_put_((torch.tensor([1]),), torch.ones(DIM, dtype=DTYPE, device=DEVICE)),
            q=BATCH_SIZE, num_restarts=10, raw_samples=128,
        )

        # Evaluate candidates in parallel on CPU
        cand_list = candidates.detach().cpu().tolist()
        with Pool(min(BATCH_SIZE, cpu_count())) as pool:
            new_Y = pool.map(_eval_worker, cand_list)

        new_X = candidates.detach().cpu().to(dtype=DTYPE, device=DEVICE)
        new_Y_t = torch.tensor(new_Y, dtype=DTYPE, device=DEVICE).unsqueeze(-1)

        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y_t])

        iter_best = max(new_Y)
        if iter_best > best_score:
            best_score = iter_best
            best_x     = cand_list[new_Y.index(iter_best)]

        if (it+1) % 5 == 0:
            print(f"  iter {it+1:3d}/{BO_ITERATIONS}  "
                  f"iter_best={iter_best:.4f}  global_best={best_score:.4f}  "
                  f"n_evals={len(train_Y)}")

    return best_x, best_score, train_X, train_Y


# ── 4. Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    best_x, best_score, all_X, all_Y = run_bo()
    elapsed = time.time() - t0

    best_P = x_to_P(best_x)
    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.0f}s   best score = {best_score:.4f}")
    print(f"{'='*55}")

    if best_score == 0:
        print("No viable configuration found — try increasing INIT_SAMPLES or BO_ITERATIONS.")
    else:
        print("\nBest parameters:")
        for name, lo, hi, is_int in PARAM_DEFS:
            print(f"  {name:<26} = {best_P[name]}")

        print("\nCharacterising best config (5 independent runs)…")
        for trial in range(5):
            h = run_sim(best_P)
            pr=[x[0] for x in h]; pd=[x[1] for x in h]
            peaks=sum(1 for i in range(1,len(pr)-1) if pr[i]>pr[i-1] and pr[i]>pr[i+1])
            s=score_history(h)
            print(f"  run {trial+1}: frames={len(h)*SNAP_EVERY:6d}  "
                  f"prey={min(pr):3d}-{max(pr):3d}  pred={min(pd):2d}-{max(pd):2d}  "
                  f"peaks={peaks:3d}  score={s:.3f}")

    # Print score progression for copy-paste into a plot if desired
    print("\nScore history (all evaluations, sorted by order):")
    for i, y in enumerate(all_Y.cpu().squeeze().tolist()):
        if i % 10 == 0: print(f"  eval {i:3d}: {y:.4f}")
