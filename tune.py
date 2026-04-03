"""
Predator-Prey parameter tuner — simulated annealing
Ports the JS agent-based sim to Python/numpy for speed,
then hill-climbs toward stable Lotka-Volterra oscillations.
"""

import numpy as np
import math, time, sys

# ── World constants ──
W, H         = 1800, 1100
PREY_MAX_E   = 100.0
PRED_MAX_E   = 180.0
EAT_RADIUS   = 12.0
MAX_PREY     = 400
MAX_PRED     = 80
FRAMES       = 28000
SNAP_EVERY   = 120
RUNS_PER_CFG = 4   # median over this many stochastic runs

rng = np.random.default_rng()

# ─────────────────────────────────────────────
# Core simulation (numpy-vectorised where easy)
# ─────────────────────────────────────────────
def run_sim(P):
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
    maxFood         = P['maxFood']
    initPrey        = P['initPrey']
    initPred        = P['initPred']

    # --- food ---
    food_x = rng.uniform(0, W, maxFood).tolist()
    food_y = rng.uniform(0, H, maxFood).tolist()

    # --- prey: list of dicts (compact) ---
    prey = []
    for _ in range(initPrey):
        a = rng.uniform(0, 2*math.pi)
        prey.append({
            'x': rng.uniform(0,W), 'y': rng.uniform(0,H),
            'vx': math.cos(a)*rng.uniform(1,2), 'vy': math.sin(a)*rng.uniform(1,2),
            'e': rng.uniform(35,60), 'cd': int(rng.uniform(0,60))
        })

    # --- predators ---
    pred = []
    for _ in range(initPred):
        pred.append({
            'x': rng.uniform(0,W), 'y': rng.uniform(0,H),
            'vx': rng.uniform(-1,1), 'vy': rng.uniform(-1,1),
            'e': rng.uniform(60,100), 'cd': int(rng.uniform(0,120))
        })

    food_timer = 0
    history = []

    def vlimit(vx, vy, mx):
        m = math.hypot(vx, vy)
        return (vx/m*mx, vy/m*mx) if m > mx else (vx, vy)

    def vsetmag(vx, vy, mg):
        m = math.hypot(vx, vy) or 1.0
        return vx/m*mg, vy/m*mg

    for frame in range(FRAMES):
        # food regen
        food_timer += 1
        if food_timer >= foodRegenRate and len(food_x) < maxFood:
            food_x.append(rng.uniform(0,W))
            food_y.append(rng.uniform(0,H))
            food_timer = 0

        # ── update prey ──
        dead_prey = []
        new_prey  = []
        for i, p in enumerate(prey):
            p['cd'] = max(0, p['cd'] - 1)
            p['e'] -= preyDrain
            if p['e'] <= 0:
                dead_prey.append(i)
                continue

            # eat food
            for fi in range(len(food_x)-1, -1, -1):
                dx = food_x[fi]-p['x']; dy = food_y[fi]-p['y']
                if dx*dx+dy*dy < EAT_RADIUS*EAT_RADIUS:
                    p['e'] = min(PREY_MAX_E, p['e'] + preyFoodEnergy)
                    food_x.pop(fi); food_y.pop(fi)
                    break

            # reproduce
            if p['e'] >= preyReproThresh and p['cd'] == 0 and len(prey)+len(new_prey) < MAX_PREY:
                p['e'] -= preyReproThresh * 0.48
                p['cd'] = 180
                a = rng.uniform(0, 2*math.pi)
                new_prey.append({
                    'x': p['x']+rng.uniform(-18,18), 'y': p['y']+rng.uniform(-18,18),
                    'vx': math.cos(a)*rng.uniform(1,2), 'vy': math.sin(a)*rng.uniform(1,2),
                    'e': preyReproThresh*0.28, 'cd': 60
                })

            # flee predators
            fx = fy = 0.0
            fleeing = False
            for pr_ in pred:
                dx = p['x']-pr_['x']; dy = p['y']-pr_['y']
                d = math.hypot(dx, dy)
                if 0 < d < predDetectRange:
                    fleeing = True
                    s = 1.0 - d/predDetectRange
                    px2, py2 = vsetmag(dx, dy, preyFleeSpeed)
                    px2 -= p['vx']; py2 -= p['vy']
                    px2, py2 = vlimit(px2, py2, 0.35)
                    fx += px2*s*4; fy += py2*s*4

            p['vx'] += fx; p['vy'] += fy
            spd = preyFleeSpeed if fleeing else preySpeed
            p['vx'], p['vy'] = vlimit(p['vx'], p['vy'], spd)
            p['x'] = (p['x']+p['vx']) % W
            p['y'] = (p['y']+p['vy']) % H

        for i in reversed(dead_prey):
            prey.pop(i)
        prey.extend(new_prey)

        # ── update predators ──
        dead_pred = []
        new_pred  = []
        for i, pr_ in enumerate(pred):
            pr_['cd'] = max(0, pr_['cd'] - 1)
            pr_['e'] -= predDrain
            if pr_['e'] <= 0:
                dead_pred.append(i)
                continue

            # find nearest prey
            fx = fy = 0.0
            target = None; targetDist = predChaseRange
            for p in prey:
                dx = p['x']-pr_['x']; dy = p['y']-pr_['y']
                d = math.hypot(dx, dy)
                if d < targetDist:
                    targetDist = d; target = p

            if target:
                dx = target['x']-pr_['x']; dy = target['y']-pr_['y']
                cx, cy = vsetmag(dx, dy, predSpeed)
                cx -= pr_['vx']; cy -= pr_['vy']
                cx, cy = vlimit(cx, cy, 0.18)
                fx += cx*1.4; fy += cy*1.4
                if targetDist < killRadius:
                    target['e'] = -999  # mark dead
                    pr_['e'] = min(PRED_MAX_E, pr_['e'] + predKillEnergy)
            else:
                fx += rng.uniform(-0.2, 0.2)
                fy += rng.uniform(-0.2, 0.2)

            # reproduce
            if pr_['e'] >= predReproThresh and pr_['cd'] == 0 and len(pred)+len(new_pred) < MAX_PRED:
                pr_['e'] -= predReproThresh * 0.52
                pr_['cd'] = 400
                new_pred.append({
                    'x': pr_['x']+rng.uniform(-25,25), 'y': pr_['y']+rng.uniform(-25,25),
                    'vx': rng.uniform(-1,1), 'vy': rng.uniform(-1,1),
                    'e': predReproThresh*0.3, 'cd': 200
                })

            pr_['vx'] += fx; pr_['vy'] += fy
            pr_['vx'], pr_['vy'] = vlimit(pr_['vx'], pr_['vy'], predSpeed)
            pr_['x'] = (pr_['x']+pr_['vx']) % W
            pr_['y'] = (pr_['y']+pr_['vy']) % H

        for i in reversed(dead_pred):
            pred.pop(i)
        pred.extend(new_pred)

        # remove killed prey (marked with e=-999)
        prey = [p for p in prey if p['e'] > -100]

        if frame % SNAP_EVERY == 0:
            history.append((len(prey), len(pred)))

        if not prey or not pred:
            break

    return history


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────
def score_history(history):
    if len(history) < 80:
        return 0.0
    pr_vals = [h[0] for h in history[40:]]
    pd_vals = [h[1] for h in history[40:]]
    if pr_vals[-1] == 0 or pd_vals[-1] == 0:
        return 0.0

    pr_min = min(pr_vals); pd_min = min(pd_vals)
    pr_mean = sum(pr_vals)/len(pr_vals)
    pd_mean = sum(pd_vals)/len(pd_vals)
    pr_std  = math.sqrt(sum((x-pr_mean)**2 for x in pr_vals)/len(pr_vals))
    pd_std  = math.sqrt(sum((x-pd_mean)**2 for x in pd_vals)/len(pd_vals))

    # Count prey peaks (oscillation cycles)
    peaks = sum(1 for i in range(1, len(pr_vals)-1)
                if pr_vals[i] > pr_vals[i-1] and pr_vals[i] > pr_vals[i+1])

    # Reward regularity: low variance of inter-peak intervals
    peak_idxs = [i for i in range(1, len(pr_vals)-1)
                 if pr_vals[i] > pr_vals[i-1] and pr_vals[i] > pr_vals[i+1]]
    if len(peak_idxs) >= 3:
        intervals = [peak_idxs[i+1]-peak_idxs[i] for i in range(len(peak_idxs)-1)]
        iv_mean = sum(intervals)/len(intervals)
        iv_std  = math.sqrt(sum((x-iv_mean)**2 for x in intervals)/len(intervals))
        regularity = 1.0 / (1.0 + iv_std/max(iv_mean,1))
    else:
        regularity = 0.0

    cv = (pr_std/(pr_mean+1) + pd_std/(pd_mean+1)) / 2
    longevity = min(len(history)/250, 1.0)

    # Soft penalties instead of hard floors — score degrades near extinction
    min_viability = min(pr_min/5.0, 1.0) * min(pd_min/2.0, 1.0)
    pop_bonus = min(pr_mean/40, 1.0) * min(pd_mean/6, 1.0)

    return cv * min(peaks/4, 1.0) * pop_bonus * longevity * min_viability * (0.4 + 0.6*regularity)


def evaluate(P):
    """Run RUNS_PER_CFG times, return median score."""
    scores = sorted(score_history(run_sim(P)) for _ in range(RUNS_PER_CFG))
    return scores[len(scores)//2]


# ─────────────────────────────────────────────
# Parameter space  (name, min, max, is_int)
# ─────────────────────────────────────────────
PARAMS = [
    ('predSpeed',          3.5,  5.5, False),
    ('preyFleeSpeed',      3.5,  5.5, False),
    ('predEnergyDrain',    0.08, 0.30, False),
    ('predReproThreshold', 80,  200,  True),
    ('preyReproThreshold', 35,   90,  True),
    ('predPreyEnergy',     60,  140,  True),
    ('killRadius',          7,   18,  True),
    ('maxFood',            100, 250,  True),
    ('foodRegenRate',        6,  20,  True),
    ('initPred',             4,  10,  True),
]

FIXED = {
    'preyFoodEnergy':  28,
    'preyEnergyDrain': 0.08,
    'preySpeed':       2.5,
    'predDetectRange': 90,
    'predChaseRange':  130,
    'initPrey':        100,
}

WARM_START = {
    # Best found by Node.js grid search — use as annealing seed
    'predSpeed':          4.4,
    'preyFleeSpeed':      4.2,
    'predEnergyDrain':    0.12,
    'predReproThreshold': 130,
    'preyReproThreshold': 50,
    'predPreyEnergy':     100,
    'killRadius':         10,
    'maxFood':            160,
    'foodRegenRate':      10,
    'initPred':           6,
}

def random_params():
    P = dict(FIXED)
    for name, lo, hi, is_int in PARAMS:
        v = rng.uniform(lo, hi)
        P[name] = int(round(v)) if is_int else round(float(v), 3)
    return P

def perturb(P, temperature):
    """Gaussian perturbation scaled by temperature."""
    Q = dict(P)
    # perturb 2-4 params at a time
    n = rng.integers(2, 5)
    chosen = rng.choice(len(PARAMS), n, replace=False)
    for idx in chosen:
        name, lo, hi, is_int = PARAMS[idx]
        scale = (hi - lo) * 0.15 * temperature
        v = Q[name] + rng.normal(0, scale)
        v = max(lo, min(hi, v))
        Q[name] = int(round(v)) if is_int else round(float(v), 3)
    return Q


# ─────────────────────────────────────────────
# Simulated annealing
# ─────────────────────────────────────────────
def anneal(max_iters=400, restarts=3):
    best_P    = None
    best_score = 0.0

    for restart in range(restarts):
        print(f"\n── Restart {restart+1}/{restarts} ──")

        # warm start: use best known, or warm_start seed, or random
        if best_P and restart > 0:
            current_P = perturb(best_P, 0.5)
        elif restart == 0:
            current_P = {**FIXED, **WARM_START}
        else:
            current_P = random_params()

        current_score = evaluate(current_P)
        local_best_P  = current_P
        local_best_s  = current_score

        T = 1.0  # temperature
        T_min = 0.05
        alpha = (T_min / T) ** (1.0 / max_iters)

        for it in range(max_iters):
            T *= alpha
            candidate = perturb(current_P, T)
            cs = evaluate(candidate)

            # Metropolis criterion
            if cs > current_score or rng.random() < math.exp((cs - current_score) / max(T, 1e-9)):
                current_P     = candidate
                current_score = cs

            if cs > local_best_s:
                local_best_s = cs
                local_best_P = candidate

            if cs > best_score:
                best_score = cs
                best_P     = candidate
                pr = [h[0] for h in run_sim(best_P)]
                pd_h = [h[1] for h in run_sim(best_P)]

            if (it+1) % 20 == 0:
                print(f"  iter {it+1:3d}  T={T:.3f}  local={local_best_s:.4f}  global_best={best_score:.4f}")
                sys.stdout.flush()

        print(f"  Restart best: {local_best_s:.4f}")

    return best_P, best_score


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting simulated annealing parameter search...")
    print(f"Each evaluation = {RUNS_PER_CFG} sim runs × {FRAMES} frames\n")
    t0 = time.time()

    best_P, best_score = anneal(max_iters=400, restarts=3)

    print(f"\n{'='*55}")
    print(f"Done in {time.time()-t0:.0f}s   best score = {best_score:.4f}")
    print(f"{'='*55}")

    if best_score == 0:
        print("No viable configuration found.")
    else:
        print("\nBest parameters:")
        for name, lo, hi, is_int in PARAMS:
            print(f"  {name:<26} = {best_P[name]}")

        # Characterise the best config
        print("\nCharacterising best config (5 runs)...")
        for trial in range(5):
            h = run_sim(best_P)
            pr = [x[0] for x in h]; pd = [x[1] for x in h]
            peaks = sum(1 for i in range(1,len(pr)-1) if pr[i]>pr[i-1] and pr[i]>pr[i+1])
            print(f"  run {trial+1}: frames={len(h)*SNAP_EVERY}  "
                  f"prey={min(pr)}-{max(pr)}  pred={min(pd)}-{max(pd)}  peaks={peaks}  "
                  f"score={score_history(h):.3f}")
