"""
GPU-vectorised Predator-Prey — Bayesian parameter search
=========================================================
Runs N_SIMS independent simulations IN PARALLEL on GPU.
Every agent's position/velocity/energy is a (N_SIMS, MAX_AGENTS) tensor.
All pairwise distances, forces, kills and food-eating are batched matrix ops.

Usage in Colab:
    !git clone https://github.com/abidhaque/predator-prey-sim.git
    %cd predator-prey-sim
    !pip install botorch -q
    !python3 tune_gpu.py
"""

import subprocess, sys, math, time, warnings
import numpy as np
import torch

def pip(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])
try:
    import botorch
except ImportError:
    print("Installing botorch…"); pip('botorch')

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
F32    = torch.float32
print(f"Device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── World / agent constants ────────────────────────────────────────────────────
W, H         = 1800.0, 1100.0
MAX_PREY     = 400
MAX_PRED     = 80
MAX_FOOD     = 300
PREY_MAX_E   = 100.0
PRED_MAX_E   = 180.0
EAT_RADIUS   = 12.0
FRAMES       = 24000
SNAP_EVERY   = 150

# How many parameter configurations to evaluate simultaneously
N_SIMS = 64 if DEVICE.type == 'cuda' else 8


# ══════════════════════════════════════════════════════════════════════════════
#  BatchSim  —  N_SIMS agent-based simulations running in parallel
# ══════════════════════════════════════════════════════════════════════════════
class BatchSim:
    def __init__(self, params_list, device=DEVICE):
        self.dev = device
        self.N   = len(params_list)

        def pt(key, dtype=F32):
            return torch.tensor([p[key] for p in params_list], dtype=dtype, device=device)

        # Per-sim tuneable parameters — all (N,) tensors
        self.predSpeed       = pt('predSpeed')
        self.preyFleeSpeed   = pt('preyFleeSpeed')
        self.predDrain       = pt('predEnergyDrain')
        self.predReproThresh = pt('predReproThreshold')
        self.preyReproThresh = pt('preyReproThreshold')
        self.predKillE       = pt('predPreyEnergy')
        self.killRadius      = pt('killRadius')
        self.maxFood         = pt('maxFood').int()
        self.foodRegenRate   = pt('foodRegenRate').int()
        self.initPred        = pt('initPred').int()

        # Fixed params
        self.preySpeed    = 2.5
        self.preyFoodE    = 28.0
        self.preyDrain    = 0.08
        self.detectRange  = 90.0
        self.chaseRange   = 130.0
        self.initPrey     = 100

        self._init_state()

    # ── Initialise all state tensors ──────────────────────────────────────────
    def _init_state(self):
        N, dev = self.N, self.dev

        def zeros(*s, dtype=F32): return torch.zeros(*s, dtype=dtype, device=dev)
        def rand(*s):             return torch.rand(*s, dtype=F32, device=dev)

        # ── Prey ──
        n = self.initPrey
        angles = rand(N, n) * 2 * math.pi
        speeds = rand(N, n) + 1.0

        self.prey_x  = _pad(rand(N, n) * W,            N, MAX_PREY, dev)
        self.prey_y  = _pad(rand(N, n) * H,            N, MAX_PREY, dev)
        self.prey_vx = _pad(torch.cos(angles) * speeds, N, MAX_PREY, dev)
        self.prey_vy = _pad(torch.sin(angles) * speeds, N, MAX_PREY, dev)
        self.prey_e  = _pad(rand(N, n) * 25 + 35,      N, MAX_PREY, dev)
        self.prey_cd = _pad((rand(N, n) * 60).int().float(), N, MAX_PREY, dev)
        self.prey_alive = zeros(N, MAX_PREY, dtype=torch.bool)
        self.prey_alive[:, :n] = True

        # ── Predators (variable initPred per sim) ──
        max_init_pred = int(self.initPred.max().item())
        self.pred_x  = zeros(N, MAX_PRED)
        self.pred_y  = zeros(N, MAX_PRED)
        self.pred_vx = zeros(N, MAX_PRED)
        self.pred_vy = zeros(N, MAX_PRED)
        self.pred_e  = zeros(N, MAX_PRED)
        self.pred_cd = zeros(N, MAX_PRED)
        self.pred_alive = zeros(N, MAX_PRED, dtype=torch.bool)

        for i, np_ in enumerate(self.initPred.tolist()):
            np_ = int(np_)
            self.pred_x[i, :np_]  = rand(np_) * W
            self.pred_y[i, :np_]  = rand(np_) * H
            self.pred_vx[i, :np_] = rand(np_) * 2 - 1
            self.pred_vy[i, :np_] = rand(np_) * 2 - 1
            self.pred_e[i, :np_]  = rand(np_) * 40 + 60
            self.pred_cd[i, :np_] = (rand(np_) * 120).int().float()
            self.pred_alive[i, :np_] = True

        # ── Food ──
        self.food_x = rand(N, MAX_FOOD) * W
        self.food_y = rand(N, MAX_FOOD) * H
        self.food_alive = zeros(N, MAX_FOOD, dtype=torch.bool)
        for i, mf in enumerate(self.maxFood.tolist()):
            self.food_alive[i, :int(mf)] = True

        self.food_timer = torch.zeros(N, dtype=torch.int32, device=dev)
        self.sim_alive  = torch.ones(N, dtype=torch.bool, device=dev)

    # ── Single simulation step (fully vectorised) ─────────────────────────────
    def step(self):
        N, dev = self.N, self.dev
        INF = 1e9

        pra = self.prey_alive  # (N, MAX_PREY) bool
        pda = self.pred_alive  # (N, MAX_PRED) bool
        fa  = self.food_alive  # (N, MAX_FOOD) bool

        # ── 1. Food regen ─────────────────────────────────────────────────────
        self.food_timer += 1
        need_spawn = (self.food_timer >= self.foodRegenRate) & (fa.sum(1) < self.maxFood)
        self.food_timer[need_spawn] = 0
        # Find first dead food slot for each sim needing spawn
        if need_spawn.any():
            dead_food_cumsum = (~fa).int().cumsum(dim=1)  # (N, MAX_FOOD)
            spawn_slot = (dead_food_cumsum == 1) & (~fa)   # first dead slot
            # Only for sims that need spawn
            spawn_slot &= need_spawn.unsqueeze(1)
            n_spawn = spawn_slot.sum().item()
            if n_spawn > 0:
                self.food_x[spawn_slot]     = torch.rand(n_spawn, device=dev) * W
                self.food_y[spawn_slot]     = torch.rand(n_spawn, device=dev) * H
                self.food_alive[spawn_slot] = True

        # ── 2. Prey energy drain & starvation ────────────────────────────────
        self.prey_e -= self.preyDrain * pra.float()
        self.prey_alive &= self.prey_e > 0

        # ── 3. Prey eat food (vectorised) ────────────────────────────────────
        if self.prey_alive.any() and self.food_alive.any():
            # (N, MAX_PREY, MAX_FOOD) distance matrix
            dpf = _dist2d(
                self.prey_x, self.prey_y,   # (N, MAX_PREY)
                self.food_x, self.food_y,   # (N, MAX_FOOD)
                self.prey_alive, self.food_alive
            )
            # For each food, give it to the closest prey
            min_prey_dist, min_prey_idx = dpf.min(dim=1)   # (N, MAX_FOOD)
            food_eaten = (min_prey_dist < EAT_RADIUS) & self.food_alive
            if food_eaten.any():
                energy_gain = food_eaten.float() * self.preyFoodE
                safe_idx = min_prey_idx.clamp(0, MAX_PREY-1)
                self.prey_e.scatter_add_(1, safe_idx, energy_gain)
                self.prey_e.clamp_(max=PREY_MAX_E)
                self.food_alive &= ~food_eaten

        # ── 4. Prey reproduction (births to first dead slot) ──────────────────
        self.prey_cd = (self.prey_cd - 1).clamp(min=0)
        can_repro = (
            (self.prey_e >= self.preyReproThresh.unsqueeze(1)) &
            (self.prey_cd == 0) & self.prey_alive
        )
        if can_repro.any():
            dead_prey_cumsum = (~self.prey_alive).int().cumsum(dim=1)  # (N, MAX_PREY)
            # Assign each reproducer a unique dead slot via cumsum of reproducers
            repro_order = can_repro.int().cumsum(dim=1) * can_repro.int()  # 1,2,3…
            for slot_rank in range(1, int(can_repro.sum(dim=1).max().item()) + 1):
                parent_mask = repro_order == slot_rank                # (N, MAX_PREY)
                birth_slot_mask = (dead_prey_cumsum == slot_rank) & (~self.prey_alive)  # (N, MAX_PREY)
                if not parent_mask.any() or not birth_slot_mask.any():
                    break
                # Parent positions (average over mask — only 1 True per row)
                p_count = parent_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                px = (self.prey_x * parent_mask.float()).sum(dim=1, keepdim=True) / p_count
                py = (self.prey_y * parent_mask.float()).sum(dim=1, keepdim=True) / p_count
                thresh = (self.preyReproThresh.unsqueeze(1) * parent_mask.float()).sum(dim=1, keepdim=True) / p_count

                # Deduct parent energy
                self.prey_e -= thresh.squeeze(1).unsqueeze(1) * 0.48 * parent_mask.float()
                self.prey_cd += 180 * parent_mask.float()

                # Place offspring
                b_count = birth_slot_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                angles = torch.rand(N, 1, device=dev) * 2 * math.pi
                spd    = torch.rand(N, 1, device=dev) + 1.0
                self.prey_x[birth_slot_mask]     = (px + torch.rand_like(px)*36-18).expand(N, MAX_PREY)[birth_slot_mask]
                self.prey_y[birth_slot_mask]     = (py + torch.rand_like(py)*36-18).expand(N, MAX_PREY)[birth_slot_mask]
                self.prey_vx[birth_slot_mask]    = (torch.cos(angles)*spd).expand(N, MAX_PREY)[birth_slot_mask]
                self.prey_vy[birth_slot_mask]    = (torch.sin(angles)*spd).expand(N, MAX_PREY)[birth_slot_mask]
                self.prey_e[birth_slot_mask]     = (thresh*0.28).expand(N, MAX_PREY)[birth_slot_mask]
                self.prey_cd[birth_slot_mask]    = 60
                self.prey_alive[birth_slot_mask] = True

        # ── 5. Prey flee predators ────────────────────────────────────────────
        if self.prey_alive.any() and self.pred_alive.any():
            # (N, MAX_PREY, MAX_PRED) vectors from pred→prey
            dx_pp, dy_pp, dist_pp = _vecdist2d(
                self.prey_x, self.prey_y,
                self.pred_x, self.pred_y,
                self.prey_alive, self.pred_alive
            )
            in_range = dist_pp < self.detectRange  # (N, MAX_PREY, MAX_PRED)
            strength = ((1.0 - dist_pp / self.detectRange) * in_range.float()).clamp(min=0)

            flee_speed = self.preyFleeSpeed.view(N, 1, 1)
            # Flee force: steer toward (away from pred direction)
            norm = dist_pp.clamp(min=1e-6)
            fx_flee = ((dx_pp / norm) * flee_speed * strength * 4).sum(dim=2)  # (N, MAX_PREY)
            fy_flee = ((dy_pp / norm) * flee_speed * strength * 4).sum(dim=2)

            steer_x = fx_flee - self.prey_vx
            steer_y = fy_flee - self.prey_vy
            steer_m = torch.sqrt(steer_x**2 + steer_y**2).clamp(min=1e-6)
            steer_s = (0.35 / steer_m).clamp(max=1.0)
            self.prey_vx += steer_x * steer_s * self.prey_alive.float()
            self.prey_vy += steer_y * steer_s * self.prey_alive.float()

            fleeing = in_range.any(dim=2)  # (N, MAX_PREY)
            max_spd = torch.where(fleeing,
                                  self.preyFleeSpeed.unsqueeze(1).expand(N, MAX_PREY),
                                  torch.full((N, MAX_PREY), self.preySpeed, device=dev))
        else:
            max_spd = torch.full((N, MAX_PREY), self.preySpeed, device=dev)

        spd_p = torch.sqrt(self.prey_vx**2 + self.prey_vy**2).clamp(min=1e-6)
        scale  = (max_spd / spd_p).clamp(max=1.0) * self.prey_alive.float()
        self.prey_vx *= scale;  self.prey_vy *= scale
        self.prey_x = (self.prey_x + self.prey_vx * self.prey_alive.float()) % W
        self.prey_y = (self.prey_y + self.prey_vy * self.prey_alive.float()) % H

        # ── 6. Predator drain & starvation ───────────────────────────────────
        self.pred_e -= self.predDrain.unsqueeze(1) * pda.float()
        self.pred_alive &= self.pred_e > 0

        # ── 7. Predator hunt + kill ───────────────────────────────────────────
        if self.pred_alive.any() and self.prey_alive.any():
            # (N, MAX_PRED, MAX_PREY) vectors from prey→pred direction (pred chases prey)
            dx_hp, dy_hp, dist_hp = _vecdist2d(
                self.pred_x, self.pred_y,
                self.prey_x, self.prey_y,
                self.pred_alive, self.prey_alive
            )
            # nearest prey for each predator
            min_dist, nearest_idx = dist_hp.min(dim=2)   # (N, MAX_PRED)
            has_target = (min_dist < self.chaseRange) & self.pred_alive

            # Chase steering
            safe_ni = nearest_idx.clamp(0, MAX_PREY-1)
            tx = self.prey_x.gather(1, safe_ni)   # (N, MAX_PRED)
            ty = self.prey_y.gather(1, safe_ni)
            cdx = tx - self.pred_x;  cdy = ty - self.pred_y
            cm  = torch.sqrt(cdx**2 + cdy**2).clamp(min=1e-6)
            ps  = self.predSpeed.unsqueeze(1)
            desired_vx = cdx / cm * ps
            desired_vy = cdy / cm * ps
            steer_x = desired_vx - self.pred_vx
            steer_y = desired_vy - self.pred_vy
            sm = torch.sqrt(steer_x**2 + steer_y**2).clamp(min=1e-6)
            ss = (0.18 / sm).clamp(max=1.0) * has_target.float()
            self.pred_vx += steer_x * ss * 1.4
            self.pred_vy += steer_y * ss * 1.4

            # Wander when no target
            no_tgt = (~has_target) & self.pred_alive
            self.pred_vx += (torch.rand_like(self.pred_vx)*0.4-0.2) * no_tgt.float()
            self.pred_vy += (torch.rand_like(self.pred_vy)*0.4-0.2) * no_tgt.float()

            # Speed limit
            spd_d = torch.sqrt(self.pred_vx**2 + self.pred_vy**2).clamp(min=1e-6)
            dscale = (ps / spd_d).clamp(max=1.0) * self.pred_alive.float()
            self.pred_vx *= dscale;  self.pred_vy *= dscale

            # Kills: predators within killRadius of their nearest prey
            kill_mask = (min_dist < self.killRadius.unsqueeze(1)) & has_target  # (N, MAX_PRED)
            if kill_mask.any():
                # Vectorised: scatter kill signals onto prey array
                kill_signal = torch.zeros(N, MAX_PREY, dtype=F32, device=dev)
                kill_signal.scatter_add_(1, safe_ni, kill_mask.float())
                killed = (kill_signal > 0) & self.prey_alive
                self.prey_alive &= ~killed
                # Energy reward for killing predators
                prey_killed_for_pred = killed.gather(1, safe_ni) & kill_mask
                self.pred_e += prey_killed_for_pred.float() * self.predKillE.unsqueeze(1)
                self.pred_e.clamp_(max=PRED_MAX_E)

        self.pred_x = (self.pred_x + self.pred_vx * self.pred_alive.float()) % W
        self.pred_y = (self.pred_y + self.pred_vy * self.pred_alive.float()) % H

        # ── 8. Predator reproduction ──────────────────────────────────────────
        self.pred_cd = (self.pred_cd - 1).clamp(min=0)
        can_repro_pred = (
            (self.pred_e >= self.predReproThresh.unsqueeze(1)) &
            (self.pred_cd == 0) & self.pred_alive
        )
        if can_repro_pred.any():
            dead_pred_cs = (~self.pred_alive).int().cumsum(dim=1)
            repro_order  = can_repro_pred.int().cumsum(dim=1) * can_repro_pred.int()
            for slot_rank in range(1, int(can_repro_pred.sum(dim=1).max().item()) + 1):
                parent_mask = repro_order == slot_rank
                birth_slot  = (dead_pred_cs == slot_rank) & (~self.pred_alive)
                if not parent_mask.any() or not birth_slot.any(): break
                p_count = parent_mask.float().sum(1, keepdim=True).clamp(min=1)
                px = (self.pred_x * parent_mask.float()).sum(1, keepdim=True) / p_count
                py = (self.pred_y * parent_mask.float()).sum(1, keepdim=True) / p_count
                thresh = (self.predReproThresh.unsqueeze(1) * parent_mask.float()).sum(1, keepdim=True) / p_count
                self.pred_e  -= thresh.squeeze(1).unsqueeze(1) * 0.52 * parent_mask.float()
                self.pred_cd += 400 * parent_mask.float()
                self.pred_x[birth_slot]     = (px + torch.rand_like(px)*50-25).expand(N, MAX_PRED)[birth_slot]
                self.pred_y[birth_slot]     = (py + torch.rand_like(py)*50-25).expand(N, MAX_PRED)[birth_slot]
                self.pred_vx[birth_slot]    = (torch.rand(N,1,device=dev)*2-1).expand(N,MAX_PRED)[birth_slot]
                self.pred_vy[birth_slot]    = (torch.rand(N,1,device=dev)*2-1).expand(N,MAX_PRED)[birth_slot]
                self.pred_e[birth_slot]     = (thresh*0.3).expand(N, MAX_PRED)[birth_slot]
                self.pred_cd[birth_slot]    = 200
                self.pred_alive[birth_slot] = True

        # Track which sims still have both populations
        self.sim_alive = self.prey_alive.any(1) & self.pred_alive.any(1)

    # ── Run all sims ──────────────────────────────────────────────────────────
    def run(self):
        history = []
        for frame in range(FRAMES):
            self.step()
            if frame % SNAP_EVERY == 0:
                history.append(torch.stack([
                    self.prey_alive.sum(1),
                    self.pred_alive.sum(1)
                ], dim=1).cpu())  # (N, 2)
            if not self.sim_alive.any():
                break
        if not history:
            return torch.zeros(self.N, 1, 2, dtype=torch.int32)
        return torch.stack(history, dim=1)  # (N, T, 2)


# ── Tensor helpers ─────────────────────────────────────────────────────────────
def _pad(tensor, N, max_size, dev):
    """Pad (N, k) → (N, max_size) with zeros."""
    out = torch.zeros(N, max_size, dtype=F32, device=dev)
    out[:, :tensor.shape[1]] = tensor
    return out

def _dist2d(ax, ay, bx, by, a_alive, b_alive, inf=1e9):
    """Masked (N, A, B) distance matrix."""
    dx = ax.unsqueeze(2) - bx.unsqueeze(1)
    dy = ay.unsqueeze(2) - by.unsqueeze(1)
    d  = torch.sqrt(dx**2 + dy**2)
    mask = a_alive.unsqueeze(2) & b_alive.unsqueeze(1)
    return d.masked_fill(~mask, inf)

def _vecdist2d(ax, ay, bx, by, a_alive, b_alive, inf=1e9):
    """Returns (dx, dy, dist) from b→a direction, (N, A, B), masked."""
    dx = ax.unsqueeze(2) - bx.unsqueeze(1)   # a - b = direction from b to a
    dy = ay.unsqueeze(2) - by.unsqueeze(1)
    d  = torch.sqrt(dx**2 + dy**2).clamp(min=1e-6)
    mask = a_alive.unsqueeze(2) & b_alive.unsqueeze(1)
    d_masked = d.masked_fill(~mask, inf)
    return dx, dy, d_masked


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════════════════════
def score_batch(history):
    """
    history: (N, T, 2) int tensor  [prey_count, pred_count]
    Returns (N,) float scores.
    """
    N, T, _ = history.shape
    scores = torch.zeros(N, dtype=torch.float64)
    if T < 50:
        return scores

    skip = T // 3   # ignore transient
    pr = history[:, skip:, 0].float()   # (N, T')
    pd = history[:, skip:, 1].float()

    # Must still be alive at end
    alive_end = (pr[:, -1] > 0) & (pd[:, -1] > 0)

    pr_mean = pr.mean(1);  pd_mean = pd.mean(1)
    pr_std  = pr.std(1);   pd_std  = pd.std(1)

    # Coefficient of variation (oscillation amplitude relative to mean)
    cv = (pr_std / (pr_mean + 1) + pd_std / (pd_mean + 1)) / 2

    # Count peaks in prey time series
    pr_mid = pr[:, 1:-1];  pr_l = pr[:, :-2];  pr_r = pr[:, 2:]
    is_peak = (pr_mid > pr_l) & (pr_mid > pr_r)
    peaks = is_peak.float().sum(1)

    # Min-population viability (soft penalty, not hard cutoff)
    pr_min = pr.min(1).values;  pd_min = pd.min(1).values
    viability = (pr_min / 5.0).clamp(max=1.0) * (pd_min / 2.0).clamp(max=1.0)

    # Population health bonus
    pop_bonus = (pr_mean / 40).clamp(max=1.0) * (pd_mean / 6).clamp(max=1.0)

    # Regularity: variance of inter-peak intervals (only if enough peaks)
    regularity = torch.zeros(N)
    for i in range(N):
        pidx = is_peak[i].nonzero(as_tuple=True)[0]
        if len(pidx) >= 4:
            ivs = (pidx[1:] - pidx[:-1]).float()
            reg = 1.0 / (1.0 + ivs.std() / ivs.mean().clamp(min=1))
            regularity[i] = reg.item()

    longevity = min(T / 160.0, 1.0)

    s = (cv * (peaks / 5).clamp(max=1.0) * pop_bonus
         * viability * longevity * (0.3 + 0.7 * regularity))
    s[~alive_end] = 0.0
    return s.double()


# ══════════════════════════════════════════════════════════════════════════════
#  Parameter space
# ══════════════════════════════════════════════════════════════════════════════
PARAM_DEFS = [
    # (name,               lo,    hi,   is_int)
    ('predSpeed',          3.5,   5.5,  False),
    ('preyFleeSpeed',      3.5,   5.5,  False),
    ('predEnergyDrain',    0.08,  0.30, False),
    ('predReproThreshold', 80.0,  200,  True),
    ('preyReproThreshold', 35.0,  90,   True),
    ('predPreyEnergy',     60.0,  140,  True),
    ('killRadius',         7.0,   18,   True),
    ('maxFood',            100,   250,  True),
    ('foodRegenRate',      6,     20,   True),
    ('initPred',           4,     10,   True),
]
DIM = len(PARAM_DEFS)

FIXED = dict(preyFoodEnergy=28, preyEnergyDrain=0.08, preySpeed=2.5,
             predDetectRange=90, predChaseRange=130, initPrey=100)

# Best from prior Node.js grid search — warm start seed
WARM_START = dict(predSpeed=4.4, preyFleeSpeed=4.2, predEnergyDrain=0.12,
                  predReproThreshold=130, preyReproThreshold=50, predPreyEnergy=100,
                  killRadius=10, maxFood=160, foodRegenRate=10, initPred=6)

def x_to_P(x_vec):
    P = dict(FIXED)
    for i, (name, lo, hi, is_int) in enumerate(PARAM_DEFS):
        v = float(x_vec[i]) * (hi - lo) + lo
        P[name] = int(round(v)) if is_int else round(v, 3)
    return P

def P_to_x(P):
    return [(P[name] - lo) / (hi - lo) for name, lo, hi, _ in PARAM_DEFS]

UNIT_BOUNDS = torch.zeros(2, DIM, dtype=torch.double)
UNIT_BOUNDS[1] = 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluate a batch of N_SIMS configs in one GPU pass
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_batch(x_matrix):
    """x_matrix: (K, DIM) unit-cube tensor → (K,) scores"""
    K = x_matrix.shape[0]
    params = [x_to_P(x_matrix[i].tolist()) for i in range(K)]
    sim = BatchSim(params)
    history = sim.run()          # (K, T, 2)
    return score_batch(history)  # (K,)


# ══════════════════════════════════════════════════════════════════════════════
#  Bayesian optimisation loop
# ══════════════════════════════════════════════════════════════════════════════
BO_DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INIT_ROUNDS = 3    # random batches before GP kicks in
BO_ROUNDS   = 12   # GP-guided rounds
BO_BATCH    = N_SIMS  # candidates per BO round = one full GPU batch

def run_bo():
    print(f"\nRunning {N_SIMS} sims in parallel per batch")
    print(f"Init: {INIT_ROUNDS} random rounds  |  BO: {BO_ROUNDS} GP-guided rounds\n")

    rng_np = np.random.default_rng(0)

    # ── Random exploration ─────────────────────────────────────────────────
    all_X = []
    all_Y = []

    # Round 0: warm start + random fill
    warm_x = torch.tensor([P_to_x({**FIXED, **WARM_START})], dtype=torch.double)
    rand_fill = torch.tensor(rng_np.uniform(0, 1, (N_SIMS-1, DIM)), dtype=torch.double)
    x_batch = torch.cat([warm_x, rand_fill], dim=0)

    for rnd in range(INIT_ROUNDS):
        t0 = time.time()
        if rnd > 0:
            x_batch = torch.tensor(rng_np.uniform(0, 1, (N_SIMS, DIM)), dtype=torch.double)
        y_batch = evaluate_batch(x_batch.float())
        all_X.append(x_batch); all_Y.append(y_batch)
        best_so_far = torch.cat(all_Y).max().item()
        print(f"  Init round {rnd+1}/{INIT_ROUNDS}  "
              f"batch_best={y_batch.max():.4f}  global_best={best_so_far:.4f}  "
              f"({time.time()-t0:.0f}s)")

    train_X = torch.cat(all_X).to(BO_DEVICE)             # (n_total, DIM)
    train_Y = torch.cat(all_Y).unsqueeze(-1).to(BO_DEVICE)  # (n_total, 1)

    # ── GP-guided rounds ───────────────────────────────────────────────────
    print(f"\nGP-guided Bayesian optimisation…")
    for rnd in range(BO_ROUNDS):
        t0 = time.time()

        # Fit GP
        y_mean = train_Y.mean(); y_std = train_Y.std().clamp(min=1e-6)
        train_Y_norm = (train_Y - y_mean) / y_std

        gp  = SingleTaskGP(train_X, train_Y_norm).to(BO_DEVICE)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Acquisition: qEI — pick BO_BATCH candidates at once
        acqf  = qExpectedImprovement(model=gp, best_f=train_Y_norm.max())
        bounds = UNIT_BOUNDS.to(BO_DEVICE)
        candidates, _ = optimize_acqf(
            acqf, bounds=bounds, q=BO_BATCH,
            num_restarts=12, raw_samples=256,
        )

        # Evaluate on GPU sim
        y_new = evaluate_batch(candidates.float().cpu()).to(BO_DEVICE)
        train_X = torch.cat([train_X, candidates.detach()])
        train_Y = torch.cat([train_Y, y_new.unsqueeze(-1)])

        best_so_far = train_Y.max().item()
        print(f"  BO round {rnd+1:2d}/{BO_ROUNDS}  "
              f"batch_best={y_new.max():.4f}  global_best={best_so_far:.4f}  "
              f"({time.time()-t0:.0f}s)")

    return train_X.cpu(), train_Y.cpu()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    all_X, all_Y = run_bo()
    elapsed = time.time() - t0

    best_idx = int(all_Y.squeeze().argmax())
    best_score = float(all_Y[best_idx])
    best_P = x_to_P(all_X[best_idx].tolist())

    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.0f}s   best score = {best_score:.4f}")
    print(f"{'='*55}")

    if best_score == 0:
        print("No viable config found — try increasing INIT_ROUNDS.")
    else:
        print("\nBest parameters (paste these into predator-prey.html):")
        for name, lo, hi, is_int in PARAM_DEFS:
            print(f"  {name:<26} = {best_P[name]}")

        print("\nCharacterising best config (8 runs)…")
        hist = BatchSim([best_P]*8).run()   # (8, T, 2)
        sc   = score_batch(hist)
        for i in range(8):
            pr = hist[i, :, 0]; pd = hist[i, :, 1]
            peaks = int(((pr[1:-1]>pr[:-2])&(pr[1:-1]>pr[2:])).sum())
            print(f"  run {i+1}: frames={hist.shape[1]*SNAP_EVERY:6d}  "
                  f"prey={int(pr.min()):3d}-{int(pr.max()):3d}  "
                  f"pred={int(pd.min()):2d}-{int(pd.max()):2d}  "
                  f"peaks={peaks:3d}  score={sc[i]:.3f}")
