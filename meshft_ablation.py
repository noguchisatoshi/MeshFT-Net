# meshft_ablation.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, random, json, os, csv
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- utils -------------------------

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try: torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_device(*tensors, device="cpu"):
    return [t.to(device) for t in tensors]

@torch.no_grad()
def psnr(pred: torch.Tensor, tgt: torch.Tensor, peak: float = 1.0) -> float:
    mse = F.mse_loss(pred, tgt).item()
    if mse <= 1e-20: return 99.0
    return 20.0 * math.log10(float(peak)) - 10.0 * math.log10(mse)


# ------------------------- mesh & incidence -------------------------

def build_periodic_grid(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
    """Regular periodic grid; returns (coords[N,2], src[E], dst[E], V0[N])."""
    xs = torch.arange(nx, dtype=torch.float32) * (Lx / nx)
    ys = torch.arange(ny, dtype=torch.float32) * (Ly / ny)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

    def nid(i, j): return i * ny + j

    src, dst = [], []
    # horizontal (wrap on y)
    for i in range(nx):
        for j in range(ny):
            a = nid(i, j); b = nid(i, (j+1) % ny)
            src.append(a); dst.append(b)
    # vertical (wrap on x)
    for i in range(nx):
        for j in range(ny):
            a = nid(i, j); b = nid((i+1) % nx, j)
            src.append(a); dst.append(b)

    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    V0  = torch.full((nx*ny,), fill_value=(Lx/nx * Ly/ny), dtype=torch.float32)
    return coords, src, dst, V0

@torch.no_grad()
def B_times_q(src: torch.Tensor, dst: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(B q)_e = q[dst] - q[src]; supports [B,N] or [N]."""
    return q[..., dst] - q[..., src]

def BT_times_e(src: torch.Tensor, dst: torch.Tensor, e: torch.Tensor, N: int) -> torch.Tensor:
    """Apply B^T to edge scalar e. Accumulate onto nodes."""
    out = torch.zeros(*e.shape[:-1], N, dtype=e.dtype, device=e.device)
    out.index_add_(-1, dst, e)
    out.index_add_(-1, src, -e)
    return out


# ------------------------- analytic plane-wave data -------------------------

def _sample_k(Lx: float, Ly: float, kmax: int) -> np.ndarray:
    kx = random.randint(1, kmax) * random.choice([-1, 1])
    ky = random.randint(0, kmax) * random.choice([-1, 1])
    if kx == 0 and ky == 0:
        ky = 1
    return np.array([2*np.pi*kx/Lx, 2*np.pi*ky/Ly], dtype=np.float32)

def plane_wave_pair(coords: torch.Tensor, V0: torch.Tensor, dt: float,
                    c: float, kmax: int, device: torch.device):
    """Build a single (x_t, x_{t+dt}) pair for canonical state x=[q,p], p=V0 * dq/dt."""
    Lx = float(coords[:,0].max() - coords[:,0].min() + 1e-12)
    Ly = float(coords[:,1].max() - coords[:,1].min() + 1e-12)
    kvec = _sample_k(Lx, Ly, kmax)
    k = torch.tensor(kvec, device=device)
    omega = float(c) * float(torch.linalg.norm(k).item())
    phi = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.5, 1.5)

    x = coords.to(device)
    phase0 = x @ k - omega * 0.0 + phi
    phase1 = x @ k - omega * dt  + phi
    q0 = torch.sin(phase0) * amp
    q1 = torch.sin(phase1) * amp
    v0 =  omega * torch.cos(phase0) * amp * (-1.0)  # dq/dt = -omega * cos
    v1 =  omega * torch.cos(phase1) * amp * (-1.0)
    p0 = V0.to(device) * v0
    p1 = V0.to(device) * v1
    x0 = torch.stack([q0, p0], dim=-1)  # [N,2]
    x1 = torch.stack([q1, p1], dim=-1)
    return x0, x1

def make_loader(coords, V0, dt, c, kmax, size, batch, device,
                noise_std: float = 0.0, noise_seed: int | None = None):
    pairs = [plane_wave_pair(coords, V0, dt, c, kmax, device) for _ in range(size)]
    X0 = torch.stack([p[0] for p in pairs], 0)  # [B,N,2]
    X1 = torch.stack([p[1] for p in pairs], 0)

    if float(noise_std) > 0.0:
        if X0.is_cuda:
            g = torch.Generator(device='cuda')
        else:
            g = torch.Generator()
        if noise_seed is not None:
            g.manual_seed(int(noise_seed))

        noise0 = torch.randn(X0.shape, dtype=X0.dtype, device=X0.device, generator=g)
        noise1 = torch.randn(X1.shape, dtype=X1.dtype, device=X1.device, generator=g)
        X0 = X0 + noise_std * noise0
        X1 = X1 + noise_std * noise1

    ds = torch.utils.data.TensorDataset(X0, X1)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)


# ------------------------- theory Hodge for evaluation -------------------------

class HodgeTheory(nn.Module):
    """Fixed theory Hodge: M = V0, W = c^2 * I (on edges)."""
    def __init__(self, V0: torch.Tensor, c: float = 1.0):
        super().__init__()
        self.register_buffer("V0", V0.clone())
        self.c2 = float(c)**2

    def M_vec(self): return self.V0

    def apply_W(self, e): return self.c2 * e

@torch.no_grad()
def energy_theory(hodge: HodgeTheory, src, dst, x: torch.Tensor) -> torch.Tensor:
    q, p = x[...,0], x[...,1]
    M = hodge.M_vec()
    Bq = B_times_q(src, dst, q)
    WBq= hodge.apply_W(Bq)
    pot = 0.5 * (Bq * WBq).sum(dim=-1)
    kin = 0.5 * ((p**2) / (M + 1e-12)).sum(dim=-1)
    return pot + kin

@torch.no_grad()
def phys_rel_final(hodge: HodgeTheory, src, dst, xT: torch.Tensor, yT: torch.Tensor) -> float:
    Ez = energy_theory(hodge, src, dst, xT - yT)
    Ey = energy_theory(hodge, src, dst, yT)
    return float(torch.sqrt(Ez / (Ey + 1e-12)).mean().item())

@torch.no_grad()
def estimate_omega_max(src, dst, hodge: HodgeTheory, iters: int = 25):
    """Power iteration on A = M^{-1}K; ω_max ≈ sqrt(λ_max)."""
    N = hodge.V0.numel()
    q = torch.randn(N, device=hodge.V0.device)
    q = q / (q.norm() + 1e-12)
    lam = torch.tensor(0.0, device=q.device)
    for _ in range(iters):
        Bq = B_times_q(src, dst, q)
        WBq= hodge.apply_W(Bq)
        Kq = BT_times_e(src, dst, WBq, N)
        v  = Kq / (hodge.V0 + 1e-12)
        lam= (q * v).sum()
        q  = v / (v.norm() + 1e-12)
    lam = lam.clamp_min(1e-12)
    return float(torch.sqrt(lam))


# ------------------------- MeshFT Variants -------------------------

class _BaseDEC(nn.Module):
    """KDK integrator with variant-specific A and W."""
    def __init__(self, src: torch.Tensor, dst: torch.Tensor, V0: torch.Tensor):
        super().__init__()
        self.src = src; self.dst = dst
        self.register_buffer("V0", V0.clone())
        self.N = int(V0.numel())
        self._nsub = 1

    # --- to be overridden ---
    def _A_apply(self, q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _AT_apply(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _W_apply(self, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    # ------------------------

    def _K_apply(self, q: torch.Tensor) -> torch.Tensor:
        Aq = self._A_apply(q)             # [B,E]
        WAq= self._W_apply(Aq)            # [B,E]
        return self._AT_apply(WAq)        # [B,N]

    def kdk(self, q, p, dt: float):
        M = self.V0; Minv = 1.0 / (M + 1e-12)
        Kq = self._K_apply(q)
        p_h = p - 0.5 * dt * Kq
        q_n = q + dt * (Minv * p_h)
        Kq2= self._K_apply(q_n)
        p_n = p_h - 0.5 * dt * Kq2
        return q_n, p_n

    def forward(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        q, p = x[...,0], x[...,1]
        dts = dt / max(1, int(self._nsub))
        for _ in range(max(1, int(self._nsub))):
            q, p = self.kdk(q, p, dts)
        return torch.stack([q, p], dim=-1)

class MeshFT_METRIC(_BaseDEC):
    """
      M_i = V0_i * exp(theta_M_i)
      W_e = c^2 * exp(theta_W_e)
    """
    def __init__(self, src, dst, V0, c: float = 1.0):
        super().__init__(src, dst, V0)
        self.c2 = float(c)**2
        N = V0.numel()
        E = src.numel()
        self.theta_M = nn.Parameter(torch.zeros(N))
        self.theta_W = nn.Parameter(torch.zeros(E))

    def _A_apply(self, q):
        return B_times_q(self.src, self.dst, q)

    def _AT_apply(self, y):
        return BT_times_e(self.src, self.dst, y, self.N)

    def _W_apply(self, e):
        # W_diag = c^2 * exp(theta_W)
        w = torch.exp(self.theta_W).unsqueeze(0)  # [1,E]
        return (self.c2 * w) * e

    def kdk(self, q, p, dt: float):
        M = self.V0 * torch.exp(self.theta_M)
        Minv = 1.0 / (M + 1e-12)
        Kq = self._K_apply(q)
        p_h = p - 0.5 * dt * Kq
        q_n = q + dt * (Minv * p_h)
        Kq2= self._K_apply(q_n)
        p_n = p_h - 0.5 * dt * Kq2
        return q_n, p_n


class MeshFT_INC(_BaseDEC):
    """Baseline: A = B (signed incidence), W = c^2 I."""
    def __init__(self, src, dst, V0, c: float = 1.0):
        super().__init__(src, dst, V0)
        self.c2 = float(c)**2

    def _A_apply(self, q): return B_times_q(self.src, self.dst, q)
    def _AT_apply(self, y): return BT_times_e(self.src, self.dst, y, self.N)
    def _W_apply(self, e): return self.c2 * e


class MeshFT_JPERM(_BaseDEC):
    """Wrong interconnection: dst endpoints randomly permuted (src fixed)."""
    def __init__(self, src, dst, V0, c: float = 1.0, seed: int = 0):
        super().__init__(src, dst, V0)
        self.c2 = float(c)**2
        g = torch.Generator(device=src.device)
        g.manual_seed(int(seed))
        perm = torch.randperm(dst.numel(), generator=g, device=dst.device)
        self.register_buffer("dst_perm", dst[perm].clone())

    def _A_apply(self, q):  # (q[dst_perm]-q[src])
        return q[..., self.dst_perm] - q[..., self.src]
    def _AT_apply(self, y):
        out = torch.zeros(*y.shape[:-1], self.N, dtype=y.dtype, device=y.device)
        out.index_add_(-1, self.dst_perm, y)
        out.index_add_(-1, self.src, -y)
        return out
    def _W_apply(self, e): return self.c2 * e


class MeshFT_WSIGN(_BaseDEC):
    """Signed W: learn per-edge weights (can be negative) → K may be indefinite."""
    def __init__(self, src, dst, V0, c_init: float = 1.0):
        super().__init__(src, dst, V0)
        E = src.numel()
        w0 = (float(c_init)**2) * torch.ones(E)
        self.w_raw = nn.Parameter(w0)  # unconstrained

    def _A_apply(self, q): return B_times_q(self.src, self.dst, q)
    def _AT_apply(self, y): return BT_times_e(self.src, self.dst, y, self.N)
    def _W_apply(self, e):
        w = self.w_raw.view(1, -1)
        return w * e


class MeshFT_UNORIENT(_BaseDEC):
    """Orientation removed: A_abs q = q_i + q_j (linear, orientation-even)."""
    def __init__(self, src, dst, V0, c: float = 1.0):
        super().__init__(src, dst, V0)
        self.c2 = float(c)**2

    def _A_apply(self, q):
        qi = q[..., self.src]; qj = q[..., self.dst]
        return qi + qj
    def _AT_apply(self, y):
        out = torch.zeros(*y.shape[:-1], self.N, dtype=y.dtype, device=y.device)
        out.index_add_(-1, self.dst, y)
        out.index_add_(-1, self.src, y)
        return out
    def _W_apply(self, e): return self.c2 * e


class MeshFT_JLEARN(_BaseDEC):
    """
    Learn incidence-like map A~ with per-edge scalars (a_e at dst, b_e at src):
      (A~ q)_e = a_e * q_j - b_e * q_i
    mode: 'psd'  -> a_e,b_e = softplus(raw)+eps  (K is PSD)
          'free' -> a_e,b_e = raw (signed)
    """
    def __init__(self, src, dst, V0, c: float = 1.0, mode: str = "psd"):
        super().__init__(src, dst, V0)
        assert mode in ("psd", "free")
        self.mode = mode
        self.c2 = float(c)**2
        E = src.numel()
        self.a_raw = nn.Parameter(0.05*torch.randn(E))
        self.b_raw = nn.Parameter(0.05*torch.randn(E))

    def _ab(self):
        if self.mode == "psd":
            a = F.softplus(self.a_raw) + 1e-8
            b = F.softplus(self.b_raw) + 1e-8
        else:
            a = self.a_raw
            b = self.b_raw
        return a, b  # [E], [E]

    def _A_apply(self, q):
        a, b = self._ab()
        qi = q[..., self.src]; qj = q[..., self.dst]
        return a.unsqueeze(0) * qj - b.unsqueeze(0) * qi

    def _AT_apply(self, y):
        a, b = self._ab()
        out = torch.zeros(*y.shape[:-1], self.N, dtype=y.dtype, device=y.device)
        out.index_add_(-1, self.dst,  a.unsqueeze(0) * y)
        out.index_add_(-1, self.src, -b.unsqueeze(0) * y)
        return out

    def _W_apply(self, e): return self.c2 * e

@torch.no_grad()
def constant_mode_leakage(model: nn.Module) -> float:
    """
    const_leakage = || A 1 ||_2 / sqrt(E)
    """
    device = model.V0.device
    N = int(model.V0.numel())
    E = int(model.src.numel())
    one = torch.ones(N, device=device)
    Ae  = model._A_apply(one)                 # [E] or [1,E]
    val = Ae.reshape(-1).norm().item() / (math.sqrt(E) + 1e-12)
    return float(val)


# ------------------------- training / eval -------------------------

def train_one(model: nn.Module, loader, dt: float, epochs: int, device: torch.device, lr: float = 1e-3, wd: float = 1e-6):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        return
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    model.train()
    for ep in range(1, epochs+1):
        loss_meter = 0.0
        for x0, x1 in loader:
            x0 = x0.to(device); x1 = x1.to(device)
            pred = model(x0, dt)
            loss = F.mse_loss(pred, x1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss_meter += float(loss.item())
        if ep == 1 or ep % max(1, epochs//5) == 0:
            print(f"[{model.__class__.__name__}] ep {ep:04d}/{epochs} loss≈{loss_meter/len(loader):.3e}")

@torch.no_grad()
def one_step_mse(model, loader, dt, device) -> float:
    model.eval(); mse=0.0; c=0
    for x0, x1 in loader:
        x0 = x0.to(device); x1 = x1.to(device)
        pred = model(x0, dt)
        mse += F.mse_loss(pred, x1).item(); c += 1
    return mse/max(1,c)

@torch.no_grad()
def rollout_eval(model, x0: torch.Tensor, steps: int, dt: float,
                 eval_hodge: HodgeTheory, src, dst,
                 tol_passive: float = 1e-5) -> Tuple[float, float, float, float]:
    """Return (rel_final, drift_mean, energy_inj, momentum_var)."""
    model.eval()
    x = x0.unsqueeze(0)   # [1,N,2]
    traj = [x.squeeze(0).clone()]
    p_series = [x0[..., 1].clone()]
    for _ in range(steps):
        x = model(x, dt)
        traj.append(x.squeeze(0).clone())
        p_series.append(x.squeeze(0)[..., 1].clone())
    traj = torch.stack(traj, 0)  # [T+1,N,2]
    P = torch.stack(p_series, 0) # [T+1,N]

    # GT rollout under theory Hodge
    z = x0.clone().unsqueeze(0)
    gt = [z.squeeze(0).clone()]
    for _ in range(steps):
        q, p = z[...,0], z[...,1]
        M  = eval_hodge.M_vec(); Minv = 1.0/(M+1e-12)
        Bq = B_times_q(src, dst, q.squeeze(0))
        WBq= eval_hodge.apply_W(Bq)
        Kq = BT_times_e(src, dst, WBq, eval_hodge.V0.numel())
        p_h = p.squeeze(0) - 0.5 * dt * Kq
        q_n = q.squeeze(0) + dt * (Minv * p_h)
        Bq2 = B_times_q(src, dst, q_n)
        WBq2= eval_hodge.apply_W(Bq2)
        Kq2 = BT_times_e(src, dst, WBq2, eval_hodge.V0.numel())
        p_n = p_h - 0.5 * dt * Kq2
        z   = torch.stack([q_n, p_n], dim=-1).unsqueeze(0)
        gt.append(z.squeeze(0))
    gt = torch.stack(gt, 0)

    relF   = phys_rel_final(eval_hodge, src, dst, traj[-1], gt[-1])
    E      = torch.stack([energy_theory(eval_hodge, src, dst, traj[t]) for t in range(steps+1)])
    drift  = ((E - E[0]).abs() / (E[0].abs() + 1e-12)).mean().item()

    viol = 0
    dE = E[1:] - E[:-1]  # [T]
    allow = tol_passive * torch.maximum(E[:-1].abs(), E[0].abs())  # [T]
    inj = torch.clamp(dE - allow, min=0.0)                        
    inj_rel  = float(inj.sum().item() / (E[0].abs().item() + 1e-12))

    Pb = P.unsqueeze(1)                            # [S,1,N]
    sump = Pb.sum(dim=2)                           # [S,1]
    denom = (Pb.abs().sum(dim=2).mean(dim=0) + 1e-12)  # [1]
    mom_var = ((sump.max(dim=0).values - sump.min(dim=0).values).abs() / denom).mean().item()
    return float(relF), float(drift), float(inj_rel), float(mom_var)

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs/meshft_ablate")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated list of seeds (e.g., '0,1,2'). If omitted, uses --seed.")
    ap.add_argument("--csv_path", type=str, default=None,
                    help="Path to aggregate CSV. Default: <out_dir>/ablation_results.csv")

    # mesh/data
    ap.add_argument("--grid", type=int, nargs=2, default=[32,32])
    ap.add_argument("--Lx", type=float, default=1.0); ap.add_argument("--Ly", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--train_pairs", type=int, default=2000)
    ap.add_argument("--val_pairs", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--rollout_T", type=int, default=500)
    ap.add_argument("--train_noise_std", type=float, default=0.001,
                    help="Std of additive Gaussian noise for TRAIN pairs (0 disables)")
    ap.add_argument("--val_noise_std", type=float, default=0.001,
                    help="Std of additive Gaussian noise for VAL pairs (0 disables)")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    device = torch.device(args.device)

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.replace(" ", "").split(",") if s != ""]
    else:
        seeds = [int(args.seed)]

    csv_path = args.csv_path or os.path.join(args.out_dir, "ablation_results.csv")
    csv_new = not os.path.exists(csv_path)
    if csv_new:
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "model", "one_step_mse", "rel_final", "energy_drift",
                        "energy_inj", "momentum_var", "const_leakage"])

    for seed in seeds:
        print(f"\n========== SEED {seed} ==========")
        set_seed(seed)

        # mesh
        nx, ny = args.grid
        coords, src, dst, V0 = build_periodic_grid(nx, ny, args.Lx, args.Ly)
        coords, src, dst, V0 = to_device(coords, src, dst, V0, device=device)

        # loaders
        train_loader = make_loader(
            coords, V0, args.dt, args.c, args.kmax, args.train_pairs, args.batch, device,
            noise_std=args.train_noise_std, noise_seed=seed*1000 + 11
        )
        val_loader = make_loader(
            coords, V0, args.dt, args.c, args.kmax, args.val_pairs, args.batch, device,
            noise_std=args.val_noise_std, noise_seed=seed*1000 + 22
        )

        # eval hodge & substeps
        eval_hodge = HodgeTheory(V0, c=args.c).to(device)
        omega = estimate_omega_max(src, dst, eval_hodge)
        sub = int(math.ceil(max(1.0, omega * args.dt)))  # safety ~ 1.0

        # models
        models: Dict[str, nn.Module] = {
            "INC"           : MeshFT_INC    (src, dst, V0, c=args.c).to(device),
            "MeshFT-Net"    : MeshFT_METRIC (src, dst, V0, c=args.c).to(device),
            "J-PERM"        : MeshFT_JPERM  (src, dst, V0, c=args.c, seed=seed+123).to(device),
            "W-SIGN"        : MeshFT_WSIGN  (src, dst, V0, c_init=args.c).to(device),
            "UNORIENT"      : MeshFT_UNORIENT(src, dst, V0, c=args.c).to(device),
            "J-LEARN(PSD)"  : MeshFT_JLEARN (src, dst, V0, c=args.c, mode="psd").to(device),
            "J-LEARN(FREE)" : MeshFT_JLEARN (src, dst, V0, c=args.c, mode="free").to(device),
        }
        for m in models.values():
            m._nsub = sub

        # train
        for name, mdl in models.items():
            print(f"\n=== Train {name} ===")
            lr = 1e-3
            wd = 1e-6 if name != "W-SIGN" else 1e-5   # a touch more wd helps stabilize free W
            train_one(mdl, train_loader, args.dt, args.epochs, device, lr=lr, wd=wd)

        # one-step & rollout
        summary = {}
        print("\n=== Validation (one-step & rollout) ===")
        with torch.no_grad():
            # pick one seed from val for rollout
            it = iter(val_loader)
            x0_b, x1_b = next(it)
            x0_one = x0_b[0].to(device)

        for name, mdl in models.items():
            mse = one_step_mse(mdl, val_loader, args.dt, device)
            relF, drift, inj_rel, mom_var = rollout_eval(mdl, x0_one, args.rollout_T, args.dt, eval_hodge, src, dst)
            cleak = constant_mode_leakage(mdl) 
            summary[name] = {
                "one_step_mse": mse,
                "rel_final": relF,
                "energy_drift": drift,
                "energy_inject": inj_rel,
                "momentum_var": mom_var,
                "const_leakage": cleak
            }
            print(f"{name:12s}  MSE={mse:.3e} | relF={relF:.3e} | drift={drift:.3e}"
                f" | energy_inj={inj_rel:.3e} | momVar={mom_var:.3e} | cLeak={cleak:.3e}")

        # save json
        json_path = os.path.join(args.out_dir, f"ablation_summary_seed{seed}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] saved: {json_path}")

        # append rows to aggregate CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for model_name, m in summary.items():
                w.writerow([seed, model_name,
                        m["one_step_mse"],
                        m["rel_final"],
                        m["energy_drift"],
                        m.get("energy_inject", float("nan")),
                        m.get("momentum_var", float("nan")),
                        m.get("const_leakage", float("nan"))])


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()