#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrapolation benchmark for four models on 2D periodic wave dynamics:
- MeshFT-Net (DEC-based Hamiltonian model; Hodge = 'theory' or 'learn_geom')
- MGN (black-box MeshGraphNet)
- MGN-HP (MGN with Hamiltonian penalty via EnergyNet)
- HNN (separable H(q,p)=U(q)+T(p), KDK / Störmer–Verlet)

Additions in this version:
- Save trained models to disk (under out_dir/models).
- Save Energy vs Step as CSV (under out_dir/energy_csv) for each model,
  with columns: step, time, E_model, E_gt.

What this script evaluates (configurable via CLI):
1) Frequency extrapolation:   kmax_test > kmax_train
2) Long-horizon extrapolation: rollout length >> train step
3) Mesh-resolution extrapolation: train on coarse grid, test on finer grid
4) Parameter extrapolation:   wave speed c_test != c_train

Training uses one-step teacher forcing. You can choose full (q,p) supervision or q-only
supervision (--q_only_supervision=1), in which case p is *unobserved* and reported as such.

All long-horizon errors and energy drift are computed under a *true* theory Hamiltonian
with the test parameters (fair, model-agnostic evaluation).
"""

import os, math, argparse, random, json, csv
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- utilities -------------------------

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try: torch.use_deterministic_algorithms(True)
        except Exception: pass

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_device(*tensors, device="cpu"):
    return [t.to(device) for t in tensors]

def psnr(pred: torch.Tensor, target: torch.Tensor, peak: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-20: return 99.0
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)

# --- channel-wise std estimation & weighted loss (add) ---
@torch.no_grad()
def estimate_channel_std(loader, device):
    sq=0.0; sp=0.0; n=0
    for x0, x1 in loader:
        x1 = x1.to(device)
        sq += (x1[...,0]**2).sum().item()
        sp += (x1[...,1]**2).sum().item()
        n  += x1.shape[0]*x1.shape[1]
    sigma_q = (sq/max(1,n))**0.5 + 1e-8
    sigma_p = (sp/max(1,n))**0.5 + 1e-8
    return float(sigma_q), float(sigma_p)

@torch.no_grad()
def compute_input_norm_stats(coords: torch.Tensor,
                             node_extras: torch.Tensor,   # [N,1] (V0)
                             eattr: torch.Tensor):        # [E,edge_in] (dx,dy,|e|)
    def _stats(x: torch.Tensor):
        mu  = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        return mu, std
    mu_xy,  std_xy  = _stats(coords)       # [1,2]
    mu_ex,  std_ex  = _stats(node_extras)  # [1,1]
    mu_e,   std_e   = _stats(eattr)        # [1,edge_in]
    return mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e

def weighted_mse(pred, tgt, sigma_q, sigma_p):
    dq2 = ((pred[...,0]-tgt[...,0])/sigma_q)**2
    dp2 = ((pred[...,1]-tgt[...,1])/sigma_p)**2
    return (dq2+dp2).mean()

# --- substep integrators (keep total Δt = dt) ---
def integrate_mgn_nsub(model: Dict, x: torch.Tensor, coords, src, dst,
                       dt: float, eattr, n_sub: int, node_extras=None, scheme: str = "kdk"):
    n = max(1, int(n_sub))
    dts = dt / n
    for _ in range(n):
        if scheme == "kdk":
            x = mgn_step_kdk(model, x, coords, src, dst, dts, eattr, node_extras=node_extras)
        elif scheme == "rk2":
            k1 = model["net"](x, coords, src, dst, dts, eattr, node_extras=node_extras)
            x_mid = x + 0.5 * dts * k1
            k2 = model["net"](x_mid, coords, src, dst, dts, eattr, node_extras=node_extras)
            x = x + dts * k2
        else:  # euler
            v = model["net"](x, coords, src, dst, dts, eattr, node_extras=node_extras)
            x = x + dts * v
    return x

def integrate_hnn_nsub(model, x: torch.Tensor, dt: float, n_sub: int):
    n = max(1, int(n_sub))
    dts = dt / n
    with torch.enable_grad():
        for _ in range(n):
            x = model(x, dts)
    return x

# --- Hamiltonian vector-field helper -----------------------------------------
def apply_J_to_grad(gradH, state_mode: str = "canonical", M_data=None):
    """
    Map gradH = [dH/dq, dH/ds] to the Hamiltonian vector field J∇H.

    state_mode:
      - "canonical": x = [q, p]  -> [dq/dt, dp/dt] = [ dH/dp, - dH/dq ]
      - "velocity" : x = [q, v]  -> [dq/dt, dv/dt] = [ dH/dv, - dH/dq ]
        (We keep the simple canonical-like form for 'velocity'. If you want a
         mass-weighted variant, pass M_data and modify here accordingly.)
    """
    gq = gradH[..., 0]
    gs = gradH[..., 1]
    dqdt = gs
    dsdt = -gq
    return torch.stack([dqdt, dsdt], dim=-1)

def mgn_step_kdk(model: Dict, x: torch.Tensor, coords, src, dst, dt: float, eattr, node_extras=None) -> torch.Tensor:
    net = model["net"]
    q, p = x[...,0], x[...,1]
    v0 = net(x, coords, src, dst, dt, eattr, node_extras=node_extras)
    p_half = p + 0.5 * dt * v0[...,1]
    x_mid  = torch.stack([q, p_half], dim=-1)
    v_mid  = net(x_mid, coords, src, dst, dt, eattr, node_extras=node_extras)
    q_new  = q + dt * v_mid[...,0]
    x_tmp  = torch.stack([q_new, p_half], dim=-1)
    v_new  = net(x_tmp, coords, src, dst, dt, eattr, node_extras=node_extras)
    p_new  = p_half + 0.5 * dt * v_new[...,1]
    return torch.stack([q_new, p_new], dim=-1)

# ------------------ Hodge transfer helpers (GeomMLP / Learn) ------------------

def export_hodge_learnables(hodge_module: torch.nn.Module) -> dict:
    """
    Return a filtered state_dict that only contains mesh-agnostic learnables.
    - For GeomMLP: node_mlp.*, edge_mlp.*, gamma_mlp.* (if exists), log_speed2/log_c2 (if exists)
    - For Learnable (per-node/edge): DO NOT carry log_m/log_w/log_gamma across meshes.
      We only keep optional global scalar log_speed2/log_c2.
    """
    full = hodge_module.state_dict()
    keep_prefix = ("node_mlp.", "edge_mlp.", "gamma_mlp.")
    out = {}
    for k, v in full.items():
        if any(k.startswith(p) for p in keep_prefix):
            out[k] = v
        elif k.endswith("log_speed2") or k.endswith("log_c2") or k in ("log_speed2","log_c2"):
            out[k] = v
        # Everything else (coords, src, dst, V0, *_raw, *_mu, *_std, etc.) is skipped.
    return out

def load_hodge_learnables(target_hodge: torch.nn.Module, saved_sd: dict):
    """
    Load the filtered learnables into the target hodge. Silently skips missing keys.
    """
    tgt_keys = set(target_hodge.state_dict().keys())
    filt = {k: v for k, v in saved_sd.items() if k in tgt_keys}
    missing, unexpected = target_hodge.load_state_dict(filt, strict=False)
    print(f"[Hodge] transferred {len(filt)} learnable keys "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

# ------------------------- mesh & incidence -------------------------

def build_periodic_grid(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
    """Regular periodic grid; returns (coords[N,2], src[E], dst[E], V0[N])."""
    xs = torch.arange(nx, dtype=torch.float32) * (Lx / nx)
    ys = torch.arange(ny, dtype=torch.float32) * (Ly / ny)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    def node_id(i, j): return i * ny + j

    src, dst = [], []
    # horizontal edges (wrap along y)
    for i in range(nx):
        for j in range(ny):
            a = node_id(i, j); b = node_id(i, (j + 1) % ny)
            src.append(a); dst.append(b)
    # vertical edges (wrap along x)
    for i in range(nx):
        for j in range(ny):
            a = node_id(i, j); b = node_id((i + 1) % nx, j)
            src.append(a); dst.append(b)

    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    V0 = torch.full((nx*ny,), fill_value=(Lx/nx * Ly/ny), dtype=torch.float32)
    return coords, src, dst, V0

@torch.no_grad()
def B_times_q(src: torch.Tensor, dst: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(B q)_e = q[dst] - q[src]; supports [B,N] or [N]."""
    return q[..., dst] - q[..., src]

def BT_times_e(src: torch.Tensor, dst: torch.Tensor, e: torch.Tensor, N: int) -> torch.Tensor:
    """Apply B^T to edge scalar e. Accumulates onto nodes."""
    out = torch.zeros(*e.shape[:-1], N, dtype=e.dtype, device=e.device)
    out.index_add_(-1, dst, e)
    out.index_add_(-1, src, -e)
    return out

def build_edge_attr(coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Edge features: (dx, dy, |e|) with periodic minimum-image convention."""
    Lx = (coords[:, 0].max() - coords[:, 0].min()).item()
    Ly = (coords[:, 1].max() - coords[:, 1].min()).item()
    dv = coords[dst] - coords[src]
    if Lx > 0 and Ly > 0:
        dvx = dv[:, 0] - torch.round(dv[:, 0] / Lx) * Lx
        dvy = dv[:, 1] - torch.round(dv[:, 1] / Ly) * Ly
        dv = torch.stack([dvx, dvy], dim=-1)
    elen = dv.norm(dim=-1, keepdim=True)
    return torch.cat([dv, elen], dim=-1)  # [E,3]

# ------------------------- Hodge (theory and learn-geom) -------------------------

def node_to_edge_mean(src: torch.Tensor, dst: torch.Tensor, x_node: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x_node[src] + x_node[dst])

class HodgeTheoryConstC(nn.Module):
    """
    Fixed Hodge with uniform wave speed c (constant): M = V0, W = c^2 * 1 on edges.
    For grids, using W = c^2 * I_e is sufficient for canonical wave benchmarks.
    """
    def __init__(self, V0: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, c: float = 1.0):
        super().__init__()
        self.register_buffer("V0", V0.clone())
        self.register_buffer("src", src.clone())
        self.register_buffer("dst", dst.clone())
        self.c2 = float(c)**2

    def M_vec(self) -> torch.Tensor:
        return self.V0

    def W_diag_vec(self) -> torch.Tensor:
        E = self.src.numel()
        return torch.full((E,), fill_value=self.c2, dtype=self.V0.dtype, device=self.V0.device)

    def apply_W(self, e: torch.Tensor) -> torch.Tensor:
        return self.W_diag_vec() * e

class HodgeGeomMLP(nn.Module):
    """
    Geometry-conditioned Hodge:
      - Predicts positive node-wise M scale and edge-wise W scale from geometry only.
      - Node features: (x, y, V0)
      - Edge features: (dx, dy, |e|) with periodic minimum-image convention
    This enables generalization across different meshes/resolutions.
    """
    def __init__(self, coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, V0: torch.Tensor,
                 hidden: int = 64, layers: int = 2, eps: float = 1e-6, use_speed_scalar: bool = True):
        super().__init__()
        self.register_buffer("coords", coords.clone())
        self.register_buffer("src", src.clone())
        self.register_buffer("dst", dst.clone())
        self.register_buffer("V0", V0.clone())
        self.eps = float(eps)
        self.use_speed_scalar = bool(use_speed_scalar)
        if self.use_speed_scalar:
            self.log_c2 = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_c2", None)

        # standardize features
        with torch.no_grad():
            node_feats = torch.cat([coords, V0.unsqueeze(-1)], dim=-1)  # [N,3]
            Lx = (coords[:, 0].max() - coords[:, 0].min()).item()
            Ly = (coords[:, 1].max() - coords[:, 1].min()).item()
            dv = coords[dst] - coords[src]
            dvx = dv[:, 0] - torch.round(dv[:, 0] / max(Lx,1.0)) * max(Lx,1.0)
            dvy = dv[:, 1] - torch.round(dv[:, 1] / max(Ly,1.0)) * max(Ly,1.0)
            dv = torch.stack([dvx, dvy], dim=-1)
            elen = dv.norm(dim=-1, keepdim=True)
            edge_feats = torch.cat([dv, elen], dim=-1)  # [E,3]

            def _stdz(x):
                m = x.mean(dim=0, keepdim=True)
                s = x.std(dim=0, keepdim=True).clamp_min(1e-6)
                return (x - m) / s, m, s
            nf, nmu, nstd = _stdz(node_feats)
            ef, emu, estd = _stdz(edge_feats)

        self.register_buffer("node_feats_raw", node_feats)
        self.register_buffer("edge_feats_raw", edge_feats)
        self.register_buffer("node_mu", nmu); self.register_buffer("node_std", nstd)
        self.register_buffer("edge_mu", emu); self.register_buffer("edge_std", estd)

        def mlp(in_dim, out_dim):
            dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
            mods = []
            for i in range(len(dims)-1):
                mods.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2: mods.append(nn.SiLU())
            return nn.Sequential(*mods)

        self.node_mlp = mlp(3, 1)
        self.edge_mlp = mlp(3, 1)

    def _stdz_node(self): return (self.node_feats_raw - self.node_mu) / (self.node_std + 1e-6)
    def _stdz_edge(self): return (self.edge_feats_raw - self.edge_mu) / (self.edge_std + 1e-6)

    def M_vec(self) -> torch.Tensor:
        m_raw = self.node_mlp(self._stdz_node()).squeeze(-1)
        m_pos = F.softplus(m_raw) + self.eps
        M = self.V0 * m_pos
        return M

    def W_diag_vec(self) -> torch.Tensor:
        w_raw = self.edge_mlp(self._stdz_edge()).squeeze(-1)
        w_pos = F.softplus(w_raw) + self.eps
        if self.log_c2 is not None:
            w_pos = w_pos * torch.exp(self.log_c2)
        return w_pos

    def apply_W(self, e: torch.Tensor) -> torch.Tensor:
        return self.W_diag_vec() * e

# ------------------------- MeshFT-Net -------------------------

class MeshFTNet(nn.Module):
    """
    DEC Hamiltonian model (canonical):
        dq/dt = M^{-1} p,   dp/dt = -K q,   K = B^T W B
    Integrator: KDK (symplectic), single W-apply per kick.
    """
    def __init__(self, src: torch.Tensor, dst: torch.Tensor, hodge_module: nn.Module):
        super().__init__()
        self.src = src; self.dst = dst
        self.hodge = hodge_module
        self.N = self.hodge.V0.numel()
        self._nsub = 1

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        q, p = x[..., 0], x[..., 1]
        M = self.hodge.M_vec()
        Bq = B_times_q(self.src, self.dst, q)
        WBq = self.hodge.apply_W(Bq)
        pot = 0.5 * (Bq * WBq).sum(dim=-1)
        kin = 0.5 * ((p**2) / (M + 1e-12)).sum(dim=-1)
        return pot + kin

    def kdk_step(self, q, p, dt: float):
        M = self.hodge.M_vec(); Minv = 1.0 / (M + 1e-12)
        Bq = B_times_q(self.src, self.dst, q)
        WBq = self.hodge.apply_W(Bq)
        Kq = BT_times_e(self.src, self.dst, WBq, self.N)
        p_half = p - 0.5 * dt * Kq
        q_new  = q + dt * (Minv * p_half)
        Bq_new = B_times_q(self.src, self.dst, q_new)
        WBq_new= self.hodge.apply_W(Bq_new)
        Kq_new = BT_times_e(self.src, self.dst, WBq_new, self.N)
        p_new  = p_half - 0.5 * dt * Kq_new
        return q_new, p_new

    def forward(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        q, p = x[..., 0], x[..., 1]
        dts = dt / max(1, int(self._nsub))
        for _ in range(max(1, int(self._nsub))):
            q, p = self.kdk_step(q, p, dts)
        return torch.stack([q, p], dim=-1)

@torch.no_grad()
def estimate_omega_max(src: torch.Tensor, dst: torch.Tensor, hodge: nn.Module, iters: int = 25) -> float:
    """Power iteration on A = M^{-1}K; ω_max ≈ sqrt(λ_max)."""
    N = hodge.V0.numel()
    q = torch.randn(N, device=hodge.V0.device, dtype=hodge.V0.dtype)
    q = q / (q.norm() + 1e-12)
    lam = torch.tensor(0.0, device=q.device, dtype=q.dtype)
    for _ in range(iters):
        Bq = B_times_q(src, dst, q)
        WBq = hodge.apply_W(Bq)
        z = BT_times_e(src, dst, WBq, N)            # K q
        v = z / (hodge.M_vec() + 1e-12)             # M^{-1} K q
        lam = (q * v).sum()
        q = v / (v.norm() + 1e-12)
    lam = lam.clamp_min(1e-12)
    return float(torch.sqrt(lam))

# ------------------------- MGN & MGN-HP -------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2):
        super().__init__()
        dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
        mods = []
        for i in range(len(dims)-1):
            mods.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2: mods.append(nn.SiLU())
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class GraphLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden):
        super().__init__()
        self.edge_mlp = MLP(2*node_dim + edge_dim, hidden, hidden, layers=2)
        self.node_mlp = MLP(node_dim + hidden, hidden, node_dim, layers=2)
    def forward(self, h, src, dst, eattr):
        hi = h[src]; hj = h[dst]
        m_ij = self.edge_mlp(torch.cat([hi, hj, eattr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_ij)
        return h + self.node_mlp(torch.cat([h, agg], dim=-1))

class MeshGraphNetVF(nn.Module):
    """
    Predicts v(x) = [dq/dt, dp/dt].
    Node features = [q, p, x, y, V0] (node_in=5)
    """
    def __init__(self, node_in=5, edge_in=3, hidden=64, layers=4, use_input_norm: bool=False):
        super().__init__()
        self.enc = MLP(node_in, hidden, hidden, layers=2)
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_in, hidden) for _ in range(layers)])
        self.dec = MLP(hidden, hidden, 2, layers=2)

        self.use_input_norm = bool(use_input_norm)
        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))
        self.register_buffer("std_ex", torch.ones(1, 1))
        self.register_buffer("mu_e",   torch.zeros(1, edge_in))
        self.register_buffer("std_e",  torch.ones(1, edge_in))
        self.register_buffer("sigma",  torch.ones(2))  # [sigma_q, sigma_p]

    @torch.no_grad()
    def set_normalization(self, mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                          sigma_q: float = 1.0, sigma_p: float = 1.0):
        self.mu_xy.copy_(mu_xy.to(self.mu_xy))
        self.std_xy.copy_(std_xy.to(self.std_xy))
        self.mu_ex.copy_(mu_ex.to(self.mu_ex))
        self.std_ex.copy_(std_ex.to(self.std_ex))
        self.mu_e.copy_(mu_e.to(self.mu_e))
        self.std_e.copy_(std_e.to(self.std_e))
        self.sigma[0] = float(sigma_q)
        self.sigma[1] = float(sigma_p)

    def forward(self, x_nodes, coords, src, dst, dt=None, eattr=None, node_extras=None):
        B, N, _ = x_nodes.shape

        if self.use_input_norm:
            x_in = x_nodes.clone()
            x_in[..., 0] = x_in[..., 0] / (self.sigma[0] + 1e-12)  # q
            x_in[..., 1] = x_in[..., 1] / (self.sigma[1] + 1e-12)  # p
        else:
            x_in = x_nodes

        extras = node_extras if node_extras is not None else coords.new_zeros(N, 1)

        if self.use_input_norm:
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
            if eattr is None:
                e0 = torch.ones(dst.numel(), self.mu_e.shape[-1], device=coords.device, dtype=coords.dtype)
            else:
                e0 = eattr
            eattr_n = (e0 - self.mu_e) / (self.std_e + 1e-6)
        else:
            coords_n, extras_n = coords, extras
            eattr_n = eattr if eattr is not None else torch.ones(dst.numel(), 3, device=coords.device, dtype=coords.dtype)

        node_ctx = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B,N,3]
        h = torch.cat([x_in, node_ctx], dim=-1)                                            # [B,N,5]
        h = self.enc(h)

        B_, N_, H = h.shape
        offs = (torch.arange(B_, device=h.device, dtype=src.dtype) * N_).view(-1, 1)
        src_b = (src.view(1, -1) + offs).reshape(-1)
        dst_b = (dst.view(1, -1) + offs).reshape(-1)
        eattr_b = eattr_n.unsqueeze(0).expand(B_, -1, -1).reshape(-1, eattr_n.shape[-1])

        h_flat = h.reshape(B_*N_, -1)
        for gl in self.layers:
            h_flat = gl(h_flat, src_b, dst_b, eattr_b)
        v = self.dec(h_flat).view(B_, N_, 2)
        return v

class EnergyNet(nn.Module):
    """
    Node-wise energy network; message passing then sum to scalar H(x).
    Node features = [q, p, x, y]; Edge features = [dx, dy, |e|].
    """
    def __init__(self, node_in=5, edge_in=3, hidden=64, layers=4, use_input_norm: bool = False):
        super().__init__()
        self.enc = MLP(node_in, hidden, hidden, layers=2)
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_in, hidden) for _ in range(layers)])
        self.dec_node = MLP(hidden, hidden, 1, layers=2)

        self.use_input_norm = bool(use_input_norm)
        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))
        self.register_buffer("std_ex", torch.ones(1, 1))
        self.register_buffer("mu_e",   torch.zeros(1, edge_in))
        self.register_buffer("std_e",  torch.ones(1, edge_in))
        self.register_buffer("sigma",  torch.ones(2)) 

    @torch.no_grad()
    def set_normalization(self, mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                          sigma_q: float = 1.0, sigma_p: float = 1.0):
        self.mu_xy.copy_(mu_xy.to(self.mu_xy))
        self.std_xy.copy_(std_xy.to(self.std_xy))
        self.mu_ex.copy_(mu_ex.to(self.mu_ex))
        self.std_ex.copy_(std_ex.to(self.std_ex))
        self.mu_e.copy_(mu_e.to(self.mu_e))
        self.std_e.copy_(std_e.to(self.std_e))
        self.sigma[0] = float(sigma_q)
        self.sigma[1] = float(sigma_p)

    def forward(self, x_nodes, coords, src, dst, eattr=None, node_extras=None):
        B, N, _ = x_nodes.shape

        if self.use_input_norm:
            x_in = x_nodes.clone()
            x_in[...,0] = x_in[...,0] / (self.sigma[0] + 1e-12)
            x_in[...,1] = x_in[...,1] / (self.sigma[1] + 1e-12)
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
        else:
            x_in = x_nodes
            coords_n = coords

        extras = node_extras if node_extras is not None else coords.new_zeros(N, 1)
        if self.use_input_norm:
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
            if eattr is None:
                e0 = torch.ones(dst.numel(), self.mu_e.shape[-1], device=coords.device, dtype=coords.dtype)
            else:
                e0 = eattr
            eattr_n = (e0 - self.mu_e) / (self.std_e + 1e-6)
        else:
            extras_n = extras
            eattr_n  = eattr if eattr is not None else torch.ones(dst.numel(), 3, device=coords.device, dtype=coords.dtype)

        node_ctx = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([x_in, node_ctx], dim=-1)
        h = self.enc(h)

        B_, N_, H = h.shape
        offs = (torch.arange(B_, device=h.device, dtype=src.dtype) * N_).view(-1, 1)
        src_b = (src.view(1, -1) + offs).reshape(-1)
        dst_b = (dst.view(1, -1) + offs).reshape(-1)
        eattr_b = eattr_n.unsqueeze(0).expand(B_, -1, -1).reshape(-1, eattr_n.shape[-1])

        h_flat = h.reshape(B_*N_, -1)
        for gl in self.layers:
            h_flat = gl(h_flat, src_b, dst_b, eattr_b)
        e_node = self.dec_node(h_flat).view(B_, N_, 1)
        H = e_node.sum(dim=[1, 2])
        return H

class _SeparableNodeEnergy(nn.Module):
    """Per-node energy from a single scalar field (q or p), plus geometry context."""
    def __init__(self, hidden=64, layers=4, edge_in=3, use_input_norm: bool = False):
        super().__init__()
        self.enc = MLP(4, hidden, hidden, layers=2)  # scalar + x + y + V0
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_in, hidden) for _ in range(layers)])
        self.dec_node = MLP(hidden, hidden, 1, layers=2)

        self.use_input_norm = bool(use_input_norm)
        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))
        self.register_buffer("std_ex", torch.ones(1, 1))
        self.register_buffer("mu_e",   torch.zeros(1, edge_in))
        self.register_buffer("std_e",  torch.ones(1, edge_in))
        self.register_buffer("sigma_field", torch.tensor(1.0))

    @torch.no_grad()
    def set_normalization(self, mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e, sigma_field: float = 1.0):
        self.mu_xy.copy_(mu_xy.to(self.mu_xy))
        self.std_xy.copy_(std_xy.to(self.std_xy))
        self.mu_ex.copy_(mu_ex.to(self.mu_ex))
        self.std_ex.copy_(std_ex.to(self.std_ex))
        self.mu_e.copy_(mu_e.to(self.mu_e))
        self.std_e.copy_(std_e.to(self.std_e))
        self.sigma_field.fill_(float(sigma_field))

    def forward(self, field_scalar: torch.Tensor, coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                eattr: torch.Tensor = None, node_extras: torch.Tensor = None) -> torch.Tensor:
        B, N = field_scalar.shape

        s = field_scalar / (self.sigma_field + 1e-12) if self.use_input_norm else field_scalar

        extras = node_extras if node_extras is not None else coords.new_zeros(N, 1)
        if self.use_input_norm:
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
            if eattr is None:
                e0 = torch.ones(dst.numel(), self.mu_e.shape[-1], device=coords.device, dtype=coords.dtype)
            else:
                e0 = eattr
            eattr_n = (e0 - self.mu_e) / (self.std_e + 1e-6)
        else:
            coords_n, extras_n = coords, extras
            eattr_n = eattr if eattr is not None else torch.ones(dst.numel(), 3, device=coords.device, dtype=coords.dtype)

        node_feat = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([s.unsqueeze(-1), node_feat], dim=-1)
        h = self.enc(h)

        offs = (torch.arange(B, device=dst.device, dtype=src.dtype) * N).view(-1, 1)
        src_b = (src.view(1, -1) + offs).reshape(-1)
        dst_b = (dst.view(1, -1) + offs).reshape(-1)
        eattr_b = eattr_n.unsqueeze(0).expand(B, -1, -1).reshape(-1, eattr_n.shape[-1])

        h_flat = h.reshape(B*N, -1)
        for gl in self.layers:
            h_flat = gl(h_flat, src_b, dst_b, eattr_b)
        e_node = self.dec_node(h_flat).view(B, N, 1)
        return e_node.sum(dim=[1, 2])

class HNNSeparableSymplectic(nn.Module):
    """
    Separable HNN with KDK (Störmer–Verlet, 2nd order):
      p_{n+1/2} = p_n - (dt/2) * dU/dq(q_n)
      q_{n+1}   = q_n +  dt    * dT/dp(p_{n+1/2})
      p_{n+1}   = p_{n+1/2} - (dt/2) * dU/dq(q_{n+1})
    """
    def __init__(self, U_net: _SeparableNodeEnergy, T_net: _SeparableNodeEnergy,
                 coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                 eattr: torch.Tensor = None, node_extras: torch.Tensor = None):
        super().__init__()
        self.U_net = U_net; self.T_net = T_net
        self.register_buffer("coords", coords.clone())
        self.register_buffer("src", src.clone())
        self.register_buffer("dst", dst.clone())
        self.eattr = eattr
        self.node_extras = node_extras
        self.state_mode = "canonical"

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        q, p = x[..., 0], x[..., 1]
        U = self.U_net(q, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        T = self.T_net(p, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        return U + T

    def _grad_U_wrt_q(self, q: torch.Tensor) -> torch.Tensor:
        q_req = q.detach().requires_grad_(True)
        U = self.U_net(q_req, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        (gq,) = torch.autograd.grad(U.sum(), q_req, create_graph=self.training)
        return gq

    def _grad_T_wrt_p(self, p: torch.Tensor) -> torch.Tensor:
        p_req = p.detach().requires_grad_(True)
        T = self.T_net(p_req, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        (gp,) = torch.autograd.grad(T.sum(), p_req, create_graph=self.training)
        return gp

    def _verlet_step(self, q, p, dt):
        dUdq = self._grad_U_wrt_q(q); p_half = p - 0.5*dt*dUdq
        dTdp_half = self._grad_T_wrt_p(p_half); q_new = q + dt*dTdp_half
        dUdq_new = self._grad_U_wrt_q(q_new);    p_new = p_half - 0.5*dt*dUdq_new
        return q_new, p_new

    def forward(self, x, dt):
        q, p = x[...,0], x[...,1]
        q, p = self._verlet_step(q, p, dt)
        out = torch.stack([q, p], dim=-1)
        if not self.training: out = out.detach()
        return out

# ------------------------- analytic plane-wave dataset -------------------------

def sample_plane_params(Lx=1.0, Ly=1.0, c=1.0, kmax=3):
    kx = random.randint(1, kmax) * random.choice([-1, 1])
    ky = random.randint(0, kmax) * random.choice([-1, 1])
    if kx == 0 and ky == 0: ky = 1
    kvec = np.array([2*np.pi*kx/Lx, 2*np.pi*ky/Ly], dtype=np.float32)
    omega = c * np.linalg.norm(kvec)
    phi = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.5, 1.5)
    return kvec.astype(np.float32), float(omega), float(phi), float(amp)

def plane_wave_q_and_v(coords: torch.Tensor, t: float, kvec, omega, phi, amp, device):
    x = coords.to(device)
    phase = x @ torch.tensor(kvec, device=device) - omega * t + phi
    q = amp * torch.sin(phase)
    v = - omega * amp * torch.cos(phase)  # dq/dt
    return q, v

class PlaneWaveDataset(torch.utils.data.Dataset):
    """
    Pairs (x_t, x_{t+dt}) with random plane-wave parameters.
    Canonical packaging: x=[q, p] with p = V0 * dq/dt (M_data=V0).
    """
    def __init__(self, coords, V0, dt, size, c_wave=1.0, kmax=3, device="cpu"):
        super().__init__()
        self.coords = coords
        self.V0 = V0.to(device)
        self.dt = float(dt); self.size = int(size)
        self.c_wave = float(c_wave); self.kmax = int(kmax)
        self.device = device
        self.params = [sample_plane_params(Lx=float(coords[:,0].max()-coords[:,0].min()+1e-9),
                                           Ly=float(coords[:,1].max()-coords[:,1].min()+1e-9),
                                           c=self.c_wave, kmax=self.kmax) for _ in range(size)]
        self.t0 = np.random.uniform(0, 2*np.pi, size).astype(np.float32)

    def __len__(self): return self.size

    def __getitem__(self, idx):
        kvec, omega, phi, amp = self.params[idx]
        t = float(self.t0[idx])
        q0, v0 = plane_wave_q_and_v(self.coords, t, kvec, omega, phi, amp, self.device)
        q1, v1 = plane_wave_q_and_v(self.coords, t + self.dt, kvec, omega, phi, amp, self.device)
        p0 = self.V0 * v0; p1 = self.V0 * v1
        x0 = torch.stack([q0, p0], dim=-1)
        x1 = torch.stack([q1, p1], dim=-1)
        return x0, x1

@torch.no_grad()
def energy_from_theory(hodge: nn.Module, src, dst, x: torch.Tensor) -> torch.Tensor:
    q, p = x[..., 0], x[..., 1]
    M = hodge.M_vec()
    Bq = B_times_q(src, dst, q)
    WBq = hodge.apply_W(Bq)
    pot = 0.5 * (Bq * WBq).sum(dim=-1)
    kin = 0.5 * ((p**2) / (M + 1e-12)).sum(dim=-1)
    return pot + kin

@torch.no_grad()
def rollout_eval(model_kind: str, model, steps: int, dt: float, x0: torch.Tensor,
                 hodge_true: nn.Module, src, dst, coords=None, eattr=None) -> Dict[str, float]:
    """
    Roll out for 'steps' and compute:
      - final relative error in the physical norm induced by (M,W) of hodge_true
      - mean energy drift over the trajectory
    """
    traj = [x0.clone()]
    x = x0.clone().unsqueeze(0)  # [1,N,2]
    for _ in range(steps):
        if model_kind in ("mgn", "mgnhp"):
            x = mgn_step_kdk(
                model, x, coords, src, dst, dt, eattr,
                node_extras=(model.get("node_extras", None) if isinstance(model, dict) else None)
            )
        elif model_kind == "hnn":
            # HNN needs autograd even at eval time
            with torch.enable_grad():
                x = model(x, dt)
            x = x.detach()
        else:  # meshft_net
            x = model(x, dt)
        traj.append(x.squeeze(0))
    traj = torch.stack(traj, dim=0)  # [T+1,N,2]

    # GT rollout with theory Hodge (KDK)
    gt = [x0.clone()]
    z = x0.clone().unsqueeze(0)
    for _ in range(steps):
        q, p = z[..., 0], z[..., 1]
        M = hodge_true.M_vec(); Minv = 1.0 / (M + 1e-12)
        Bq = B_times_q(src, dst, q.squeeze(0))
        WBq = hodge_true.apply_W(Bq)
        Kq = BT_times_e(src, dst, WBq, hodge_true.V0.numel())
        p_half = p.squeeze(0) - 0.5 * dt * Kq
        q_new  = q.squeeze(0) + dt * (Minv * p_half)
        Bq_new = B_times_q(src, dst, q_new)
        WBq_new= hodge_true.apply_W(Bq_new)
        Kq_new = BT_times_e(src, dst, WBq_new, hodge_true.V0.numel())
        p_new  = p_half - 0.5 * dt * Kq_new
        z = torch.stack([q_new, p_new], dim=-1).unsqueeze(0)
        gt.append(z.squeeze(0))
    gt = torch.stack(gt, dim=0)

    xT, yT = traj[-1], gt[-1]
    Ez = energy_from_theory(hodge_true, src, dst, xT - yT)
    Ey = energy_from_theory(hodge_true, src, dst, yT)
    rel_final = float(torch.sqrt(Ez / (Ey + 1e-12)).mean().item())

    # energy drift (mean over steps)
    E = torch.stack([energy_from_theory(hodge_true, src, dst, traj[t]) for t in range(steps+1)])
    drift = (E - E[0]).abs() / (E[0].abs() + 1e-12)
    drift_mean = float(drift.mean().item())
    return {"rel_final": rel_final, "drift_mean": drift_mean}

@torch.no_grad()
def rollout_energy_to_csv(model_kind: str, model, steps: int, dt: float, x0: torch.Tensor,
                          hodge_true: nn.Module, src, dst, out_csv_path: str,
                          coords=None, eattr=None):
    """
    Roll out and save Energy vs Step to CSV with columns:
      step, time, E_model, E_gt
    """
    # model rollout
    traj = [x0.clone()]
    x = x0.clone().unsqueeze(0)
    for _ in range(steps):
        if model_kind in ("mgn", "mgnhp"):
            x = mgn_step_kdk(
                model, x, coords, src, dst, dt, eattr,
                node_extras=(model.get("node_extras", None) if isinstance(model, dict) else None)
            )
        elif model_kind == "hnn":
            with torch.enable_grad():
                x = model(x, dt)
            x = x.detach()
        else:  # meshft
            x = model(x, dt)
        traj.append(x.squeeze(0))
    traj = torch.stack(traj, dim=0)  # [T+1,N,2]
    E_model = torch.stack([energy_from_theory(hodge_true, src, dst, traj[t]) for t in range(steps+1)])  # [T+1,]

    # ground truth rollout with theory Hodge
    gt = [x0.clone()]
    z = x0.clone().unsqueeze(0)
    for _ in range(steps):
        q, p = z[..., 0], z[..., 1]
        M = hodge_true.M_vec(); Minv = 1.0 / (M + 1e-12)
        Bq = B_times_q(src, dst, q.squeeze(0))
        WBq = hodge_true.apply_W(Bq)
        Kq = BT_times_e(src, dst, WBq, hodge_true.V0.numel())
        p_half = p.squeeze(0) - 0.5 * dt * Kq
        q_new  = q.squeeze(0) + dt * (Minv * p_half)
        Bq_new = B_times_q(src, dst, q_new)
        WBq_new= hodge_true.apply_W(Bq_new)
        Kq_new = BT_times_e(src, dst, WBq_new, hodge_true.V0.numel())
        p_new  = p_half - 0.5 * dt * Kq_new
        z = torch.stack([q_new, p_new], dim=-1).unsqueeze(0)
        gt.append(z.squeeze(0))
    gt = torch.stack(gt, dim=0)
    E_gt = torch.stack([energy_from_theory(hodge_true, src, dst, gt[t]) for t in range(steps+1)])

    # write CSV
    ensure_dir(os.path.dirname(out_csv_path) or ".")
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time", "E_model", "E_gt"])
        for s in range(steps+1):
            w.writerow([s, s*dt, float(E_model[s].mean().item()), float(E_gt[s].mean().item())])

# ------------------------- training helpers -------------------------

def train_mgn_or_mgnhp(model: Dict, loader, dt, coords, src, dst, eattr, epochs: int,
                       q_only: bool, lam_ham: float = 0.0, energy_net: EnergyNet = None,
                       sigma_q: float = 1.0, sigma_p: float = 1.0):
    opt = model["_opt"]                        
    for ep in range(1, epochs+1):
        loss_meter = 0.0
        for x0, x1 in loader:
            x0 = x0.to(coords.device); x1 = x1.to(coords.device)
            # Use KDK for fair 2nd-order integration during training
            x_pred = mgn_step_kdk(
                model, x0, coords, src, dst, dt, eattr,
                node_extras=model.get("node_extras", None)
            )
            k1 = model["net"](
                x0, coords, src, dst, dt, eattr,
                node_extras=model.get("node_extras", None)
            )            

            loss = (F.mse_loss(x_pred[...,0], x1[...,0]) if q_only
                    else F.mse_loss(x_pred, x1))
                    # else weighted_mse(x_pred, x1, sigma_q, sigma_p))
            if lam_ham > 0.0 and energy_net is not None:
                x0_req = x0.detach().requires_grad_(True)
                H = energy_net(x0_req, coords, src, dst, eattr, node_extras=model.get("node_extras", None))
                (g,) = torch.autograd.grad(H.sum(), x0_req, create_graph=False)
                v_ham = apply_J_to_grad(g)
                loss = loss + lam_ham * F.mse_loss(k1, v_ham)                 

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model["net"].parameters(), 1.0)
            if energy_net is not None:
                torch.nn.utils.clip_grad_norm_(energy_net.parameters(), 1.0)
            opt.step(); loss_meter += float(loss.item())
        if ep % max(1, epochs//5) == 0 or ep == 1:
            print(f"[MGN{'-HP' if lam_ham>0 else ''}] ep {ep:04d}/{epochs}  loss≈{loss_meter/len(loader):.3e}")

def train_meshft(model: MeshFTNet, loader, dt, epochs: int, q_only: bool, sigma_q: float, sigma_p: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        print("[MeshFT-Net] theory Hodge -> no trainable params; skipping training.")
        return
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    for ep in range(1, epochs+1):
        loss_meter = 0.0
        for x0, x1 in loader:
            x0 = x0.to(params[0].device); x1 = x1.to(params[0].device)
            pred = model(x0, dt)
            loss = (F.mse_loss(pred[...,0], x1[...,0]) if q_only
                    else F.mse_loss(pred, x1))
                    # else weighted_mse(pred, x1, sigma_q, sigma_p))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); loss_meter += float(loss.item())
        if ep % max(1, epochs//5) == 0 or ep == 1:
            print(f"[MeshFT-Net] ep {ep:04d}/{epochs}  loss≈{loss_meter/len(loader):.3e}")

def train_hnn(model: HNNSeparableSymplectic, loader, dt, epochs: int, q_only: bool,
              sigma_q: float, sigma_p: float):
    params = list(model.U_net.parameters()) + list(model.T_net.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)                          
    for ep in range(1, epochs+1):
        loss_meter = 0.0
        for x0, x1 in loader:
            x0 = x0.to(params[0].device); x1 = x1.to(params[0].device)
            with torch.enable_grad():
                pred = model(x0, dt)                   
                loss = (F.mse_loss(pred[...,0], x1[...,0]) if q_only
                        else F.mse_loss(pred, x1))
                        # else weighted_mse(pred, x1, sigma_q, sigma_p))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); loss_meter += float(loss.item())
        if ep % max(1, epochs//5) == 0 or ep == 1:
            print(f"[HNN] ep {ep:04d}/{epochs}  loss≈{loss_meter/len(loader):.3e}")

# ------------------------- main: build, train, extrapolate -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs/extrapolation_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Train distribution
    ap.add_argument("--train_grid", type=int, nargs=2, default=[32, 32])
    ap.add_argument("--train_dt", type=float, default=0.004)
    ap.add_argument("--train_kmax", type=int, default=3)
    ap.add_argument("--train_c", type=float, default=1.0)
    ap.add_argument("--train_pairs", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--q_only_supervision", type=int, default=0, help="1 to supervise only q")

    ap.add_argument("--std_inputs", type=int, default=1,
                    help="standardize geometry inputs: coords/V0/edge_attr (0/1)")
    ap.add_argument("--std_state", type=int, default=1,
                    help="standardize state channels q,p using dataset std (0/1)")

    # Test distribution (extrapolation)
    ap.add_argument("--test_grid", type=int, nargs=2, default=[64, 64], help="mesh-resolution extrapolation")
    ap.add_argument("--std_geo_eval", type=str, default="test",
                    choices=["train", "test"],
                    help="Which geometry normalization (coords/V0/edge) to use at TEST time")
    ap.add_argument("--test_dt", type=float, default=0.004)
    ap.add_argument("--test_kmax", type=int, default=6, help="frequency extrapolation if > train_kmax")
    ap.add_argument("--test_c", type=float, default=1.0, help="parameter extrapolation if != train_c")
    ap.add_argument("--test_pairs", type=int, default=512)
    ap.add_argument("--rollout_T", type=int, default=200, help="long-horizon extrapolation length")
    ap.add_argument("--test_batch", type=int, default=4, help="batch size for test/eval to save memory")

    # MeshFT-Net Hodge choice
    ap.add_argument("--meshft_hodge_mode", type=str, default="learn_geom", choices=["theory","learn_geom"])
    ap.add_argument("--meshft_hidden", type=int, default=64)
    ap.add_argument("--meshft_layers", type=int, default=2)

    # MGN / MGN-HP
    ap.add_argument("--mgn_hidden", type=int, default=64)
    ap.add_argument("--mgn_layers", type=int, default=4)
    ap.add_argument("--lam_ham", type=float, default=0.05, help="Hamiltonian penalty weight for MGN-HP")

    # HNN
    ap.add_argument("--hnn_hidden", type=int, default=64)
    ap.add_argument("--hnn_layers", type=int, default=4)

    # Saving toggles
    ap.add_argument("--save_models", type=int, default=1)
    ap.add_argument("--save_energy_csv", type=int, default=1)

    args = ap.parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = torch.device(args.device)

    # ---------------- Train mesh & data ----------------
    nx_tr, ny_tr = args.train_grid
    coords_tr, src_tr, dst_tr, V0_tr = build_periodic_grid(nx_tr, ny_tr, 1.0, 1.0)
    coords_tr, src_tr, dst_tr, V0_tr = to_device(coords_tr, src_tr, dst_tr, V0_tr, device=device)
    eattr_tr = build_edge_attr(coords_tr, src_tr, dst_tr).to(device)

    train_ds = PlaneWaveDataset(coords_tr, V0_tr, dt=args.train_dt, size=args.train_pairs,
                                c_wave=args.train_c, kmax=args.train_kmax, device=device)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ---------------- Test mesh & data (OOD) ----------------
    nx_te, ny_te = args.test_grid
    coords_te, src_te, dst_te, V0_te = build_periodic_grid(nx_te, ny_te, 1.0, 1.0)
    coords_te, src_te, dst_te, V0_te = to_device(coords_te, src_te, dst_te, V0_te, device=device)
    eattr_te = build_edge_attr(coords_te, src_te, dst_te).to(device)

    test_ds = PlaneWaveDataset(coords_te, V0_te, dt=args.test_dt, size=args.test_pairs,
                               c_wave=args.test_c, kmax=args.test_kmax, device=device)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch, shuffle=False, num_workers=0)

    sigma_q, sigma_p = estimate_channel_std(train_loader, device)

    node_extras_tr = V0_tr.unsqueeze(-1)
    mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e = compute_input_norm_stats(
        coords_tr, node_extras_tr, eattr_tr
    )

    node_extras_te = V0_te.unsqueeze(-1)
    mu_xy_te, std_xy_te, mu_ex_te, std_ex_te, mu_e_te, std_e_te = compute_input_norm_stats(
        coords_te, node_extras_te, eattr_te
    )

    use_norm = bool(args.std_inputs or args.std_state)
    sigma_q_used = sigma_q if args.std_state else 1.0
    sigma_p_used = sigma_p if args.std_state else 1.0

    # Theory Hodge (for *evaluation*) built with test parameters
    eval_hodge_test = HodgeTheoryConstC(V0_te, src_te, dst_te, c=args.test_c).to(device)

    # ---------------- Build models ----------------
    # MeshFT-Net
    if args.meshft_hodge_mode == "theory":
        hodge_tr = HodgeTheoryConstC(V0_tr, src_tr, dst_tr, c=args.train_c).to(device)
    else:
        hodge_tr = HodgeGeomMLP(coords_tr, src_tr, dst_tr, V0_tr, hidden=args.meshft_hidden, layers=args.meshft_layers).to(device)
    meshft_net = MeshFTNet(src_tr, dst_tr, hodge_tr).to(device)

    # MGN
    node_extras_tr = V0_tr.unsqueeze(-1)
    mgn_net = MeshGraphNetVF(node_in=5, edge_in=eattr_tr.shape[1],
                            hidden=args.mgn_hidden, layers=args.mgn_layers,
                            use_input_norm=use_norm).to(device)
    mgn_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                            sigma_q=sigma_q_used, sigma_p=sigma_p_used)
    mgn = {"net": mgn_net,
           "_opt": torch.optim.AdamW(mgn_net.parameters(), lr=1e-3, weight_decay=1e-6),
           "node_extras": node_extras_tr}

    # MGN-HP
    mgnhp_net = MeshGraphNetVF(node_in=5, edge_in=eattr_tr.shape[1],
                            hidden=args.mgn_hidden, layers=args.mgn_layers,
                            use_input_norm=use_norm).to(device)
    mgnhp_en  = EnergyNet(node_in=5, edge_in=eattr_tr.shape[1],
                        hidden=args.mgn_hidden, layers=args.mgn_layers,
                        use_input_norm=use_norm).to(device)

    mgnhp_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                sigma_q=sigma_q_used, sigma_p=sigma_p_used)
    mgnhp_en.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                            sigma_q=sigma_q_used, sigma_p=sigma_p_used)
                            
    mgnhp = {"net": mgnhp_net,
             "_opt": torch.optim.AdamW(list(mgnhp_net.parameters())+list(mgnhp_en.parameters()),
                                       lr=1e-3, weight_decay=1e-6),
             "node_extras": node_extras_tr}
    # HNN
    node_extras_tr = V0_tr.unsqueeze(-1)
    eattr_tr_buf = eattr_tr
    U_net = _SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                edge_in=eattr_tr.shape[1], use_input_norm=use_norm).to(device)
    T_net = _SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                edge_in=eattr_tr.shape[1], use_input_norm=use_norm).to(device)

    U_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e, sigma_field=sigma_q_used)
    T_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e, sigma_field=sigma_p_used)

    hnn = HNNSeparableSymplectic(U_net, T_net, coords_tr, src_tr, dst_tr, eattr=eattr_tr_buf, node_extras=node_extras_tr).to(device)

    hodge_theory_tr = HodgeTheoryConstC(V0_tr, src_tr, dst_tr, c=args.train_c).to(device)
    omega_tr = estimate_omega_max(src_tr, dst_tr, hodge_theory_tr)
    sigma_cfl = 0.9
    alpha_tr = min(1.0, sigma_cfl / max(1e-12, omega_tr * args.train_dt))
    meshft_net._nsub = int(math.ceil(max(1.0, (omega_tr * args.train_dt) / sigma_cfl)))
    print(f"[CFL:train] omega≈{omega_tr:.3e}, alpha={alpha_tr:.3f}, MeshFT-Net substeps={meshft_net._nsub}")

    # ---------------- Train ----------------
    print("\n=== Training phase (train distribution) ===")
    train_meshft(meshft_net, train_loader, args.train_dt, args.epochs, q_only=bool(args.q_only_supervision),
            sigma_q=sigma_q, sigma_p=sigma_p)
    train_mgn_or_mgnhp(mgn,   train_loader, args.train_dt, coords_tr, src_tr, dst_tr, eattr_tr,
            epochs=args.epochs, q_only=bool(args.q_only_supervision),
            lam_ham=0.0, energy_net=None, sigma_q=sigma_q, sigma_p=sigma_p)
    train_mgn_or_mgnhp(mgnhp, train_loader, args.train_dt,
            coords_tr, src_tr, dst_tr, eattr_tr, epochs=args.epochs, q_only=bool(args.q_only_supervision),
            lam_ham=args.lam_ham, energy_net=mgnhp_en, sigma_q=sigma_q, sigma_p=sigma_p)
    train_hnn(hnn, train_loader, args.train_dt, args.epochs, q_only=bool(args.q_only_supervision),
            sigma_q=sigma_q, sigma_p=sigma_p)

    # ---------------- Save models ----------------
    if int(args.save_models) == 1:
        models_dir = os.path.join(args.out_dir, "models"); ensure_dir(models_dir)
        # MeshFT-Net
        if args.meshft_hodge_mode == "learn_geom":
            torch.save(
                {"hodge_learn": export_hodge_learnables(meshft_net.hodge),
                 "cls": "HodgeGeomMLP", "hidden": args.meshft_hidden, "layers": args.meshft_layers},
                os.path.join(models_dir, "meshft_hodge_learn.pt")
            )
        else:
            with open(os.path.join(models_dir, "meshft_theory_config.json"), "w") as f:
                json.dump({"mode": "theory", "c_train": args.train_c}, f, indent=2)
        # MGN
        torch.save(mgn_net.state_dict(), os.path.join(models_dir, "mgn_net.pt"))
        # MGN-HP
        torch.save({"net": mgnhp_net.state_dict(), "energy_net": mgnhp_en.state_dict()},
                   os.path.join(models_dir, "mgnhp.pt"))
        # HNN
        torch.save({"U_net": U_net.state_dict(), "T_net": T_net.state_dict()},
                   os.path.join(models_dir, "hnn.pt"))
        print(f"[save] saved trained models to {models_dir}")

    # ---------------- Extrapolation evaluation ----------------
    print("\n=== Extrapolation evaluation (test distribution) ===")

    # MeshFT-Net on test mesh
    if args.meshft_hodge_mode == "theory":
        hodge_te = HodgeTheoryConstC(V0_te, src_te, dst_te, c=args.test_c).to(device)
    else:
        hodge_te = HodgeGeomMLP(coords_te, src_te, dst_te, V0_te,
                                hidden=args.meshft_hidden, layers=args.meshft_layers).to(device)
        sd_learn = export_hodge_learnables(meshft_net.hodge)
        load_hodge_learnables(hodge_te, sd_learn)

    meshft_te = MeshFTNet(src_te, dst_te, hodge_te).to(device)

    # Shared CFL gate on test distribution from theory Hodge
    omega_te = estimate_omega_max(src_te, dst_te, eval_hodge_test)
    alpha_te = min(1.0, sigma_cfl / max(1e-12, omega_te * args.test_dt))
    meshft_te._nsub = int(math.ceil(max(1.0, (omega_te * args.test_dt) / sigma_cfl)))

    node_extras_te = V0_te.unsqueeze(-1)
    hnn_te = HNNSeparableSymplectic(
        U_net, T_net, coords_te, src_te, dst_te, eattr=eattr_te, node_extras=node_extras_te
    ).to(device)

    # --- choose geometry stats for EVAL ---
    if use_norm:
        if args.std_geo_eval == "test":
            mu_xy_eval, std_xy_eval = mu_xy_te, std_xy_te
            mu_ex_eval, std_ex_eval = mu_ex_te, std_ex_te
            mu_e_eval,  std_e_eval  = mu_e_te,  std_e_te
        else:  # "train"
            mu_xy_eval, std_xy_eval = mu_xy, std_xy
            mu_ex_eval, std_ex_eval = mu_ex, std_ex
            mu_e_eval,  std_e_eval  = mu_e,  std_e

        # --- apply to all nets (geometry only swapped; state sigmas stay from TRAIN) ---
        mgn_net.set_normalization(mu_xy_eval, std_xy_eval, mu_ex_eval, std_ex_eval, mu_e_eval, std_e_eval,
                                sigma_q=sigma_q_used, sigma_p=sigma_p_used)
        mgnhp_net.set_normalization(mu_xy_eval, std_xy_eval, mu_ex_eval, std_ex_eval, mu_e_eval, std_e_eval,
                                    sigma_q=sigma_q_used, sigma_p=sigma_p_used)
        mgnhp_en.set_normalization(mu_xy_eval, std_xy_eval, mu_ex_eval, std_ex_eval, mu_e_eval, std_e_eval,
                                sigma_q=sigma_q_used, sigma_p=sigma_p_used)
        U_net.set_normalization(mu_xy_eval, std_xy_eval, mu_ex_eval, std_ex_eval, mu_e_eval, std_e_eval,
                                sigma_field=(sigma_q_used))
        T_net.set_normalization(mu_xy_eval, std_xy_eval, mu_ex_eval, std_ex_eval, mu_e_eval, std_e_eval,
                                sigma_field=(sigma_p_used))

    hnn_te.eval()
    meshft_te.eval()
    mgn_net.eval()
    mgnhp_net.eval()
    # Release any training-time cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # get a batch from test loader
    test_iter = iter(test_loader)
    x0_batch, x1_batch = next(test_iter)
    x0_batch = x0_batch.to(device); x1_batch = x1_batch.to(device)

    # one-step metrics (q-only if requested)
    with torch.no_grad():
        # MeshFT-Net
        e1_meshft = (F.mse_loss(meshft_te(x0_batch, args.test_dt), x1_batch) if args.q_only_supervision==0
          else F.mse_loss(meshft_te(x0_batch, args.test_dt)[...,0], x1_batch[...,0]))

        # MGN
        x_pred_mgn = mgn_step_kdk({"net": mgn_net},  x0_batch, coords_te, src_te, dst_te, args.test_dt, eattr_te, node_extras=node_extras_te)
        e1_mgn = F.mse_loss(x_pred_mgn, x1_batch) if args.q_only_supervision==0 \
                else F.mse_loss(x_pred_mgn[...,0], x1_batch[...,0])

        # MGN-HP
        x_pred_hp  = mgn_step_kdk({"net": mgnhp_net}, x0_batch, coords_te, src_te, dst_te, args.test_dt, eattr_te, node_extras=node_extras_te)
        e1_hp  = F.mse_loss(x_pred_hp, x1_batch) if args.q_only_supervision==0 \
                else F.mse_loss(x_pred_hp[...,0], x1_batch[...,0])

        # HNN (KDK; dt_eff)
        with torch.enable_grad():
            pred_hnn = hnn_te(x0_batch, args.test_dt)
        e1_hnn = F.mse_loss(pred_hnn, x1_batch) if args.q_only_supervision==0 \
                else F.mse_loss(pred_hnn[...,0], x1_batch[...,0])

    print(f"[One-step MSE @ test] MeshFT-Net={e1_meshft:.3e} | MGN={e1_mgn:.3e} | MGN-HP={e1_hp:.3e} | HNN={e1_hnn:.3e}")

    # Long-horizon extrapolation (single seed from batch)
    with torch.no_grad():
        x0 = x0_batch[0]  # [N,2]
        res_meshft = rollout_eval("meshft_net", meshft_te, args.rollout_T, args.test_dt, x0, eval_hodge_test, src_te, dst_te)
        res_mgn = rollout_eval("mgn",   {"net": mgn_net,   "node_extras": node_extras_te}, args.rollout_T, args.test_dt, x0, eval_hodge_test, src_te, dst_te, coords=coords_te, eattr=eattr_te)
        res_hp  = rollout_eval("mgnhp", {"net": mgnhp_net, "node_extras": node_extras_te}, args.rollout_T, args.test_dt, x0, eval_hodge_test, src_te, dst_te, coords=coords_te, eattr=eattr_te)
        res_hnn = rollout_eval("hnn",   hnn_te,                                 args.rollout_T, args.test_dt, x0, eval_hodge_test, src_te, dst_te)

    # Save summary JSON
    summary = {
        "train": {
            "grid": args.train_grid, "dt": args.train_dt, "kmax": args.train_kmax, "c": args.train_c,
            "pairs": args.train_pairs, "epochs": args.epochs, "q_only": int(args.q_only_supervision),
            "meshft_hodge_mode": args.meshft_hodge_mode,
            "std_inputs": int(args.std_inputs),
            "std_state":  int(args.std_state),
        },
        "test": {
            "grid": args.test_grid, "dt": args.test_dt, "kmax": args.test_kmax, "c": args.test_c,
            "pairs": args.test_pairs, "rollout_T": args.rollout_T
        },
        "one_step_mse": {
            "MeshFT-Net": float(e1_meshft), "MGN": float(e1_mgn), "MGN-HP": float(e1_hp), "HNN": float(e1_hnn)
        },
        "rollout": {
            "MeshFT-Net": res_meshft, "MGN": res_mgn, "MGN-HP": res_hp, "HNN": res_hnn
        }
    }
    with open(os.path.join(args.out_dir, "extrapolation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Extrapolation summary ===")
    print(json.dumps(summary, indent=2))

    # Energy plots & CSV
    png_dir = args.out_dir
    csv_dir = os.path.join(args.out_dir, "energy_csv"); ensure_dir(csv_dir)

    def save_energy_plot_and_csv(model_kind, label, model, coords=None, eattr=None):
        with torch.no_grad():
            x0 = x0_batch[0]
            # Build trajectory with KDK (and dt_eff)
            traj = [x0.clone()]
            x = x0.clone().unsqueeze(0)
            for _ in range(args.rollout_T):
                if model_kind in ("mgn","mgnhp"):
                    x = mgn_step_kdk(model, x, coords_te, src_te, dst_te, args.test_dt, eattr_te, node_extras=node_extras_te)
                else:
                    step_dt = args.test_dt  # MeshFT-Net uses substep internally; others = raw dt
                    if model_kind == "hnn":
                        with torch.enable_grad():
                            x = model(x, step_dt)
                        x = x.detach()
                    else:  # meshft_net
                        x = model(x, step_dt)
                traj.append(x.squeeze(0))
            traj = torch.stack(traj, dim=0)

            E = torch.stack([energy_from_theory(eval_hodge_test, src_te, dst_te, traj[t])
                            for t in range(args.rollout_T+1)])
            plt.figure(figsize=(3.6, 3.0))
            plt.plot(E.detach().cpu().numpy())
            plt.xlabel("Step"); plt.ylabel("Energy"); plt.title(f"Energy vs step ({label})")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(png_dir, f"energy_{label}.png"), dpi=300); plt.close()

            # CSV also with dt_eff (time column should reflect the effective step)
            rollout_energy_to_csv(model_kind, model, args.rollout_T, args.test_dt, x0,
                                eval_hodge_test, src_te, dst_te,
                                out_csv_path=os.path.join(csv_dir, f"energy_{label}.csv"),
                                coords=coords_te, eattr=eattr_te)

    save_energy_plot_and_csv("meshft_net", "MeshFT-Net", meshft_te)
    save_energy_plot_and_csv("mgn",   "MGN",   {"net": mgn_net,   "node_extras": node_extras_te}, coords=coords_te, eattr=eattr_te)
    save_energy_plot_and_csv("mgnhp", "MGN-HP",{"net": mgnhp_net, "node_extras": node_extras_te}, coords=coords_te, eattr=eattr_te)
    save_energy_plot_and_csv("hnn",   "HNN",   hnn_te, coords=coords_te, eattr=eattr_te)

    print(f"[done] artifacts saved under: {args.out_dir}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()