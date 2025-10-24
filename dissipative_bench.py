#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dissipative (Rayleigh) benchmark on a 2D periodic grid comparing four models:
  - MeshFT-Net (DEC-based) with explicit damping operator D(v)
  - MGN (message-passing vector field) baseline
  - MGN-HP (MGN + Hamiltonian penalty via EnergyNet)
  - HNN (separable) with explicit damping operator D(v)

Dynamics (canonical state): x = [q, p], v := dq/dt = M^{-1} p
  dq/dt = v
  dp/dt = -K q - D(v)      with  K = B^T W B,  M = V0,  W = c^2 V1inv

Dataset:
  Damped plane waves with amplitude ~ exp(-γ t). Each sample carries its own γ.

Evaluation (common across models, using theory Hodge for fairness):
  - One-step MSE / PSNR
  - Long rollout relative error (energy-induced norm)
  - Energy drift
  - Passivity violation rate (fraction of steps with E_{t+1} > E_t + tol)

Notes:
  * MeshFT-Net and HNN can use the same DampingOp (theory_meta_scalar / learn_global / learn_diag).
  * MGN/MGN-HP do NOT get an explicit damping operator (they must learn dissipation from data).
  * Hamiltonian penalty in MGN-HP pushes the conservative component towards J∇H; it does not
    enforce energy conservation (we do NOT use a ΔH≈0 consistency term here since data is dissipative).
"""

import os, math, csv, argparse, json, random
from typing import Dict, Tuple

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import io
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ------------------------- utils -------------------------

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

# --- add this small helper near the top-level utils ---
def meta_to_tensor(meta, key: str, device, dtype, view_b11: bool = True):
    """Handle both dict-of-lists and list-of-dicts from DataLoader collate."""
    if isinstance(meta, dict):
        vals = meta[key]                      # e.g., list[float] or tensor[B]
    elif isinstance(meta, (list, tuple)):
        vals = [m[key] for m in meta]         # list of dicts -> list
    else:
        raise TypeError(f"Unsupported meta type: {type(meta)}")
    t = torch.as_tensor(vals, device=device, dtype=dtype)
    return t.view(-1, 1, 1) if view_b11 else t

# --- channel-wise std estimation & weighted loss ---
@torch.no_grad()
def estimate_channel_std(loader, device):
    sq = 0.0; sp = 0.0; n = 0
    for x0, x1, *_ in loader:         # support (x0,x1) or (x0,x1,meta)
        x1 = x1.to(device)
        sq += (x1[...,0]**2).sum().item()
        sp += (x1[...,1]**2).sum().item()
        n  += x1.shape[0] * x1.shape[1]
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

def mgn_step_kdk(model: Dict, x: torch.Tensor, coords, src, dst, dt_eff: float,
                 eattr=None, node_extras=None) -> torch.Tensor:
    """
    Component-split KDK using the learned vector field v(x) = [v_q, v_p]:
      p_{n+1/2} = p_n + (dt/2) * v_p(x_n)
      q_{n+1}   = q_n +  dt    * v_q(q_n, p_{n+1/2})
      p_{n+1}   = p_{n+1/2} + (dt/2) * v_p(q_{n+1}, p_{n+1/2})
    """
    net = model["net"]
    q, p = x[...,0], x[...,1]
    v0 = net(x, coords, src, dst, eattr=eattr, node_extras=node_extras)
    p_half = p + 0.5 * dt_eff * v0[...,1]
    x_mid  = torch.stack([q, p_half], dim=-1)
    v_mid  = net(x_mid, coords, src, dst, eattr=eattr, node_extras=node_extras)
    q_new  = q + dt_eff * v_mid[...,0]
    x_tmp  = torch.stack([q_new, p_half], dim=-1)
    v_new  = net(x_tmp, coords, src, dst, eattr=eattr, node_extras=node_extras)
    p_new  = p_half + 0.5 * dt_eff * v_new[...,1]
    return torch.stack([q_new, p_new], dim=-1)

# ------------------------- grid & DEC -------------------------

def build_periodic_grid(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
    """Periodic rectangular grid: returns (coords[N,2], src[E], dst[E], V0[N], elen[E])."""
    xs = torch.arange(nx, dtype=torch.float32) * (Lx / nx)
    ys = torch.arange(ny, dtype=torch.float32) * (Ly / ny)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    def nid(i, j): return i*ny + j
    src, dst, elen = [], [], []
    hx, hy = Lx/nx, Ly/ny
    # horizontal
    for i in range(nx):
        for j in range(ny):
            a = nid(i, j); b = nid(i, (j+1)%ny)
            src.append(a); dst.append(b); elen.append(hy)
    # vertical
    for i in range(nx):
        for j in range(ny):
            a = nid(i, j); b = nid((i+1)%nx, j)
            src.append(a); dst.append(b); elen.append(hx)
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    elen= torch.tensor(elen, dtype=torch.float32)
    V0  = torch.full((nx*ny,), fill_value=hx*hy, dtype=torch.float32)
    return coords, src, dst, V0, elen

@torch.no_grad()
def B_times_q(src: torch.Tensor, dst: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return q[..., dst] - q[..., src]

def BT_times_e(src: torch.Tensor, dst: torch.Tensor, e: torch.Tensor, N: int) -> torch.Tensor:
    out = torch.zeros(*e.shape[:-1], N, dtype=e.dtype, device=e.device)
    out.index_add_(-1, dst, e); out.index_add_(-1, src, -e)
    return out

# ------------------------- Hodge (theory) -------------------------

class HodgeTheory(nn.Module):
    """Fixed Hodge: M = V0, W = (c^2) * V1inv."""
    def __init__(self, V0: torch.Tensor, V1inv: torch.Tensor, c_speed: float = 1.0):
        super().__init__()
        self.register_buffer("V0", V0.clone())
        self.register_buffer("V1inv", V1inv.clone())
        self.c2 = float(c_speed) ** 2
    def M_vec(self): return self.V0
    def apply_W(self, e, src, dst, N): return (self.c2 * self.V1inv) * e

# ------------------------- Hodge (learn_geom) -------------------------

class HodgeGeomMLP(nn.Module):
    """
    Geometry-conditioned Hodge block compatible with HodgeTheory API.
      M(q): node-wise   = V0 * softplus( MLP_node([x, y, V0]) )
      W(e): edge-wise   = V1inv * softplus( MLP_edge([dx, dy, |e|]) )
    Notes:
      * Uses periodic minimum-image convention for edge features.
      * Optional mean-normalization and global speed^2 scalar.
    """
    def __init__(self, coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                 V0: torch.Tensor, V1inv: torch.Tensor,
                 hidden: int = 64, layers: int = 2, use_sn: bool = False,
                 normalize: bool = False, use_speed_scalar: bool = False, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("coords", coords.clone())
        self.register_buffer("src", src.clone())
        self.register_buffer("dst", dst.clone())
        self.register_buffer("V0", V0.clone())
        self.register_buffer("V1inv", V1inv.clone())
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.use_speed_scalar = bool(use_speed_scalar)
        if self.use_speed_scalar:
            self.log_speed2 = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_speed2", None)

        # domain sizes（最短像用）
        Lx = (coords[:,0].max() - coords[:,0].min()).item()
        Ly = (coords[:,1].max() - coords[:,1].min()).item()
        Lx = 1.0 if Lx <= 0 else float(Lx)
        Ly = 1.0 if Ly <= 0 else float(Ly)
        self.register_buffer("Lx", torch.tensor(Lx, dtype=coords.dtype))
        self.register_buffer("Ly", torch.tensor(Ly, dtype=coords.dtype))

        # precompute (raw) geom features
        with torch.no_grad():
            node_feats = torch.cat([coords, V0.unsqueeze(-1)], dim=-1)  # [N,3]
            dv = coords[dst] - coords[src]
            dvx = dv[:,0] - torch.round(dvx := dv[:,0]/Lx)*Lx if Lx>0 else dv[:,0]
            dvy = dv[:,1] - torch.round(dvy := dv[:,1]/Ly)*Ly if Ly>0 else dv[:,1]
            dv = torch.stack([dvx, dvy], dim=-1)
            elen = dv.norm(dim=-1, keepdim=True)
            edge_feats = torch.cat([dv, elen], dim=-1)                  # [E,3]

            def _std(x):
                mu = x.mean(dim=0, keepdim=True)
                sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)
                return (x - mu)/sd, mu, sd
            nf, nmu, nstd = _std(node_feats)
            ef, emu, estd = _std(edge_feats)

        self.register_buffer("node_feats_raw", node_feats)
        self.register_buffer("edge_feats_raw", edge_feats)
        self.register_buffer("node_mu", nmu); self.register_buffer("node_std", nstd)
        self.register_buffer("edge_mu", emu); self.register_buffer("edge_std", estd)

        def mlp(din, dout, L):
            dims = [din] + [hidden]*(L-1) + [dout]
            mods = []
            for i in range(len(dims)-1):
                lin = nn.Linear(dims[i], dims[i+1])
                if use_sn: lin = nn.utils.spectral_norm(lin)
                mods.append(lin)
                if i < len(dims)-2: mods.append(nn.SiLU())
            return nn.Sequential(*mods)

        self.node_mlp = mlp(3, 1, layers)  # -> log_m proxy
        self.edge_mlp = mlp(3, 1, layers)  # -> log_w proxy

    def _std_node(self):
        x = self.node_feats_raw
        return (x - self.node_mu) / (self.node_std + 1e-6)

    def _std_edge(self):
        x = self.edge_feats_raw
        return (x - self.edge_mu) / (self.edge_std + 1e-6)

    def M_vec(self) -> torch.Tensor:
        m_raw = self.node_mlp(self._std_node()).squeeze(-1)
        m_pos = F.softplus(m_raw) + self.eps
        M = self.V0 * m_pos
        if self.normalize:
            M = M / (M.mean() + 1e-12)
        return M

    def W_diag_vec(self) -> torch.Tensor:
        w_raw = self.edge_mlp(self._std_edge()).squeeze(-1)
        w_pos = F.softplus(w_raw) + self.eps
        Wd = self.V1inv * w_pos
        if getattr(self, "log_speed2", None) is not None:
            Wd = Wd * torch.exp(self.log_speed2)
        if self.normalize:
            Wd = Wd / (Wd.mean() + 1e-12)
        return Wd

    def apply_W(self, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, Nnodes: int) -> torch.Tensor:
        return self.W_diag_vec() * e

# ------------------------- Damping operators -------------------------

class DampingOp(nn.Module):
    """
    Produces nonnegative per-node damping rates r used as D(p) = r ⊙ p
    in the momentum equation (equivalently Rayleigh 2γ q̇ when p = M q̇).

    Modes:
      - 'theory_meta_scalar': r = 2*gamma   (broadcast per sample to all nodes)
      - 'learn_global'      : r = softplus(rho)             (shared across nodes/samples)
      - 'learn_diag'        : r_i = softplus(r_i_raw)       (per node, shared across samples)
      - 'learn_diag_latent' : r_{b,i} inferred from (x, mesh) by a GNN (per sample, per node)
    """
    def __init__(self, mode: str, N: int, infer_net: nn.Module = None):
        super().__init__()
        assert mode in ("theory_meta_scalar", "learn_global", "learn_diag", "learn_diag_latent")
        self.mode = mode
        if mode == "learn_global":
            self.rho_raw = nn.Parameter(torch.tensor(0.0))
        elif mode == "learn_diag":
            self.r_raw = nn.Parameter(torch.zeros(N))
        elif mode == "learn_diag_latent":
            assert infer_net is not None, "learn_diag_latent requires an inference net"
            self.infer_net = infer_net

    def rates(self, x_nodes: torch.Tensor, *,
              gamma: torch.Tensor = None,
              coords: torch.Tensor = None, src: torch.Tensor = None, dst: torch.Tensor = None,
              eattr: torch.Tensor = None, node_extras: torch.Tensor = None) -> torch.Tensor:
        """
        Return per-sample per-node rates r >= 0 with shape [B,N].
        """
        B, N = x_nodes.shape[0], x_nodes.shape[1]
        if self.mode == "theory_meta_scalar":
            assert gamma is not None, "gamma is required for 'theory_meta_scalar'"
            # gamma: [B,1] -> r: [B,N]
            return (2.0 * gamma.view(-1, 1)).expand(B, N)
        elif self.mode == "learn_global":
            rho = F.softplus(self.rho_raw) + 1e-12
            return rho.expand(B, N)
        elif self.mode == "learn_diag":
            r = F.softplus(self.r_raw) + 1e-12      # [N]
            return r.view(1, -1).expand(B, -1)
        else:  # learn_diag_latent
            r_raw = self.infer_net(x_nodes, coords, src, dst, eattr=eattr, node_extras=node_extras)  # [B,N]
            return F.softplus(r_raw) + 1e-12

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        if self.mode == "learn_global":
            return {"rho": float(F.softplus(self.rho_raw).item())}
        elif self.mode == "learn_diag":
            r = F.softplus(self.r_raw).detach().cpu().numpy()
            return {"r_min": float(np.min(r)), "r_mean": float(np.mean(r)), "r_max": float(np.max(r))}
        else:
            return {}

    # kept for completeness if you still call it elsewhere
    def infer_rates(self, x_nodes, coords, src, dst, eattr=None, node_extras=None) -> torch.Tensor:
        assert self.mode == "learn_diag_latent"
        r_raw = self.infer_net(x_nodes, coords, src, dst, eattr=eattr, node_extras=node_extras)
        return F.softplus(r_raw) + 1e-12

# ------------------------- Models: MeshFT-Net (damped) -------------------------

class MeshFTNetDamped(nn.Module):
    """
    DEC Hamiltonian with linear damping acting on canonical momentum:
      dq/dt = v = M^{-1} p
      dp/dt = -K q - r ⊙ p
    Time stepping: KDK with exact half-step damping (Strang splitting).
    """
    def __init__(self, src, dst, hodge: HodgeTheory, damping: DampingOp, coords: torch.Tensor=None, eattr: torch.Tensor=None):
        super().__init__()
        self.src = src; self.dst = dst; self.hodge = hodge; self.damping = damping
        self.coords = coords
        self.eattr = eattr
        self.N = hodge.V0.numel()
        self.state_mode = "canonical"
        self._nsub = 1

    def energy(self, x):
        q, p = x[...,0], x[...,1]
        M = self.hodge.M_vec()
        Bq = B_times_q(self.src, self.dst, q)
        WBq= self.hodge.apply_W(Bq, self.src, self.dst, self.N)
        pot = 0.5 * (Bq * WBq).sum(dim=-1)
        kin = 0.5 * ((p**2) / (M + 1e-12)).sum(dim=-1)
        return pot + kin

    def kdk_step(self, q, p, dt: float, r_b: torch.Tensor):
        M = self.hodge.M_vec()
        Minv = 1.0 / (M + 1e-12)

        # ---- KICK (stiffness only at q_n)
        Bq  = B_times_q(self.src, self.dst, q)
        WBq = self.hodge.apply_W(Bq, self.src, self.dst, self.N)
        Kq  = BT_times_e(self.src, self.dst, WBq, self.N)
        p_h = p - 0.5 * dt * Kq

        # ---- DAMP (exact half-step on momentum)
        damp = torch.exp(-0.5 * dt * r_b)   # [B,N]
        p_h = p_h * damp

        # ---- DRIFT
        q_n = q + dt * (Minv * p_h)

        # ---- KICK (stiffness only at q_{n+1})
        Bq2  = B_times_q(self.src, self.dst, q_n)
        WBq2 = self.hodge.apply_W(Bq2, self.src, self.dst, self.N)
        Kq2  = BT_times_e(self.src, self.dst, WBq2, self.N)
        p_n  = p_h - 0.5 * dt * Kq2

        # ---- DAMP (second exact half-step)
        p_n = p_n * damp
        return q_n, p_n

    def forward(self, x, dt: float, gamma_b: torch.Tensor = None):
        q, p = x[...,0], x[...,1]
        # build per-sample per-node rates once and keep them fixed within the forward pass
        g = None if gamma_b is None else gamma_b.view(-1, 1)  # [B,1]
        node_extras = self.hodge.V0.unsqueeze(-1) if self.coords is not None else None
        r_b = self.damping.rates(x, gamma=g, coords=self.coords, src=self.src, dst=self.dst,
                                 eattr=self.eattr, node_extras=node_extras)  # [B,N]

        n = max(1, int(self._nsub)); dts = dt / n
        for _ in range(n):
            q, p = self.kdk_step(q, p, dts, r_b=r_b)
        return torch.stack([q, p], dim=-1)

# ------------------------- Models: Graph blocks for MGN/HP -------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2, use_sn=False):
        super().__init__()
        dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
        mods = []
        for i in range(len(dims)-1):
            lin = nn.Linear(dims[i], dims[i+1])
            if use_sn: lin = nn.utils.spectral_norm(lin)
            mods.append(lin)
            if i < len(dims)-2: mods.append(nn.SiLU())
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class GraphLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, use_sn=False):
        super().__init__()
        self.edge_mlp = MLP(2*node_dim + edge_dim, hidden, hidden, layers=2, use_sn=use_sn)
        self.node_mlp = MLP(node_dim + hidden, hidden, node_dim, layers=2, use_sn=use_sn)
    def forward(self, h, src, dst, eattr):
        hi = h[src]; hj = h[dst]
        m_ij = self.edge_mlp(torch.cat([hi, hj, eattr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_ij)
        return h + self.node_mlp(torch.cat([h, agg], dim=-1))

def build_edge_attr(coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Edge features: (dx, dy, |e|) with periodic min-image convention (for grid it's straightforward)."""
    Lx = (coords[:,0].max() - coords[:,0].min()).item()
    Ly = (coords[:,1].max() - coords[:,1].min()).item()
    dv = coords[dst] - coords[src]
    if Lx > 0 and Ly > 0:
        dvx = dv[:,0] - torch.round(dv[:,0]/Lx)*Lx
        dvy = dv[:,1] - torch.round(dv[:,1]/Ly)*Ly
        dv = torch.stack([dvx,dvy], dim=-1)
    elen = dv.norm(dim=-1, keepdim=True)
    return torch.cat([dv, elen], dim=-1)

class MeshGraphNetVF(nn.Module):
    """Predicts vector field v(x); used for MGN and MGN-HP (KDK update via wrapper)."""
    def __init__(self, in_dim=5, hidden=64, layers=4, out_dim=2, edge_in_dim=3, use_sn=False, use_input_norm: bool = False):
        super().__init__()
        def mlp(din, dout, L):
            dims = [din] + [hidden]*(L-1) + [dout]
            mods = []
            for i in range(len(dims)-1):
                lin = nn.Linear(dims[i], dims[i+1])
                if use_sn: lin = nn.utils.spectral_norm(lin)
                mods.append(lin)
                if i < len(dims)-2: mods.append(nn.SiLU())
            return nn.Sequential(*mods)
        self.enc = mlp(in_dim, hidden, 2)
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_in_dim, hidden, use_sn=use_sn) for _ in range(layers)])
        self.dec = mlp(hidden, out_dim, 2)
        self.edge_in_dim = edge_in_dim

        self.use_input_norm = bool(use_input_norm)  

        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))      # V0
        self.register_buffer("std_ex", torch.ones(1, 1))
        self.register_buffer("mu_e",   torch.zeros(1, edge_in_dim))
        self.register_buffer("std_e",  torch.ones(1, edge_in_dim))
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
        B,N,_ = x_nodes.shape

        if self.use_input_norm:
            x_in = x_nodes.clone()
            x_in[...,0] = x_in[...,0] / (self.sigma[0] + 1e-12)  # q
            x_in[...,1] = x_in[...,1] / (self.sigma[1] + 1e-12)  # p
        else:
            x_in = x_nodes

        extras = node_extras if node_extras is not None else coords.new_zeros(N,0)

        if self.use_input_norm:
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
            ea0 = eattr if eattr is not None else \
                  torch.ones(dst.numel(), self.edge_in_dim, device=coords.device, dtype=coords.dtype)
            eattr_n = (ea0 - self.mu_e) / (self.std_e + 1e-6)
        else:
            coords_n, extras_n = coords, extras
            eattr_n = eattr if eattr is not None else \
                      torch.ones(dst.numel(), self.edge_in_dim, device=coords.device, dtype=coords.dtype)

        Hnode = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B,-1,-1)
        h = torch.cat([x_in, Hnode], dim=-1)
        h = self.enc(h).reshape(B*N, -1)

        eattr_b = eattr_n.unsqueeze(0).expand(B,-1,-1).reshape(-1, eattr_n.shape[-1])
        offs = (torch.arange(B, device=dst.device, dtype=dst.dtype) * N).view(-1,1)
        src_b = (src.view(1,-1) + offs).reshape(-1)
        dst_b = (dst.view(1,-1) + offs).reshape(-1)

        for gl in self.layers:
            h = gl(h, src_b, dst_b, eattr_b)
        out = self.dec(h).view(B,N,-1)
        return out

class GammaInferNet(nn.Module):
    """Infer per-sample, per-node damping r_i (raw), later passed through softplus."""
    def __init__(
        self,
        node_in_dim=5,
        edge_in_dim=3,
        hidden=64,
        layers=2,
        use_sn=False,
        init_target_rate: float = None,      
        init_weight_scale: float = 1e-3,
        use_input_norm: bool = False      
    ):
        super().__init__()
        def mlp(din, dout, L):
            dims = [din] + [hidden]*(L-1) + [dout]
            mods=[]
            for i in range(len(dims)-1):
                lin = nn.Linear(dims[i], dims[i+1])
                if use_sn: lin = nn.utils.spectral_norm(lin)
                mods.append(lin)
                if i < len(dims)-2: mods.append(nn.SiLU())
            return nn.Sequential(*mods)

        self.enc = mlp(node_in_dim, hidden, 2)
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_in_dim, hidden, use_sn=use_sn) for _ in range(layers)])
        self.dec_node = mlp(hidden, 1, 2)  # raw (pre-softplus)
        self.edge_in_dim = edge_in_dim

        self.use_input_norm = bool(use_input_norm) 
        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))
        self.register_buffer("std_ex", torch.ones(1, 1))
        self.register_buffer("mu_e",   torch.zeros(1, edge_in_dim))
        self.register_buffer("std_e",  torch.ones(1, edge_in_dim))
        self.register_buffer("sigma",  torch.ones(2)) 

        # ---- scale init so that softplus(bias) ~= init_target_rate ----
        if init_target_rate is not None:
            self._scale_init(init_target_rate, init_weight_scale)

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

    @staticmethod
    def _softplus_inv(y: float) -> float:
        """Inverse of softplus for a positive target y."""
        # y here is small (~1e-1), simple formula is fine; for extra stability you can use math.expm1
        return math.log(math.exp(float(y)) - 1.0)

    def _scale_init(self, target_rate: float, weight_scale: float):
        """Set last-layer bias to softplus^{-1}(target_rate) and shrink its weights."""
        last = self.dec_node[-1]  # final Linear(hidden->1)
        with torch.no_grad():
            if getattr(last, "bias", None) is not None:
                last.bias.fill_(self._softplus_inv(target_rate))
            # Handle both plain Linear and spectral-norm-wrapped Linear
            if hasattr(last, "weight_orig"):        # spectral_norm case
                last.weight_orig.mul_(weight_scale)
            elif hasattr(last, "weight"):
                last.weight.mul_(weight_scale)

    def forward(self, x_nodes, coords, src, dst, eattr=None, node_extras=None):
        B,N,_ = x_nodes.shape

        if self.use_input_norm:
            x_in = x_nodes.clone()
            x_in[...,0] = x_in[...,0] / (self.sigma[0] + 1e-12)
            x_in[...,1] = x_in[...,1] / (self.sigma[1] + 1e-12)
        else:
            x_in = x_nodes

        extras = node_extras if node_extras is not None else coords.new_zeros(N,0)
        if self.use_input_norm:
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
            ea0 = eattr if eattr is not None else \
                  torch.ones(dst.numel(), self.edge_in_dim, device=coords.device, dtype=coords.dtype)
            eattr_n = (ea0 - self.mu_e) / (self.std_e + 1e-6)
        else:
            coords_n, extras_n = coords, extras
            eattr_n = eattr if eattr is not None else \
                      torch.ones(dst.numel(), self.edge_in_dim, device=coords.device, dtype=coords.dtype)

        Hnode = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B,-1,-1)
        h = torch.cat([x_in, Hnode], dim=-1)
        h = self.enc(h).reshape(B*N, -1)

        eattr_b = eattr_n.unsqueeze(0).expand(B,-1,-1).reshape(-1, eattr_n.shape[-1])
        offs = (torch.arange(B, device=dst.device, dtype=dst.dtype) * N).view(-1,1)
        src_b = (src.view(1,-1) + offs).reshape(-1)
        dst_b = (dst.view(1,-1) + offs).reshape(-1)

        for gl in self.layers:
            h = gl(h, src_b, dst_b, eattr_b)
        r_raw = self.dec_node(h).view(B, N)
        return r_raw

# ------------------------- EnergyNet & HP penalty helpers -------------------------

def apply_J_to_grad(gradH: torch.Tensor) -> torch.Tensor:
    """Canonical symplectic map J∇H: [dH/dp, -dH/dq]."""
    gq, gp = gradH[...,0], gradH[...,1]
    return torch.stack([gp, -gq], dim=-1)

class EnergyNet(nn.Module):
    """Scalar Hamiltonian H(x) (sum over per-node energies via message passing)."""
    def __init__(self, node_in_dim=5, edge_in_dim=3, hidden=64, layers=4, use_sn=False, use_input_norm: bool = False):
        super().__init__()
        def mlp(din, dout, L):
            dims = [din] + [hidden]*(L-1) + [dout]
            mods=[]
            for i in range(len(dims)-1):
                lin=nn.Linear(dims[i],dims[i+1])
                if use_sn: lin=nn.utils.spectral_norm(lin)
                mods.append(lin)
                if i < len(dims)-2: mods.append(nn.SiLU())
            return nn.Sequential(*mods)
        self.enc = mlp(node_in_dim, hidden, 2)
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_in_dim, hidden, use_sn=use_sn) for _ in range(layers)])
        self.dec_node = mlp(hidden, 1, 2)
        self.edge_in_dim = edge_in_dim

        self.use_input_norm = bool(use_input_norm)  # NEW
        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))
        self.register_buffer("std_ex", torch.ones(1, 1))
        self.register_buffer("mu_e",   torch.zeros(1, edge_in_dim))
        self.register_buffer("std_e",  torch.ones(1, edge_in_dim))
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
        B,N,_ = x_nodes.shape

        if self.use_input_norm:
            x_in = x_nodes.clone()
            x_in[...,0] = x_in[...,0] / (self.sigma[0] + 1e-12)
            x_in[...,1] = x_in[...,1] / (self.sigma[1] + 1e-12)
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
        else:
            x_in = x_nodes
            coords_n = coords

        extras = node_extras if node_extras is not None else coords.new_zeros(N,0)
        if self.use_input_norm:
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
            ea0 = eattr if eattr is not None else \
                  torch.ones(dst.numel(), self.edge_in_dim, device=coords.device, dtype=coords.dtype)
            eattr_n = (ea0 - self.mu_e) / (self.std_e + 1e-6)
        else:
            extras_n = extras
            eattr_n  = eattr if eattr is not None else \
                       torch.ones(dst.numel(), self.edge_in_dim, device=coords.device, dtype=coords.dtype)

        Hnode = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B,-1,-1)
        h = torch.cat([x_in, Hnode], dim=-1)
        h = self.enc(h).reshape(B*N, -1)

        eattr_b = eattr_n.unsqueeze(0).expand(B,-1,-1).reshape(-1, eattr_n.shape[-1])
        offs = (torch.arange(B, device=dst.device, dtype=dst.dtype) * N).view(-1,1)
        src_b = (src.view(1,-1) + offs).reshape(-1)
        dst_b = (dst.view(1,-1) + offs).reshape(-1)

        for gl in self.layers:
            h = gl(h, src_b, dst_b, eattr_b)
        e_node = self.dec_node(h).view(B,N,1)
        return e_node.sum(dim=[1,2])

# ------------------------- HNN (separable) with damping -------------------------

class _SeparableNodeEnergy(nn.Module):
    """Per-node energy network for a single scalar field (q or p)."""
    def __init__(self, hidden=64, layers=4, edge_in_dim=3, use_sn=False, use_input_norm: bool = False):
        super().__init__()
        def mlp(din,dout,L):
            dims=[din]+[hidden]*(L-1)+[dout]
            mods=[]
            for i in range(len(dims)-1):
                lin=nn.Linear(dims[i],dims[i+1])
                if use_sn: lin=nn.utils.spectral_norm(lin)
                mods.append(lin)
                if i < len(dims)-2: mods.append(nn.SiLU())
            return nn.Sequential(*mods)
        self.enc = mlp(4, hidden, 2)  # scalar + x + y + V0
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_in_dim, hidden, use_sn=use_sn) for _ in range(layers)])
        self.dec_node = mlp(hidden, 1, 2)
        self.edge_in_dim = edge_in_dim

        self.use_input_norm = bool(use_input_norm)  
        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))
        self.register_buffer("std_ex", torch.ones(1, 1))
        self.register_buffer("mu_e",   torch.zeros(1, edge_in_dim))
        self.register_buffer("std_e",  torch.ones(1, edge_in_dim))
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
                eattr: torch.Tensor=None, node_extras: torch.Tensor=None) -> torch.Tensor:
        B,N = field_scalar.shape
        s = field_scalar / (self.sigma_field + 1e-12) if self.use_input_norm else field_scalar

        extras = node_extras if node_extras is not None else coords.new_zeros(N,0)
        if self.use_input_norm:
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
            ea0 = eattr if eattr is not None else \
                  torch.ones(dst.numel(), self.edge_in_dim, device=coords.device, dtype=coords.dtype)
            eattr_n = (ea0 - self.mu_e) / (self.std_e + 1e-6)
        else:
            coords_n, extras_n = coords, extras
            eattr_n = eattr if eattr is not None else \
                      torch.ones(dst.numel(), self.edge_in_dim, device=coords.device, dtype=coords.dtype)

        node_feat = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B,-1,-1)
        h = torch.cat([s.unsqueeze(-1), node_feat], dim=-1)
        h = self.enc(h).reshape(B*N, -1)

        eattr_b = eattr_n.unsqueeze(0).expand(B,-1,-1).reshape(-1, eattr_n.shape[-1])
        offs = (torch.arange(B, device=dst.device, dtype=src.dtype) * N).view(-1,1)
        src_b = (src.view(1,-1) + offs).reshape(-1)
        dst_b = (dst.view(1,-1) + offs).reshape(-1)

        for gl in self.layers:
            h = gl(h, src_b, dst_b, eattr_b)
        e_node = self.dec_node(h).view(B,N,1)
        return e_node.sum(dim=[1,2])

class HNNSeparableDamped(nn.Module):
    """
    Separable HNN with linear damping acting on canonical momentum:
      dq/dt = ∂T/∂p  ,  dp/dt = -∂U/∂q - r ⊙ p
    Time stepping mirrors MeshFT-Net: KDK with exact half-step damping.
    """
    def __init__(self, U_net: _SeparableNodeEnergy, T_net: _SeparableNodeEnergy,
                 coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                 node_extras: torch.Tensor=None, eattr: torch.Tensor=None,
                 damping: DampingOp=None, V0: torch.Tensor=None):
        super().__init__()
        self.U_net, self.T_net = U_net, T_net
        self.register_buffer("coords", coords.clone())
        self.register_buffer("src", src.clone()); self.register_buffer("dst", dst.clone())
        self.node_extras = node_extras; self.eattr = eattr
        self.damping = damping
        self.register_buffer("V0", V0.clone() if V0 is not None else torch.ones(coords.shape[0]))
        self._nsub = 1
        self.state_mode = "canonical"

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        q, p = x[...,0], x[...,1]
        U = self.U_net(q, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        T = self.T_net(p, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        return U + T

    def _grad_U(self, q: torch.Tensor) -> torch.Tensor:
        q_req = q.detach().requires_grad_(True)
        U = self.U_net(q_req, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        (gq,) = torch.autograd.grad(U.sum(), q_req, create_graph=self.training)
        return gq

    def _grad_T(self, p: torch.Tensor) -> torch.Tensor:
        p_req = p.detach().requires_grad_(True)
        T = self.T_net(p_req, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        (gp,) = torch.autograd.grad(T.sum(), p_req, create_graph=self.training)
        return gp  # equals v := ∂T/∂p

    def kdk_step(self, q, p, dt: float, r_b: torch.Tensor):
        # ---- KICK (conservative force only)
        dU_dq = self._grad_U(q)
        p_half = p - 0.5 * dt * dU_dq

        # ---- DAMP (exact half-step on momentum)
        damp = torch.exp(-0.5 * dt * r_b)   # [B,N]
        p_half = p_half * damp

        # ---- DRIFT
        v_half = self._grad_T(p_half)       # equals dq/dt at the half step
        q_new  = q + dt * v_half

        # ---- KICK (at q_new)
        dU_dq2 = self._grad_U(q_new)
        p_new  = p_half - 0.5 * dt * dU_dq2

        # ---- DAMP (second exact half-step)
        p_new = p_new * damp
        return q_new, p_new

    def forward(self, x: torch.Tensor, dt: float, gamma_b: torch.Tensor=None) -> torch.Tensor:
        q, p = x[...,0], x[...,1]
        g = None if gamma_b is None else gamma_b.view(-1, 1)  # [B,1]
        node_extras = self.V0.unsqueeze(-1)
        r_b = self.damping.rates(x, gamma=g, coords=self.coords, src=self.src, dst=self.dst,
                                 eattr=self.eattr, node_extras=node_extras)  # [B,N]
        n = max(1, int(self._nsub)); dts = dt / n
        for _ in range(n):
            q, p = self.kdk_step(q, p, dts, r_b=r_b)
        return torch.stack([q, p], dim=-1)

# ------------------------- Damped plane-wave dataset -------------------------

def sample_plane_params(Lx=1.0, Ly=1.0, c=1.0, kmax=4, gamma_min=0.01, gamma_max=0.1):
    kx = random.randint(1, kmax) * random.choice([-1,1])
    ky = random.randint(0, kmax) * random.choice([-1,1])
    if kx==0 and ky==0: ky=1
    kvec = np.array([2*np.pi*kx/Lx, 2*np.pi*ky/Ly], dtype=np.float32)
    omega = c * np.linalg.norm(kvec)
    phi   = np.random.uniform(0, 2*np.pi)
    amp   = np.random.uniform(0.5, 1.5)
    gamma = np.random.uniform(gamma_min, gamma_max)
    return kvec.astype(np.float32), float(omega), float(phi), float(amp), float(gamma)

def damped_q_and_v(coords: torch.Tensor, t: float, kvec, omega, phi, amp, gamma, device):
    """q = A e^{-γ t} sin(k·x - ω t + φ), dq/dt = A e^{-γ t}[-γ sin(⋯) - ω cos(⋯)]."""
    x = coords.to(device)
    theta = x @ torch.tensor(kvec, device=device) - omega*t + phi
    e = math.exp(-gamma * t)
    q = amp * e * torch.sin(theta)
    v = amp * e * ( -gamma * torch.sin(theta) - omega * torch.cos(theta) )
    return q, v

class DampedPlaneWaveDataset(torch.utils.data.Dataset):
    """Pairs (x_t, x_{t+dt}) with canonical packaging p = V0 * dq/dt. Returns meta including γ."""
    def __init__(self, coords, V0, dt, size, c_wave=1.0, kmax=4,
                 gamma_min=0.01, gamma_max=0.1, device="cpu"):
        super().__init__()
        self.coords = coords
        self.V0 = V0.to(device)
        self.dt = float(dt); self.size = int(size)
        self.c_wave = float(c_wave); self.kmax=int(kmax); self.device=device
        Lx = float(coords[:,0].max()-coords[:,0].min()+1e-9)
        Ly = float(coords[:,1].max()-coords[:,1].min()+1e-9)
        self.params = [sample_plane_params(Lx, Ly, self.c_wave, self.kmax, gamma_min, gamma_max) for _ in range(size)]
        self.t0 = np.random.uniform(0, 2*np.pi, size).astype(np.float32)

    def __len__(self): return self.size

    def __getitem__(self, idx):
        kvec, omega, phi, amp, gamma = self.params[idx]
        t = float(self.t0[idx])
        q0,v0 = damped_q_and_v(self.coords, t, kvec, omega, phi, amp, gamma, self.device)
        q1,v1 = damped_q_and_v(self.coords, t+self.dt, kvec, omega, phi, amp, gamma, self.device)
        p0 = self.V0 * v0; p1 = self.V0 * v1
        x0 = torch.stack([q0, p0], dim=-1); x1 = torch.stack([q1, p1], dim=-1)
        meta = dict(kvec=torch.tensor(kvec), omega=omega, phi=phi, amp=amp, gamma=gamma, t=t)
        return x0, x1, meta

# ------------------------- energy & rollout -------------------------

def energy_from_hodge(hodge: HodgeTheory, src, dst, x: torch.Tensor) -> torch.Tensor:
    q, p = x[...,0], x[...,1]
    M = hodge.M_vec()
    Bq = B_times_q(src, dst, q)
    WBq= hodge.apply_W(Bq, src, dst, q.shape[-2] if x.dim()==3 else q.shape[-1])
    pot = 0.5 * (Bq * WBq).sum(dim=-1)
    kin = 0.5 * ((p**2) / (M + 1e-12)).sum(dim=-1)
    return pot + kin

@torch.no_grad()
def phys_rel_error(hodge: HodgeTheory, src, dst, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    Ez = energy_from_hodge(hodge, src, dst, x - y)
    Ey = energy_from_hodge(hodge, src, dst, y)
    return torch.sqrt(Ez / (Ey + 1e-12))

@torch.no_grad()
def estimate_omega_max(src, dst, hodge: HodgeTheory, iters: int = 20):
    """Power iteration on A = M^{-1}K -> ω_max ≈ sqrt(λ_max)."""
    N = hodge.V0.numel()
    q = torch.randn(N, device=hodge.V0.device, dtype=hodge.V0.dtype)
    q = q / (q.norm() + 1e-12)
    lam = torch.tensor(0.0, device=q.device, dtype=q.dtype)
    for _ in range(iters):
        Bq = B_times_q(src, dst, q)
        W_Bq = hodge.apply_W(Bq, src, dst, N)
        Kq = BT_times_e(src, dst, W_Bq, N)
        v  = Kq / (hodge.V0 + 1e-12)
        lam = (q * v).sum()
        vnorm = v.norm() + 1e-12
        q = v / vnorm
    lam = lam.clamp_min(1e-12)
    return float(torch.sqrt(lam))

@torch.no_grad()
def rollout_generic(model_kind: str, model, src, dst, coords, eval_hodge: HodgeTheory,
                    meta_batch: Dict[str, torch.Tensor], dt: float, T: int, device: str,
                    alpha_gate: float = 1.0, eattr: torch.Tensor=None, node_extras: torch.Tensor=None,
                    tol_passive: float = 1e-5):
    """Generic rollout for MeshFT-Net/HNN (damped) and MGN/MGN-HP; returns metrics and energy trace."""
    B = len(meta_batch["t"])
    # initial x
    x_list=[]
    for i in range(B):
        q0, v0 = damped_q_and_v(
            coords, float(meta_batch["t"][i]),
            meta_batch["kvec"][i].cpu().numpy(),
            float(meta_batch["omega"][i]), float(meta_batch["phi"][i]),
            float(meta_batch["amp"][i]), float(meta_batch["gamma"][i]), device
        )
        p0 = eval_hodge.V0 * v0
        x_list.append(torch.stack([q0, p0], dim=-1))
    x = torch.stack(x_list, dim=0)  # [B,N,2]
    gamma_b = meta_batch["gamma"].to(device)

    E = [energy_from_hodge(eval_hodge, src, dst, x)]
    Egt = [energy_from_hodge(eval_hodge, src, dst, x)] # ground truth
    rel_hist = []; steps = 0
    for k in range(T):
        if model_kind == "hnn":
            with torch.enable_grad():
                x = model(x, dt, gamma_b=gamma_b)
        elif model_kind == "meshft_net":
            x = model(x, dt, gamma_b=gamma_b)
        else:
            # MGN / MGN-HP
            x = mgn_step_kdk(model, x, coords, src, dst, dt, eattr=eattr, node_extras=node_extras)

        # ground truth at current time
        t_now = meta_batch["t"].to(device) + (k+1) * dt
        gt=[]
        for i in range(B):
            qg, vg = damped_q_and_v(
                coords, float(t_now[i]),
                meta_batch["kvec"][i].cpu().numpy(),
                float(meta_batch["omega"][i]), float(meta_batch["phi"][i]),
                float(meta_batch["amp"][i]), float(meta_batch["gamma"][i]), device
            )
            pg = eval_hodge.V0 * vg
            gt.append(torch.stack([qg, pg], dim=-1))
        gt = torch.stack(gt, dim=0)

        rel = phys_rel_error(eval_hodge, src, dst, x, gt)
        rel_hist.append(rel); steps += 1
        E.append(energy_from_hodge(eval_hodge, src, dst, x))
        Egt.append(energy_from_hodge(eval_hodge, src, dst, gt))

    E = torch.stack(E, dim=0)           # [S,B]
    Egt= torch.stack(Egt, dim=0)        # [S,B] ground truth
    rel_hist = torch.stack(rel_hist, 0) # [T,B]
    relF = rel_hist[-1].mean().item()
    relMed = rel_hist.median().item()
    E0, ET = E[0], E[-1]
    drift = ((ET - E0).abs() / (E0.abs() + 1e-12)).mean().item()

    Egt0 = (Egt[0].abs() + 1e-12)
    eng_err = (E - Egt).abs() / Egt0  # [S,B]
    eng_err_mean = eng_err.mean().item()
    eng_err_final = eng_err[-1].mean().item()

    viol = 0
    for s in range(E.shape[0]-1):
        allow = tol_passive * torch.maximum(E[s], E0.mean())
        viol += ((E[s+1] - E[s]) > allow).float().mean().item()

    pv_rate = viol / max(1, (E.shape[0]-1))
    return relF, relMed, drift, pv_rate, E, Egt, eng_err_final, eng_err_mean

# ------------------------- training -------------------------

def train_meshft(model: MeshFTNetDamped, loader, dt, epochs: int, device: str,
              use_tqdm: bool = True, sigma_q: float = 1.0, sigma_p: float = 1.0):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params)==0: return
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    for ep in range(epochs):
        it = tqdm(loader, desc=f"MeshFT-Net {ep+1}/{epochs}", leave=False, dynamic_ncols=True) if use_tqdm else loader
        for x0, x1, meta in it:
            x0 = x0.to(device); x1 = x1.to(device)
            gamma_b = meta_to_tensor(meta, "gamma", device=device, dtype=x0.dtype)
            pred = model(x0, dt, gamma_b=gamma_b)
            # loss = weighted_mse(pred, x1, sigma_q, sigma_p)
            loss = F.mse_loss(pred, x1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

@torch.no_grad()
def one_step_metrics_meshft(model: MeshFTNetDamped, loader, dt, device: str):
    model.eval(); mse=ps=0.0; c=0
    for x0, x1, meta in loader:
        x0 = x0.to(device); x1 = x1.to(device)
        gamma_b = meta_to_tensor(meta, "gamma", device=device, dtype=x0.dtype)
        pred = model(x0, dt, gamma_b=gamma_b)
        m = F.mse_loss(pred, x1).item(); mse += m
        ps += (99.0 if m<=1e-20 else 20.0*math.log10(1.0) - 10.0*math.log10(m))
        c += 1
    return mse/max(1,c), ps/max(1,c)

def train_mgn(model: Dict, loader, dt, epochs: int, device: str,
              use_tqdm: bool = True, sigma_q: float = 1.0, sigma_p: float = 1.0):
    net = model["net"]
    params = [p for p in net.parameters() if p.requires_grad]
    if len(params)==0: return
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    for ep in range(epochs):
        it = tqdm(loader, desc=f"MGN {ep+1}/{epochs}", leave=False, dynamic_ncols=True) if use_tqdm else loader
        for x0, x1, meta in it:
            x0 = x0.to(device); x1 = x1.to(device)
            # v_pred = net(x0, model["coords"], model["src"], model["dst"], eattr=model["eattr"], node_extras=model["node_extras"])
            x_pred = mgn_step_kdk(model, x0, model["coords"], model["src"], model["dst"], dt,
                                 eattr=model["eattr"], node_extras=model["node_extras"])
            # loss = weighted_mse(x_pred, x1, sigma_q, sigma_p)
            loss = F.mse_loss(x_pred, x1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

@torch.no_grad()
def one_step_metrics_mgn(model: Dict, loader, dt, device: str):
    net = model["net"]; net.eval(); mse=ps=0.0; c=0
    for x0, x1, meta in loader:
        x0 = x0.to(device); x1 = x1.to(device)
        # v_pred = net(x0, model["coords"], model["src"], model["dst"], eattr=model["eattr"], node_extras=model["node_extras"])
        x_pred = mgn_step_kdk(model, x0, model["coords"], model["src"], model["dst"], dt,
                            eattr=model["eattr"], node_extras=model["node_extras"])
        m = F.mse_loss(x_pred, x1).item(); mse += m
        ps += (99.0 if m<=1e-20 else 20.0*math.log10(1.0) - 10.0*math.log10(m))
        c += 1
    return mse/max(1,c), ps/max(1,c)

def train_mgnhp(model: Dict, loader, dt, epochs: int, device: str,
                lam_ham: float = 0.05, use_tqdm: bool = True,
                sigma_q: float = 1.0, sigma_p: float = 1.0):
    net = model["net"]; enet = model["energy_net"]
    params = [p for p in list(net.parameters()) + list(enet.parameters()) if p.requires_grad]
    if len(params)==0: return
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    warmup = max(1, int(0.3*epochs))
    for ep in range(epochs):
        it = tqdm(loader, desc=f"MGN-HP {ep+1}/{epochs}", leave=False, dynamic_ncols=True) if use_tqdm else loader
        for x0, x1, meta in it:
            x0 = x0.to(device); x1 = x1.to(device)
            v_pred = net(x0, model["coords"], model["src"], model["dst"],
                            eattr=model["eattr"], node_extras=model["node_extras"])
            x_pred = mgn_step_kdk(model, x0, model["coords"], model["src"], model["dst"], dt,
                                    eattr=model["eattr"], node_extras=model["node_extras"])
            # loss_fit = weighted_mse(x_pred, x1, sigma_q, sigma_p)
            loss_fit = F.mse_loss(x_pred, x1)

            # HP term: encourage v ≈ J∇H(x)
            x_req = x0.detach().requires_grad_(True)
            H = enet(x_req, model["coords"], model["src"], model["dst"], eattr=model["eattr"], node_extras=model["node_extras"])
            (g,) = torch.autograd.grad(H.sum(), x_req, create_graph=True)
            v_ham = apply_J_to_grad(g)
            eps = 1e-8
            a = F.normalize(v_pred.reshape(v_pred.shape[0], -1, v_pred.shape[-1]), dim=-1, eps=1e-8)
            b = F.normalize(v_ham.reshape_as(a),                                  dim=-1, eps=1e-8)
            loss_ham = F.mse_loss(a, b)
            lam = lam_ham * min(1.0, (ep+1)/warmup)
            loss = loss_fit + lam * loss_ham

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

@torch.no_grad()
def one_step_metrics_mgnhp(model: Dict, loader, dt, device: str):
    net = model["net"]; net.eval(); mse=ps=0.0; c=0
    for x0, x1, meta in loader:
        x0 = x0.to(device); x1 = x1.to(device)
        # v_pred = net(x0, model["coords"], model["src"], model["dst"], eattr=model["eattr"], node_extras=model["node_extras"])
        x_pred = mgn_step_kdk(model, x0, model["coords"], model["src"], model["dst"], dt,
                             eattr=model["eattr"], node_extras=model["node_extras"])
        m = F.mse_loss(x_pred, x1).item(); mse += m
        ps += (99.0 if m<=1e-20 else 20.0*math.log10(1.0) - 10.0*math.log10(m))
        c += 1
    return mse/max(1,c), ps/max(1,c)

def train_hnn(model: HNNSeparableDamped, loader, dt, epochs: int, device: str,
              use_tqdm: bool = True, sigma_q: float = 1.0, sigma_p: float = 1.0):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params)==0: return
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    for ep in range(epochs):
        it = tqdm(loader, desc=f"HNN {ep+1}/{epochs}", leave=False, dynamic_ncols=True) if use_tqdm else loader
        for x0, x1, meta in it:
            x0 = x0.to(device); x1 = x1.to(device)
            gamma_b = meta_to_tensor(meta, "gamma", device=device, dtype=x0.dtype)
            pred = model(x0, dt, gamma_b=gamma_b)
            # loss = weighted_mse(pred, x1, sigma_q, sigma_p)
            loss = F.mse_loss(pred, x1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

@torch.no_grad()
def one_step_metrics_hnn(model: HNNSeparableDamped, loader, dt, device: str):
    model.eval(); mse=ps=0.0; c=0
    for x0, x1, meta in loader:
        x0 = x0.to(device); x1 = x1.to(device)
        gamma_b = meta_to_tensor(meta, "gamma", device=device, dtype=x0.dtype)
        with torch.enable_grad():
            pred = model(x0, dt, gamma_b=gamma_b)
        pred = pred.detach()
        m = F.mse_loss(pred, x1).item(); mse += m
        ps += (99.0 if m<=1e-20 else 20.0*math.log10(1.0) - 10.0*math.log10(m))
        c += 1
    return mse/max(1,c), ps/max(1,c)

@torch.no_grad()
def _rollout_step_mgn(x, model_dict, dt, coords, src, dst, eattr=None, node_extras=None):
   return mgn_step_kdk(model_dict, x, coords, src, dst, dt, eattr=eattr, node_extras=node_extras)

def _collect_sequences_for_val_idx(idx: int, val_ds, coords, src, dst, V0,
                                   models: Dict[str, object], dt: float, T: int,
                                   device: str, eattr: torch.Tensor=None, node_extras: torch.Tensor=None):
    """
    For a validation sample `idx`, collect q(t) sequences for [GT, MeshFT-Net, MGN, MGN-HP, HNN].
    Returns:
      sequences: dict[name] -> list of length (T+1), each entry is q as a flat array of length N
      meta_tuple: (omega, phi, amp, gamma, t0) for reference
    """
    # Fetch the pair (x0: [N,2])
    x0, _, meta = val_ds[idx]
    x0 = x0.unsqueeze(0).to(device)  # [1,N,2]
    gamma = torch.tensor([float(meta["gamma"])], device=device, dtype=x0.dtype)

    # GT parameters
    kvec  = meta["kvec"].cpu().numpy() if torch.is_tensor(meta["kvec"]) else np.array(meta["kvec"], dtype=np.float32)
    omega = float(meta["omega"]); phi=float(meta["phi"]); amp=float(meta["amp"]); gam=float(meta["gamma"])
    t0    = float(meta["t"])

    seq = {"GT": [], "MeshFT-Net": [], "MGN": [], "MGN-HP": [], "HNN": []}

    # Initial q
    qg0, _ = damped_q_and_v(coords, t0, kvec, omega, phi, amp, gam, device)
    seq["GT"].append(qg0.detach().cpu().numpy())
    seq["MeshFT-Net"].append(x0[0,:,0].detach().cpu().numpy())
    seq["MGN"].append(x0[0,:,0].detach().cpu().numpy())
    seq["MGN-HP"].append(x0[0,:,0].detach().cpu().numpy())
    seq["HNN"].append(x0[0,:,0].detach().cpu().numpy())

    # Copies for each model
    x_meshft = x0.clone()
    x_mgn = x0.clone()
    x_hp  = x0.clone()
    x_hnn = x0.clone()

    for s in range(T):
        # GT
        t_now = t0 + (s+1)*dt
        qg, _ = damped_q_and_v(coords, t_now, kvec, omega, phi, amp, gam, device)
        seq["GT"].append(qg.detach().cpu().numpy())

        # MeshFT-Net
        x_meshft = models["MeshFT-Net"](x_meshft, dt, gamma_b=gamma)
        seq["MeshFT-Net"].append(x_meshft[0,:,0].detach().cpu().numpy())

        # MGN
        x_mgn = _rollout_step_mgn(x_mgn, models["MGN"], dt, coords, src, dst, eattr=eattr, node_extras=node_extras)
        seq["MGN"].append(x_mgn[0,:,0].detach().cpu().numpy())

        # MGN-HP
        x_hp  = _rollout_step_mgn(x_hp,  models["MGN-HP"], dt, coords, src, dst, eattr=eattr, node_extras=node_extras)
        seq["MGN-HP"].append(x_hp[0,:,0].detach().cpu().numpy())

        # HNN (forward uses autograd; be explicit with enable_grad)
        with torch.enable_grad():
            x_hnn = models["HNN"](x_hnn, dt, gamma_b=gamma)
        seq["HNN"].append(x_hnn[0,:,0].detach().cpu().numpy())

    return seq, (omega, phi, amp, gam, t0)

def _write_comparison_gif(path: str, sequences: Dict[str, list], nx: int, ny: int,
                          dt: float, meta_tuple, fps: int = 12, stride: int = 1):
    """
    sequences: dict[name] -> list of q arrays (length N), length (T+1).
    Color scale is fixed using all GT frames (diverging).
    """
    order = ["GT","MeshFT-Net","MGN","MGN-HP","HNN"]
    # Fix color scale from all GT frames; use a symmetric (diverging) range
    qgt = np.stack(sequences["GT"], axis=0)  # [S,N]
    vmax = float(np.abs(qgt).max())
    vmax = 1e-6 if vmax < 1e-12 else vmax
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 21)

    with imageio.get_writer(path, mode="I", fps=int(fps), loop=0) as writer:
        S = len(sequences["GT"])
        for s in range(0, S, max(1,int(stride))):
            fig, axs = plt.subplots(1, len(order), figsize=(3.0*len(order), 3.2), constrained_layout=True)
            if not isinstance(axs, np.ndarray): axs = np.array([axs])

            for j, name in enumerate(order):
                q = sequences[name][s].reshape(nx, ny)  # nid(i,j)=i*ny + j, so reshape (nx,ny) is correct
                cs = axs[j].contourf(q.T, levels=levels, extend="both", cmap="RdBu_r")  # transpose to plot y vertically
                axs[j].set_title(name)
                axs[j].set_xticks([]); axs[j].set_yticks([])

            fig.colorbar(cs, ax=axs.ravel().tolist(), shrink=0.85, pad=0.02)
            fig.suptitle(f"t = {s*dt:.3f}")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130)
            plt.close(fig)
            buf.seek(0)
            writer.append_data(imageio.imread(buf))

def generate_val_gifs(gif_dir: str, val_ds, coords, src, dst, V0, theory,
                      models: Dict[str, object], dt: float, T: int, stride: int,
                      grid_shape: Tuple[int,int], device: str,
                      eattr: torch.Tensor=None, node_extras: torch.Tensor=None, n_samples: int = 4, fps: int = 12):
    """Export comparison GIFs for the first `n_samples` validation items."""
    ensure_dir(gif_dir)
    nx, ny = grid_shape
    total = min(int(n_samples), len(val_ds))
    for idx in range(total):
        seq, meta_tuple = _collect_sequences_for_val_idx(
            idx, val_ds, coords, src, dst, V0, models, dt, T, device, eattr=eattr, node_extras=node_extras
        )
        out_path = os.path.join(gif_dir, f"val{idx:04d}_q_comparison.gif")
        _write_comparison_gif(out_path, seq, nx, ny, dt, meta_tuple, fps=fps, stride=stride)
        print(f"[GIF] wrote: {out_path}")

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs/dissipative_bench")
    ap.add_argument("--out_csv", type=str, default="runs/dissipative_bench/results.csv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # mesh
    ap.add_argument("--grid", type=int, nargs=2, default=[32,32])
    ap.add_argument("--Lx", type=float, default=1.0); ap.add_argument("--Ly", type=float, default=1.0)

    # dynamics / data
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--c_speed", type=float, default=1.0)
    ap.add_argument("--c_wave", type=float, default=None)
    ap.add_argument("--gamma_min", type=float, default=0.01)
    ap.add_argument("--gamma_max", type=float, default=0.1)

    ap.add_argument("--std_inputs", type=int, default=1,
                    help="standardize geometry inputs: coords/V0/edge_attr (0/1)")
    ap.add_argument("--std_state", type=int, default=1,
                    help="standardize state channels q,p using dataset std (0/1)")

    # train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--train_size", type=int, default=4000)
    ap.add_argument("--val_size", type=int, default=256)

    # damping
    ap.add_argument("--damp_mode", type=str, default="learn_diag_latent",
                    choices=["theory_meta_scalar","learn_global","learn_diag","learn_diag_latent"])
    ap.add_argument("--gamma_infer_hidden", type=int, default=64)
    ap.add_argument("--gamma_infer_layers", type=int, default=2)

    ap.add_argument("--meshft_hodge_mode", type=str, default="learn_geom",
                    choices=["theory","learn_geom"],
                    help="Hodge used INSIDE MeshFT dynamics (evaluation always uses theory).")
    ap.add_argument("--meshft_geom_hidden", type=int, default=64)
    ap.add_argument("--meshft_geom_layers", type=int, default=2)
    ap.add_argument("--normalize_hodge", type=int, default=0,
                    help="mean-normalize M/W produced by HodgeGeomMLP (0/1)")
    ap.add_argument("--use_speed_scalar", type=int, default=0,
                    help="enable a global speed^2 scalar in HodgeGeomMLP (0/1)")

    # MGN / HP / HNN hyperparams
    ap.add_argument("--mgn_hidden", type=int, default=64)
    ap.add_argument("--mgn_layers", type=int, default=4)
    ap.add_argument("--use_sn", type=int, default=0)
    ap.add_argument("--lam_ham", type=float, default=0.05)
    ap.add_argument("--hnn_hidden", type=int, default=64)
    ap.add_argument("--hnn_layers", type=int, default=4)

    # rollout
    ap.add_argument("--rollout_T", type=int, default=200)
    ap.add_argument("--save_details", type=int, default=1)
    ap.add_argument("--tqdm", type=int, default=1, help="show tqdm progress bars (1=yes, 0=no)")

    ap.add_argument("--save_models", type=int, default=1,
                    help="Save state_dicts of all models after training (1=yes, 0=no)")
    ap.add_argument("--ckpt_dir", type=str, default=None,
                    help="Directory to save checkpoints (default: <out_dir>/ckpts)")

    ap.add_argument("--make_gifs", type=int, default=1,
                    help="Export time-series contour comparison GIFs for the field q on the validation set (1=yes, 0=no)")
    ap.add_argument("--gif_dir", type=str, default=None,
                    help="Directory to save GIFs (default: <out_dir>/gifs)")
    ap.add_argument("--gif_n", type=int, default=2,
                    help="Number of validation samples (from the beginning) to create GIFs for")
    ap.add_argument("--gif_T", type=int, default=None,
                    help="Number of steps in GIFs (default: rollout_T)")
    ap.add_argument("--gif_stride", type=int, default=1,
                    help="Frame subsampling stride between steps")
    ap.add_argument("--gif_fps", type=int, default=12,
                    help="Frames per second for GIFs")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    device = torch.device(args.device)
    set_seed(args.seed)
    if args.c_wave is None: args.c_wave = float(args.c_speed)

    # mesh
    nx,ny = args.grid; Lx,Ly = args.Lx, args.Ly
    coords, src, dst, V0, elen = build_periodic_grid(nx, ny, Lx, Ly)
    V1inv = torch.ones_like(elen)  # grid edges
    coords, src, dst, V0, V1inv = to_device(coords, src, dst, V0, V1inv, device=device)

    # dataset & loaders
    train_ds = DampedPlaneWaveDataset(coords, V0, dt=args.dt, size=args.train_size,
                                      c_wave=args.c_wave, kmax=args.kmax,
                                      gamma_min=args.gamma_min, gamma_max=args.gamma_max,
                                      device=device.type)
    val_ds   = DampedPlaneWaveDataset(coords, V0, dt=args.dt, size=args.val_size,
                                      c_wave=args.c_wave, kmax=args.kmax,
                                      gamma_min=args.gamma_min, gamma_max=args.gamma_max,
                                      device=device.type)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    sigma_q, sigma_p = estimate_channel_std(train_loader, device)

    # theory Hodge (shared evaluation)
    eval_hodge = HodgeTheory(V0, V1inv, c_speed=args.c_speed).to(device)

    # edge/node features for GNNs
    eattr = build_edge_attr(coords, src, dst).to(device)
    node_extras = V0.unsqueeze(-1).to(device)

    mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e = compute_input_norm_stats(coords, node_extras, eattr)

    use_norm = bool(args.std_inputs or args.std_state)
    sigma_q_used = sigma_q if args.std_state else 1.0
    sigma_p_used = sigma_p if args.std_state else 1.0

    # ------------ MeshFT-Net (damped) ------------
    gamma_meshft = None
    if args.damp_mode == "learn_diag_latent":
        r0 = float(args.gamma_min + args.gamma_max)
        gamma_meshft = GammaInferNet(node_in_dim=5, edge_in_dim=eattr.shape[1],
                                  hidden=args.gamma_infer_hidden, layers=args.gamma_infer_layers,
                                  use_sn=bool(args.use_sn),
                                  init_target_rate=r0, init_weight_scale=1e-3, use_input_norm=use_norm).to(device)
        gamma_meshft.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                sigma_q=sigma_q_used, sigma_p=sigma_p_used)
    damping_meshft = DampingOp(args.damp_mode, N=coords.shape[0], infer_net=gamma_meshft).to(device)

    if args.meshft_hodge_mode == "learn_geom":
        meshft_hodge = HodgeGeomMLP(
            coords, src, dst, V0, V1inv,
            hidden=int(args.meshft_geom_hidden),
            layers=int(args.meshft_geom_layers),
            use_sn=bool(args.use_sn),
            normalize=bool(args.normalize_hodge),
            use_speed_scalar=bool(args.use_speed_scalar)
        ).to(device)
    else:
        meshft_hodge = HodgeTheory(V0, V1inv, c_speed=args.c_speed).to(device)

    meshft_net = MeshFTNetDamped(src, dst, meshft_hodge, damping_meshft, coords=coords, eattr=eattr).to(device)
    # substepping from ω_max
    try:
        omega = estimate_omega_max(src, dst, meshft_hodge, iters=25)
        meshft_net._nsub = int(math.ceil(max(1.0, omega * args.dt)))
        print(f"[MeshFT-Net:init] omega_max≈{omega:.3e}, dt={args.dt:.3e} -> substeps={meshft_net._nsub}")
    except Exception as e:
        meshft_net._nsub = 1
        print(f"[MeshFT-Net:init] omega_max estimation failed ({e}); using substeps=1.")
    print("[train] MeshFT-Net (damped) ...")
    train_meshft(meshft_net, train_loader, args.dt, args.epochs, device=str(device), use_tqdm=bool(args.tqdm),
          sigma_q=sigma_q, sigma_p=sigma_p)
    meshft_mse, meshft_ps = one_step_metrics_meshft(meshft_net, val_loader, args.dt, device=str(device))

    # ------------ MGN baseline ------------
    mgn_net = MeshGraphNetVF(in_dim=5, hidden=args.mgn_hidden, layers=args.mgn_layers, out_dim=2,
                            edge_in_dim=eattr.shape[1], use_sn=bool(args.use_sn),
                            use_input_norm=use_norm
                            ).to(device)
    mgn_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                            sigma_q=sigma_q_used, sigma_p=sigma_p_used) 
    mgn = {"net": mgn_net, "coords": coords, "src": src, "dst": dst, "eattr": eattr, "node_extras": node_extras}

    print("[train] MGN ...")
    train_mgn(mgn, train_loader, args.dt, args.epochs, device=str(device), use_tqdm=bool(args.tqdm),
          sigma_q=sigma_q, sigma_p=sigma_p)
    mgn_mse, mgn_ps = one_step_metrics_mgn(mgn, val_loader, args.dt, device=str(device))

    # ------------ MGN-HP ------------
    mgnhp_net  = MeshGraphNetVF(in_dim=5, hidden=args.mgn_hidden, layers=args.mgn_layers, out_dim=2,
                                edge_in_dim=eattr.shape[1], use_sn=bool(args.use_sn),
                                use_input_norm=use_norm
                                ).to(device)
    mgnhp_enet = EnergyNet(node_in_dim=5, edge_in_dim=eattr.shape[1], hidden=args.mgn_hidden,
                        layers=args.mgn_layers, use_sn=bool(args.use_sn),
                        use_input_norm=use_norm
                        ).to(device)

    mgnhp_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                sigma_q=sigma_q_used, sigma_p=sigma_p_used)
    mgnhp_enet.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                sigma_q=sigma_q_used, sigma_p=sigma_p_used)

    mgnhp = {"net": mgnhp_net, "energy_net": mgnhp_enet, "coords": coords, "src": src, "dst": dst,
         "eattr": eattr, "node_extras": node_extras}
    print("[train] MGN-HP (λ={:.3g}) ...".format(args.lam_ham))
    train_mgnhp(mgnhp, train_loader, args.dt, args.epochs, device=str(device), lam_ham=float(args.lam_ham),
            use_tqdm=bool(args.tqdm), sigma_q=sigma_q, sigma_p=sigma_p)
    mgnhp_mse, mgnhp_ps = one_step_metrics_mgnhp(mgnhp, val_loader, args.dt, device=str(device))

    # ------------ HNN (separable, damped) ------------
    U_net = _SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                edge_in_dim=eattr.shape[1], use_sn=bool(args.use_sn),
                                use_input_norm=use_norm
                                ).to(device)
    T_net = _SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                edge_in_dim=eattr.shape[1], use_sn=bool(args.use_sn),
                                use_input_norm=use_norm
                                ).to(device)

    U_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                            sigma_field=(sigma_q if args.std_state else 1.0))
    T_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                            sigma_field=(sigma_p if args.std_state else 1.0)) 
    gamma_hnn = None
    if args.damp_mode == "learn_diag_latent":
        r0 = float(args.gamma_min + args.gamma_max)
        gamma_hnn = GammaInferNet(node_in_dim=5, edge_in_dim=eattr.shape[1],
                                hidden=args.gamma_infer_hidden, layers=args.gamma_infer_layers,
                                use_sn=bool(args.use_sn), init_target_rate=r0, init_weight_scale=1e-3,
                                use_input_norm=use_norm
                                ).to(device)
        gamma_hnn.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                    sigma_q=sigma_q_used, sigma_p=sigma_p_used)
    damping_hnn = DampingOp(args.damp_mode, N=coords.shape[0], infer_net=gamma_hnn).to(device)
    hnn = HNNSeparableDamped(U_net, T_net, coords, src, dst, node_extras=node_extras, eattr=eattr,
                             damping=damping_hnn, V0=V0).to(device)
    try:
        omega_h = estimate_omega_max(src, dst, eval_hodge, iters=25)
        hnn._nsub = int(math.ceil(max(1.0, omega_h * args.dt)))
        print(f"[HNN:init] omega_max≈{omega_h:.3e}, dt={args.dt:.3e} -> substeps={hnn._nsub}")
    except Exception as e:
        hnn._nsub = 1
        print(f"[HNN:init] omega_max estimation failed ({e}); using substeps=1.")
    print("[train] HNN (damped) ...")
    train_hnn(hnn, train_loader, args.dt, args.epochs, device=str(device), use_tqdm=bool(args.tqdm),
          sigma_q=sigma_q, sigma_p=sigma_p)
    hnn_mse, hnn_ps = one_step_metrics_hnn(hnn, val_loader, args.dt, device=str(device))

    # meta-batch for rollout
    B = min(8, len(val_ds))
    meta_batch = {
        "t": torch.tensor([val_ds[i][2]["t"] for i in range(B)], device=device),
        "kvec": torch.stack([val_ds[i][2]["kvec"].to(device) for i in range(B)], dim=0),
        "omega": [val_ds[i][2]["omega"] for i in range(B)],
        "phi":   [val_ds[i][2]["phi"]   for i in range(B)],
        "amp":   [val_ds[i][2]["amp"]   for i in range(B)],
        "gamma": torch.tensor([val_ds[i][2]["gamma"] for i in range(B)], device=device),
    }

    meshft_net.eval(); mgn_net.eval(); mgnhp_net.eval(); mgnhp_enet.eval(); hnn.eval()
    # rollouts
    meshft_relF, meshft_relMed, meshft_drift, meshft_pv, meshft_E, meshft_Egt, meshft_EerrF, meshft_EerrMean = rollout_generic(
        "meshft_net", meshft_net, src, dst, coords, eval_hodge, meta_batch, args.dt, args.rollout_T, str(device))
    mgn_relF, mgn_relMed, mgn_drift, mgn_pv, mgn_E, mgn_Egt, mgn_EerrF, mgn_EerrMean = rollout_generic(
        "mgn", mgn, src, dst, coords, eval_hodge, meta_batch, args.dt, args.rollout_T, str(device),
        eattr=eattr, node_extras=node_extras)
    mgnhp_relF, mgnhp_relMed, mgnhp_drift, mgnhp_pv, mgnhp_E, mgnhp_Egt, mgnhp_EerrF, mgnhp_EerrMean = rollout_generic(
        "mgn", mgnhp, src, dst, coords, eval_hodge, meta_batch, args.dt, args.rollout_T, str(device),
        eattr=eattr, node_extras=node_extras)
    hnn_relF, hnn_relMed, hnn_drift, hnn_pv, hnn_E, hnn_Egt, hnn_EerrF, hnn_EerrMean = rollout_generic(
        "hnn", hnn, src, dst, coords, eval_hodge, meta_batch, args.dt, args.rollout_T, str(device))

    # damping stats
    d_meshft = damping_meshft.stats()
    d_hnn = damping_hnn.stats()

    # CSV row
    row = dict(
        seed=args.seed, grid=f"{nx}x{ny}", dt=args.dt, c_speed=args.c_speed, c_wave=args.c_wave,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max, damp_mode=args.damp_mode,
        epochs=args.epochs, batch_size=args.batch_size, train_size=args.train_size, val_size=args.val_size, rollout_T=args.rollout_T,
        std_inputs=int(args.std_inputs),
        std_state=int(args.std_state),
        # one-step
        meshft_mse=meshft_mse, meshft_psnr=meshft_ps, mgn_mse=mgn_mse, mgn_psnr=mgn_ps,
        mgnhp_mse=mgnhp_mse, mgnhp_psnr=mgnhp_ps, hnn_mse=hnn_mse, hnn_psnr=hnn_ps,
        # rollout
        meshft_relF=meshft_relF, meshft_relMed=meshft_relMed, meshft_drift=meshft_drift, meshft_pv=meshft_pv,
        mgn_relF=mgn_relF, mgn_relMed=mgn_relMed, mgn_drift=mgn_drift, mgn_pv=mgn_pv,
        mgnhp_relF=mgnhp_relF, mgnhp_relMed=mgnhp_relMed, mgnhp_drift=mgnhp_drift, mgnhp_pv=mgnhp_pv,
        hnn_relF=hnn_relF, hnn_relMed=hnn_relMed, hnn_drift=hnn_drift, hnn_pv=hnn_pv,
        meshft_EerrF=meshft_EerrF, meshft_EerrMean=meshft_EerrMean,
        mgn_EerrF=mgn_EerrF, mgn_EerrMean=mgn_EerrMean,
        mgnhp_EerrF=mgnhp_EerrF, mgnhp_EerrMean=mgnhp_EerrMean,
        hnn_EerrF=hnn_EerrF, hnn_EerrMean=hnn_EerrMean,
        lam_ham=args.lam_ham,
        **{f"meshft_{k}": v for k,v in d_meshft.items()},
        **{f"hnn_{k}": v for k,v in d_hnn.items()},
    )
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    is_new = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if is_new: w.writerow(list(row.keys()))
        w.writerow([row[k] for k in row.keys()])

    # optional JSON details (energy traces)
    if int(args.save_details)==1:
        details = dict(
            meta=dict(B=meshft_E.shape[1], T=args.rollout_T, dt=args.dt),
            damping={"meshft_net": d_meshft, "hnn": d_hnn},
            energy_traces={
                "MeshFT-Net":  [[float(x) for x in meshft_E[s].detach().cpu().numpy().tolist()] for s in range(meshft_E.shape[0])],
                "MGN":  [[float(x) for x in mgn_E[s].detach().cpu().numpy().tolist()] for s in range(mgn_E.shape[0])],
                "MGN-HP": [[float(x) for x in mgnhp_E[s].detach().cpu().numpy().tolist()] for s in range(mgnhp_E.shape[0])],
                "HNN":  [[float(x) for x in hnn_E[s].detach().cpu().numpy().tolist()] for s in range(hnn_E.shape[0])],
            },
            energy_traces_gt=[
                [float(x) for x in meshft_Egt[s].detach().cpu().numpy().tolist()] for s in range(meshft_Egt.shape[0])
            ],
        )
        with open(os.path.join(args.out_dir, "dissipative_summary.json"), "w") as f:
            json.dump(details, f, indent=2)

    if int(args.save_models) == 1:
        ckpt_dir = args.ckpt_dir or os.path.join(args.out_dir, "ckpts")
        ensure_dir(ckpt_dir)
        torch.save(meshft_net.state_dict(),         os.path.join(ckpt_dir, "meshft_net.pt"))
        torch.save(mgn["net"].state_dict(),          os.path.join(ckpt_dir, "mgn.pt"))
        torch.save(mgnhp["net"].state_dict(),        os.path.join(ckpt_dir, "mgnhp_net.pt"))
        torch.save(mgnhp["energy_net"].state_dict(), os.path.join(ckpt_dir, "mgnhp_energy.pt"))
        torch.save(hnn.state_dict(),                 os.path.join(ckpt_dir, "hnn.pt"))
        with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
            json.dump({
                "args": vars(args),
                "grid": [nx, ny],
                "model_keys": ["MeshFT-Net", "MGN", "MGN-HP(net)", "MGN-HP(energy)", "HNN"]
            }, f, indent=2)
        print(f"[ckpt] saved to: {ckpt_dir}")

    if int(args.make_gifs) == 1:
        gif_dir = args.gif_dir or os.path.join(args.out_dir, "gifs")
        Tgif   = int(args.gif_T) if args.gif_T is not None else int(args.rollout_T)
        models_for_gif = {
            "MeshFT-Net": meshft_net,
            "MGN": mgn,
            "MGN-HP": mgnhp,
            "HNN": hnn,
        }
        generate_val_gifs(
            gif_dir, val_ds, coords, src, dst, V0, eval_hodge,
            models_for_gif, dt=args.dt, T=Tgif, stride=int(args.gif_stride),
            grid_shape=(nx,ny), device=str(device),
            eattr=eattr, node_extras=node_extras,
            n_samples=int(args.gif_n), fps=int(args.gif_fps)
        )

    # terminal summary
    print("\n=== Dissipative benchmark summary ===")
    print(f"One-step MSE/PSNR:")
    print(f"  MeshFT-Net : {meshft_mse:.3e} / {meshft_ps:.2f} dB")
    print(f"  MGN    : {mgn_mse:.3e} / {mgn_ps:.2f} dB")
    print(f"  MGN-HP : {mgnhp_mse:.3e} / {mgnhp_ps:.2f} dB (λ={args.lam_ham})")
    print(f"  HNN    : {hnn_mse:.3e} / {hnn_ps:.2f} dB")
    print(f"Rollout relF | drift | PV-rate:")
    print(f"  MeshFT-Net : {meshft_relF:.3e} | {meshft_drift:.3e} | {meshft_pv:.3e} {(' '+str(d_meshft)) if d_meshft else ''}")
    print(f"  MGN    : {mgn_relF:.3e} | {mgn_drift:.3e} | {mgn_pv:.3e}")
    print(f"  MGN-HP : {mgnhp_relF:.3e} | {mgnhp_drift:.3e} | {mgnhp_pv:.3e}")
    print(f"  HNN    : {hnn_relF:.3e} | {hnn_drift:.3e} | {hnn_pv:.3e} {(' '+str(d_hnn)) if d_hnn else ''}")
    print(f"Energy mismatch vs GT (final | mean, normalized by E_GT(0)):")
    print(f"  MeshFT-Net : {meshft_EerrF:.3e} | {meshft_EerrMean:.3e}")
    print(f"  MGN    : {mgn_EerrF:.3e} | {mgn_EerrMean:.3e}")
    print(f"  MGN-HP : {mgnhp_EerrF:.3e} | {mgnhp_EerrMean:.3e}")
    print(f"  HNN    : {hnn_EerrF:.3e} | {hnn_EerrMean:.3e}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()