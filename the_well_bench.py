#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified benchmark runner for The Well (Acoustics only) with Hamiltonian (and dissipative) models.

Dataset:
  - acoustic_scattering_discontinuous (2D acoustics; near-Hamiltonian; discontinuous media)

Models:
  - MeshFT-Net       (DEC-based Hamiltonian model; KDK symplectic step)
  - MeshFT-Net+Diss  (MeshFT-Net with Rayleigh split dissipation)

Packaging:
  - Canonical state x = [q, s], with s = M * dq/dt.
  - DEC operators on a Cartesian grid (mixed BCs supported); energy diagnostics use a THEORY Hodge
    (M = V0, W = c^2 V1inv) for apples-to-apples comparisons.

Requires:
  - the_well (pip install the_well), torch, numpy, matplotlib, tqdm
"""

import os, math, argparse, random, json, datetime, csv
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
torch.set_float32_matmul_precision('high') 
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# The Well
from the_well.data import WellDataset  # pip install the_well

# ----------------------------- utilities -----------------------------

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try: torch.use_deterministic_algorithms(True)
        except Exception: pass

def to_device(*tensors, device="cpu"):
    return [t.to(device) for t in tensors]

def psnr_q(pred_q: torch.Tensor, tgt_q: torch.Tensor) -> float:
    """PSNR computed per-batch with dynamic peak based on target range."""
    peak = tgt_q.abs().amax().item()
    if peak <= 1e-12: peak = 1.0
    mse = Fnn.mse_loss(pred_q, tgt_q).item()
    if mse <= 1e-20: return 99.0
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)

def _meshft_needed_substeps(dt: float, eattr: torch.Tensor, cfl_target: float) -> int:
    """Compute MeshFT-Net substeps to satisfy 2D CFL using edge length as grid scale."""
    # eattr[...,2] stores |e| from edge_attr_from_coords
    hmin = float(eattr[:, 2].min().item())
    # 2D wave CFL: dt_max ~ h / sqrt(2)
    dt_max = max(1e-12, cfl_target * hmin / math.sqrt(2.0))
    return max(1, int(math.ceil(dt / dt_max)))

def _meshft_step(mdl: nn.Module, x: torch.Tensor, dt: float, aux: Dict[str,torch.Tensor], nsub: int):
    """Advance MeshFT-Net by dt using nsub substeps."""
    if nsub <= 1:
        return mdl(x, dt, aux=aux)
    dts = dt / float(nsub)
    out = x
    for _ in range(nsub):
        out = mdl(out, dts, aux=aux)
    return out

class _MeshFTNetSubstep(nn.Module):
    """One MeshFT-Net substep as an nn.Module (stable for checkpoint_sequential)."""
    def __init__(self, mdl: nn.Module, dts: float, aux: Dict[str, torch.Tensor]):
        super().__init__()
        self.mdl = mdl
        self.dts = float(dts)
        self.aux = aux

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mdl(z, self.dts, aux=self.aux)

def _meshft_step_ckpt(
    mdl: nn.Module,
    x: torch.Tensor,
    dt: float,
    aux: Dict[str, torch.Tensor],
    nsub: int,
    seg_size: int = 0,
):
    if nsub <= 1:
        return mdl(x, dt, aux=aux)

    dts = dt / float(nsub)
    
    if seg_size is None or seg_size <= 1:
        out = x
        for _ in range(nsub):
            out = mdl(out, dts, aux=aux)
        return out

    steps = nn.Sequential(*[_MeshFTNetSubstep(mdl, dts, aux) for _ in range(nsub)])

    import math
    chunks = max(1, math.ceil(nsub / int(seg_size)))

    y = x if x.requires_grad else x.detach().requires_grad_(True)

    out = torch.utils.checkpoint.checkpoint_sequential(steps, chunks, y)
    return out

def _meshft_step_adaptive(
    mdl: nn.Module,
    x: torch.Tensor,
    dt: float,
    aux: Dict[str, torch.Tensor],
    nsub: int,
    threshold: int = 16,
    seg_size: int = 16,
):
    """
    Use the plain loop when nsub < threshold (no splitting);
    otherwise use checkpointed segments with seg_size substeps each.
    """
    if nsub <= 1 or seg_size <= 1 or nsub < max(1, threshold):
        return _meshft_step(mdl, x, dt, aux, nsub)
    return _meshft_step_ckpt(mdl, x, dt, aux, nsub, seg_size=seg_size)

def vfield_mse(x_pred, x_t, x_tp1, dt: float):
    """
    One-step vector-field loss:
      L = || (x_pred - x_t)/dt  -  (x_tp1 - x_t)/dt ||_2^2
    Applies to both channels [q,s]; no q_{t+2} required.
    """
    v_pred = (x_pred - x_t) / max(dt, 1e-12)
    v_true = (x_tp1 - x_t) / max(dt, 1e-12)
    return Fnn.mse_loss(v_pred, v_true)

def apply_shared_sponge_step(x, gamma_bias: torch.Tensor, dt: float):
    """
    Add the same open-boundary sponge to all models:
      s_{t+1} <- s_{t+1} - gamma_bias * s_t * dt
    x: [B,N,2], gamma_bias: [N] on device, dt: float
    """
    if gamma_bias is None: 
        return x
    s = x[..., 1]
    s_new = s - gamma_bias.unsqueeze(0) * s * dt
    return torch.stack([x[...,0], s_new], dim=-1)

# ------------------------- grid & DEC operators -------------------------

def build_grid_mixed_bc(nx: int, ny: int,
                        Lx: float = None, Ly: float = None,
                        bc_x: str = "reflect", bc_y: str = "open",
                        x0: float = 0.0, x1: float = None,
                        y0: float = 0.0, y1: float = None):
    """
    2D Cartesian grid with mixed BCs per-axis.
    You can specify either (Lx,Ly) with (x0,y0) or directly (x0,x1,y0,y1).
    Priority: if x1/y1 given -> use (x0,x1),(y0,y1); else use (x0+Lx, y0+Ly).
    """

    # 終点解決
    if x1 is None:
        assert Lx is not None, "Either x1 or Lx must be provided"
        x1 = x0 + float(Lx)
    if y1 is None:
        assert Ly is not None, "Either y1 or Ly must be provided"
        y1 = y0 + float(Ly)

    Lx_eff = float(x1 - x0)
    Ly_eff = float(y1 - y0)
    hx, hy = Lx_eff / nx, Ly_eff / ny

    # 等間隔座標（endpoint=False 相当）
    xs = x0 + torch.arange(nx, dtype=torch.float32) * hx
    ys = y0 + torch.arange(ny, dtype=torch.float32) * hy

    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)  # [N,2]

    def nid(i, j): return i*ny + j
    src, dst, elen = [], [], []

    # y-neighbors (+y)
    for i in range(nx):
        for j in range(ny-1):
            a = nid(i, j); b = nid(i, j+1)
            src.append(a); dst.append(b); elen.append(hy)
        if bc_y == "periodic":
            a = nid(i, ny-1); b = nid(i, 0)
            src.append(a); dst.append(b); elen.append(hy)

    # x-neighbors (+x)
    for i in range(nx-1):
        for j in range(ny):
            a = nid(i, j); b = nid(i+1, j)
            src.append(a); dst.append(b); elen.append(hx)
    if bc_x == "periodic":
        for j in range(ny):
            a = nid(nx-1, j); b = nid(0, j)
            src.append(a); dst.append(b); elen.append(hx)

    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    elen = torch.tensor(elen, dtype=torch.float32)
    V0   = torch.full((nx*ny,), fill_value=(hx*hy), dtype=torch.float32)
    V1inv = torch.ones_like(elen)
    return coords, src, dst, V0, V1inv

@torch.no_grad()
def B_times_q(src: torch.Tensor, dst: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Apply incidence to node scalar q: (B q)_e = q[dst] - q[src]. Supports [B,N] or [N]."""
    return q[..., dst] - q[..., src]

def BT_times_e(src: torch.Tensor, dst: torch.Tensor, e: torch.Tensor, N: int) -> torch.Tensor:
    """Apply transpose incidence to edge scalar e (accumulate to nodes)."""
    out = torch.zeros(*e.shape[:-1], N, dtype=e.dtype, device=e.device)
    out.index_add_(-1, dst, e)
    out.index_add_(-1, src, -e)
    return out

# ------------------------- Hodge blocks (theory/learned) -------------------------

class HodgeTheory(nn.Module):
    """
    Fixed Hodge with optional spatially-varying wave speed c(x)^2 on edges.

    M = V0 (node diagonal), W acts on edges as W e = (c^2_edge * V1inv) * e.
    If c2_edge is None, uses a global c^2 scalar.
    """
    def __init__(self, V0: torch.Tensor, V1inv: torch.Tensor, c2: float = 1.0,
                 c2_edge: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("V0", V0.clone())
        self.register_buffer("V1inv", V1inv.clone())
        self.c2 = float(c2)
        if c2_edge is not None:
            assert c2_edge.shape == V1inv.shape
            self.register_buffer("c2_edge", c2_edge.clone())
        else:
            self.register_buffer("c2_edge", None)

    def M_vec(self) -> torch.Tensor:
        return self.V0

    def apply_W(self, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, Nnodes: int,
                c2_edge_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Allow per-(batch,edge) override for c^2 on edges.
        c2_edge_override: None | [E] | [B,E]
        """
        if c2_edge_override is not None:
            w = c2_edge_override * self.V1inv            # broadcast with [E]
        elif self.c2_edge is not None:
            w = self.c2_edge * self.V1inv
        else:
            w = self.c2 * self.V1inv
        return w * e

class HodgeLearned(nn.Module):
    """
    Learnable diagonal Hodge:
      M = V0_base * exp(theta_M)                      (node-wise)
      W e = (c2_edge_base * exp(theta_W) * V1inv) * e (edge-wise)
    θ=0 => base (theory). Parts can be disabled.
    """
    def __init__(self,
                 V0_base: torch.Tensor,
                 V1inv: torch.Tensor,
                 c2_edge_base: Optional[torch.Tensor] = None,
                 learn_M: bool = True,
                 learn_W: bool = True):
        super().__init__()
        self.register_buffer("V0_base", V0_base.clone())
        self.register_buffer("V1inv",   V1inv.clone())
        if c2_edge_base is None:
            c2_edge_base = torch.ones_like(V1inv)
        self.register_buffer("c2_edge_base", c2_edge_base.clone())

        self.learn_M = bool(learn_M)
        self.learn_W = bool(learn_W)
        if self.learn_M:
            self.theta_M = nn.Parameter(torch.zeros_like(V0_base))
        else:
            self.register_buffer("theta_M", torch.zeros_like(V0_base))
        if self.learn_W:
            self.theta_W = nn.Parameter(torch.zeros_like(V1inv))
        else:
            self.register_buffer("theta_W", torch.zeros_like(V1inv))

    def M_vec(self) -> torch.Tensor:
        return self.V0_base * torch.exp(self.theta_M)

    def apply_W(self, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, Nnodes: int,
                c2_edge_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Combine learned log-scale with optional sample-dependent c^2.
        """
        theta = torch.exp(self.theta_W)  # [E]
        w_edge = ((c2_edge_override if c2_edge_override is not None else self.c2_edge_base) * theta) * self.V1inv
        return w_edge * e

    def reg_loss(self) -> torch.Tensor:
        return (self.theta_M.pow(2).mean() + self.theta_W.pow(2).mean())

# ------------------------- Local-MLP Hodge (node/edge-wise) -------------------------

class HodgeLocalMLP(HodgeTheory):
    """
    Local, data-driven Hodge using small MLPs:
      - Node mass:    M_node = V0_base * exp( clip(tanh(MLP_node(...)) * logscale_max_M) )
      - Edge weight:  c2_edge_scale = exp( clip(tanh(MLP_edge(...)) * logscale_max_W) )
        Effective W on edges is built as:
          W_edge = (c2_edge_base_from_data * c2_edge_scale) * V1inv
        where c2_edge_base_from_data comes from aux['c'] if available, otherwise 1.0.

    Notes:
      * This class stores coords/src/dst/eattr internally, so MeshFT-Net doesn't need to.
      * Training/loss are unchanged; MeshFT-Net will call .node_mass_local() and .edge_c2_scale().
    """
    def __init__(self,
                 V0_base: torch.Tensor,
                 V1inv: torch.Tensor,
                 coords: torch.Tensor,
                 src: torch.Tensor,
                 dst: torch.Tensor,
                 eattr: torch.Tensor,
                 learn_M: bool = True,
                 learn_W: bool = True,
                 node_hidden: int = 32,
                 edge_hidden: int = 32,
                 node_layers: int = 2,
                 edge_layers: int = 2,
                 logscale_max_M: float = 2.0,
                 logscale_max_W: float = 2.0,
                 use_q: bool = True,
                 use_s: bool = True
                 ):
        super().__init__(V0_base, V1inv, c2=1.0, c2_edge=None)
        # geometry cached inside
        self.register_buffer("coords_buf", coords.clone())   # [N,2]
        self.register_buffer("src_buf", src.clone())         # [E]
        self.register_buffer("dst_buf", dst.clone())         # [E]
        self.register_buffer("eattr_buf", eattr.clone())     # [E,3] = (dx,dy,|e|)

        self.learn_M = bool(learn_M)
        self.learn_W = bool(learn_W)
        self.logscale_max_M = float(logscale_max_M)
        self.logscale_max_W = float(logscale_max_W)
        self.use_q = bool(use_q)
        self.use_s = bool(use_s)

        # Small MLPs (zero-centered outputs are preferred -> use Tanh via SiLU stacks)
        # Node features: [q, s, x, y, (optional aux... fed externally inside MeshFT-Net)]
        self.node_in_dim = 10  # [q,s,x,y] + up to 6 small aux (rho,u,v,c,A,temperature)
        self.node_mlp = MLP(self.node_in_dim, node_hidden, 1, layers=node_layers) if self.learn_M else None

        # Edge features: [q_i,q_j,s_i,s_j, dx,dy,|e|, (optional c_i,c_j,A_i,A_j...)]
        in_edge = 7  # base edge features without aux
        self.edge_mlp = MLP(in_edge, edge_hidden, 1, layers=edge_layers) if self.learn_W else None

    # --- helpers to build dynamic features ---

    def _gather_edge(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # T: [B,N] -> gather at edges -> [B,E]
        return T.index_select(1, self.src_buf), T.index_select(1, self.dst_buf)

    # --- public API used by MeshFT-Net ---

    def node_mass_local(self, q, s, aux=None):
        B, N = q.shape
        base = self.V0.unsqueeze(0).expand(B, -1)
        if not self.learn_M or self.node_mlp is None:
            return base

        xy = self.coords_buf.unsqueeze(0).expand(B,-1,-1)
        q_in = q if self.use_q else 0.0 * q
        s_in = s if self.use_s else 0.0 * s

        parts = [q_in.unsqueeze(-1), s_in.unsqueeze(-1), xy]  # -> [B,N,4]
        if isinstance(aux, dict):
            for k in ("A","rho","u","v","temperature","c"):
                if k in aux:
                    parts.append(aux[k].unsqueeze(-1))         # -> [B,N,?]

        Fnode = torch.cat(parts, dim=-1)                       # [B,N,Fvar]
        X = Fnode.view(B*N, -1)                                # [B*N,Fvar]

        # pad / truncate to self.node_in_dim
        F = self.node_in_dim
        if X.shape[1] < F:
            pad = torch.zeros(B*N, F - X.shape[1], device=X.device, dtype=X.dtype)
            X = torch.cat([X, pad], dim=1)
        elif X.shape[1] > F:
            X = X[:, :F]

        out = self.node_mlp(X).view(B, N)
        scale = torch.exp(torch.tanh(out) * self.logscale_max_M)
        return base * scale

    def edge_c2_scale(self, q: torch.Tensor, s: torch.Tensor, aux: Optional[Dict[str,torch.Tensor]] = None) -> torch.Tensor:
        """
        Predict per-edge multiplicative scale for c^2. Returns [B,E] >= 0:
          c2_scale = exp( tanh(edge_mlp(.)) * logscale_max_W )
        MeshFT-Net will multiply this to data-provided c^2_edge (if available), else it becomes the c^2 itself.
        """
        B, N = q.shape
        E = self.src_buf.numel()
        if not self.learn_W or self.edge_mlp is None:
            return torch.ones(B, E, device=q.device, dtype=q.dtype)

        qi, qj = self._gather_edge(q)       # [B,E]
        si, sj = self._gather_edge(s)       # [B,E]

        if not self.use_q:
            qi = 0.0 * qi; qj = 0.0 * qj
        if not self.use_s:
            si = 0.0 * si; sj = 0.0 * sj

        eattr = self.eattr_buf.unsqueeze(0).expand(B, -1, -1)  # [B,E,3]
        feats = [qi.unsqueeze(-1), qj.unsqueeze(-1), si.unsqueeze(-1), sj.unsqueeze(-1), eattr]  # -> [...,1] x4 + [...,3]
        # (optional aux on edges)
        if isinstance(aux, dict):
            for k in ("c", "A"):
                if k in aux:
                    ai, aj = self._gather_edge(aux[k])  # [B,E]
                    feats.append(ai.unsqueeze(-1)); feats.append(aj.unsqueeze(-1))
        Fedge = torch.cat(feats, dim=-1)                 # [B,E,F]
        BEF   = Fedge.view(B*E, -1)
        out   = self.edge_mlp(BEF).view(B, E)            # unconstrained
        scale = torch.exp(torch.tanh(out) * self.logscale_max_W)
        return scale

    # Keep a trivial reg to satisfy existing calls; rely on weight decay in practice.
    def reg_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.V0.device)

# ------------------------- MeshFT-Net core (+ dissipative split) -------------------------

class MeshFTNet(nn.Module):
    """
    Hamiltonian Mesh Network (DEC-based) with optional local-MLP Hodge and local dissipation.
      State x = [q, s], s = M dq/dt.
      Energy H = 0.5 * (B q)^T W (B q) + 0.5 * s^T M^{-1} s.

    Time stepping: KDK symplectic. Optional Strang split for dissipation.
    """
    def __init__(self, src, dst, hodge: HodgeTheory, dissipative_op=None):
        super().__init__()
        self.src = src; self.dst = dst
        # support both fixed and dynamic M
        Mvec = hodge.M_vec()
        self.N = int(Mvec.numel())
        self.hodge = hodge
        self.dissipative_op = dissipative_op  # (q,s,aux) -> (dq_d, ds_d)

    def energy(self, x):
        q, s = x[...,0], x[...,1]
        M = self.hodge.M_vec()
        Bq = B_times_q(self.src, self.dst, q)
        W_Bq = self.hodge.apply_W(Bq, self.src, self.dst, self.N)
        term_q = 0.5 * (Bq * W_Bq).sum(dim=-1)
        term_s = 0.5 * ((s**2) / (M + 1e-12)).sum(dim=-1)
        return term_q + term_s

    def _mass_batched(self, q, s, aux):
        """
        Get batched M[B,N] if available; otherwise broadcast static V0[N].
        """
        M = self.hodge.M_vec()                  # [N]
        if hasattr(self.hodge, "node_mass_local"):
            M_loc = self.hodge.node_mass_local(q, s, aux)  # [B,N]
            return M_loc
        else:
            return M.unsqueeze(0).expand_as(q)  # [B,N]

    def _kick(self, q, s, dt, aux: Optional[Dict[str, torch.Tensor]] = None):
        """
        s <- s - dt * K q with possible sample/edge-wise W through c^2 overrides.
        Combines data-provided c (if any) with learned local edge scale (if provided).
        """
        Bq = B_times_q(self.src, self.dst, q)         # [B,E]
        c2_edge_override = None

        # (1) build c^2 from aux['c'] if available: avg of src/dst squares -> [B,E]
        if isinstance(aux, dict) and ("c" in aux):
            c = aux["c"]                              # [B,N]
            ci = c.index_select(1, self.src)         # [B,E]
            cj = c.index_select(1, self.dst)         # [B,E]
            c2_edge_override = 0.5 * (ci*ci + cj*cj) # [B,E]

        # (2) local multiplicative scale from HodgeLocalMLP (if provided)
        if hasattr(self.hodge, "edge_c2_scale"):
            scale = self.hodge.edge_c2_scale(q, s, aux)  # [B,E] >= 0
            c2_edge_override = scale if c2_edge_override is None else (c2_edge_override * scale)

        # (3) apply W with override (supports [E] or [B,E])
        W_Bq = self.hodge.apply_W(Bq, self.src, self.dst, self.N, c2_edge_override=c2_edge_override)
        Kq = BT_times_e(self.src, self.dst, W_Bq, self.N)
        return s - dt * Kq

    def kdk(self, q, s, dt, aux=None):
        M = self._mass_batched(q, s, aux)            # [B,N]
        Minv = 1.0 / (M + 1e-12)
        s_half = self._kick(q, s, 0.5*dt, aux=aux)
        q_new  = q + dt * (Minv * s_half)
        s_new  = self._kick(q_new, s_half, 0.5*dt, aux=aux)
        return q_new, s_new

    def forward(self, x, dt: float, aux: Optional[Dict[str,torch.Tensor]] = None):
        q, s = x[...,0], x[...,1]
        if self.dissipative_op is not None:
            dq_d, ds_d = self.dissipative_op(q, s, aux)
            q = q + 0.5*dt * dq_d
            s = s + 0.5*dt * ds_d
        q, s = self.kdk(q, s, dt, aux=aux)
        if self.dissipative_op is not None:
            dq_d, ds_d = self.dissipative_op(q, s, aux)
            q = q + 0.5*dt * dq_d
            s = s + 0.5*dt * ds_d
        return torch.stack([q, s], dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2, use_sn: bool=False):
        super().__init__()
        dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
        mods=[]
        for i in range(len(dims)-1):
            lin = nn.Linear(dims[i], dims[i+1])
            if use_sn: lin = nn.utils.spectral_norm(lin)
            mods.append(lin)
            if i < len(dims)-2: mods.append(nn.SiLU())
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

# ------------------------- edge features (Cartesian torus) -------------------------

def edge_attr_from_coords_bc(coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Non-periodic edge features: (dx, dy, |e|) without minimum-image wrap."""
    dv = coords[dst] - coords[src]
    elen = dv.norm(dim=-1, keepdim=True)
    return torch.cat([dv, elen], dim=-1)

# ------------------------- dissipative operators (learned everywhere) -------------------------

class LearnedRayleighDiss(nn.Module):
    """Learn gamma(x_t) >= 0 from local state (+ optional aux). Acts on s only."""
    def __init__(self, in_features: int = 2, hidden: int = 64, layers: int = 3, min_gamma: float = 0.0):
        super().__init__()
        self.in_features = int(in_features)
        self.mlp = MLP(self.in_features, hidden, 1, layers=layers)
        self.min_gamma = float(min_gamma)

    def forward(self, q, s, aux=None):
        feats = [q, s]
        if isinstance(aux, dict):
            for k in ("c"):
                if k in aux: feats.append(aux[k])
        Fnode = torch.stack(feats, dim=-1)  # [B, N, Fdim]
        B, N, Fdim = Fnode.shape
        X = Fnode.view(B*N, Fdim)
        if Fdim < self.in_features:
            pad = torch.zeros(B*N, self.in_features - Fdim, device=X.device, dtype=X.dtype)
            X = torch.cat([X, pad], dim=-1)
        elif Fdim > self.in_features:
            X = X[:, :self.in_features]

        gamma = Fnn.softplus(self.mlp(X)).view(B, N) + self.min_gamma
        dq_diss = torch.zeros_like(q)
        ds_diss = - gamma * s
        return dq_diss, ds_diss

# ------------------------- Local-MLP Rayleigh dissipation -------------------------

class RayleighDissLocalMLP(nn.Module):
    """Local gamma(x) >= 0; optional per-node gamma_bias for sponge near open boundaries."""
    def __init__(self, coords: torch.Tensor, hidden: int = 32, layers: int = 2,
                 min_gamma: float = 0.0, gamma_bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("coords_buf", coords.clone())
        in_dim = 8  # [q,s,x,y] + up to 4 aux, padded/truncated to 8
        self.mlp = MLP(in_dim, hidden, 1, layers=layers)
        self.min_gamma = float(min_gamma)
        self.register_buffer("gamma_bias", gamma_bias.clone() if gamma_bias is not None else None)

    def forward(self, q, s, aux=None):
        B, N = q.shape
        xy = self.coords_buf.unsqueeze(0).expand(B,-1,-1)
        feats = [q.unsqueeze(-1), s.unsqueeze(-1), xy]
        if isinstance(aux, dict):
            for k in ("c"):
                if k in aux: feats.append(aux[k].unsqueeze(-1))
        F = torch.cat(feats, dim=-1)
        if F.shape[-1] < 8:
            pad = torch.zeros(B, N, 8-F.shape[-1], device=F.device, dtype=F.dtype)
            F = torch.cat([F, pad], dim=-1)
        elif F.shape[-1] > 8:
            F = F[:, :, :8]
        out = self.mlp(F.view(B*N, -1)).view(B, N)
        gamma = Fnn.softplus(out) + self.min_gamma
        if self.gamma_bias is not None:
            gamma = gamma + self.gamma_bias.unsqueeze(0)
        dq_diss = torch.zeros_like(q)
        ds_diss = - gamma * s
        return dq_diss, ds_diss

# ------------------------- The Well -> pairwise dataset adapter -------------------------

class WellPairDataset(Dataset):
    """
    Produce (x_t, x_{t+1}, aux) where s-components are intended to be built
    via centered differences from data (using extra frames when available).
    Keys: 'input_fields' [T_in,H,W,F], 'output_fields' [T_out,H,W,F]
    """
    def __init__(self, base_path: str, dataset_name: str, split: str,
                 field_name_q: str, dt_data: Optional[float],
                 take_aux: bool, aux_keys: Tuple[str, ...],
                 nx: int, ny: int, Lx: float, Ly: float):
        super().__init__()
        self.ds = WellDataset(
            well_base_path=base_path,
            well_dataset_name=dataset_name,
            well_split_name=split,
            # Need q_{t-1}, q_t, q_{t+1}, q_{t+2} for centered s labels
            n_steps_input=2,
            n_steps_output=2,
            flatten_tensors=True,
            return_grid=True
        )
        self.Nseq = len(self.ds)
        self.dt = float(dt_data) if dt_data is not None else 1.0
        self.take_aux = bool(take_aux)
        self.aux_keys = aux_keys

        self.field_order = []
        meta = getattr(self.ds, "metadata", None)
        if meta is not None and hasattr(meta, "field_names"):
            for _, names in meta.field_names.items():
                self.field_order += list(names)
        self.fi_q = self._safe_index(field_name_q, default=0)
        self.fi_A = self._safe_index("A", default=None)
        self.fi_c = self._safe_index("c", default=None, aliases=("c", "sound_speed", "speed_of_sound"))

        self.fi_rho = self._safe_index("rho", default=None, aliases=("density",))
        self.fi_u   = self._safe_index("u",   default=None, aliases=("velocity_x"))
        self.fi_v   = self._safe_index("v",   default=None, aliases=("velocity_y"))

    def _safe_index(self, name: str, default=None, aliases: Tuple[str, ...]=()):
        if not self.field_order:
            return default
        for n in [name]+list(aliases):
            if n in self.field_order:
                return self.field_order.index(n)
        return default

    def __len__(self):
        return max(1, self.Nseq) * 32

    def __getitem__(self, idx):
        k = random.randrange(self.Nseq)
        item = self.ds[k]
        x_in  = item["input_fields"]
        x_out = item["output_fields"]
        if isinstance(x_in, np.ndarray):  x_in  = torch.from_numpy(x_in)
        if isinstance(x_out, np.ndarray): x_out = torch.from_numpy(x_out)
        Tin, Tout = x_in.shape[0], x_out.shape[0]
        q_tm1 = x_in[-2, :, :, self.fi_q] if Tin  >= 2 else x_in[-1, :, :, self.fi_q]
        q_t   = x_in[-1, :, :, self.fi_q]
        q_tp1 = x_out[0, :, :, self.fi_q]
        q_tp2 = x_out[1, :, :, self.fi_q] if Tout >= 2 else x_out[0, :, :, self.fi_q]
        aux = {}
        if self.take_aux and self.aux_keys:
            for key in self.aux_keys:
                if key == "c" and self.fi_c is not None:
                    aux["c"] = x_in[-1, :, :, self.fi_c]
        return {
            "q_tm1": q_tm1.float(),
            "q_t":   q_t.float(),
            "q_tp1": q_tp1.float(),
            "q_tp2": q_tp2.float(),
            "aux": {k: v.float() for k, v in aux.items()}}

def collate_pairs(batch, V0: torch.Tensor, dt_scalar: float):
    """
    Build (x_t, x_{t+1}, aux) without using q_{t+2}.
    s_t   = M * (q_{t+1} - q_{t-1}) / (2 dt)   (centered around t)
    s_{t+1} ≈ M * (q_{t+1} - q_t) / dt         (forward at t+1)   <-- no q_{t+2}
    """
    q_tm1 = torch.stack([b["q_tm1"] for b in batch], dim=0)
    q_t   = torch.stack([b["q_t"]   for b in batch], dim=0)
    q_tp1 = torch.stack([b["q_tp1"] for b in batch], dim=0)
    B,H,W = q_t.shape; N = H*W
    q_tm1_f, q_t_f, q_tp1_f = q_tm1.view(B,N), q_t.view(B,N), q_tp1.view(B,N)

    M_vec = V0.to(q_t_f.device); dt = float(dt_scalar)
    s_t = M_vec.unsqueeze(0) * (q_tp1_f - q_tm1_f) / (2.0*dt)
    # per-sample fallback when q_tm1 ≈ q_t
    same = (q_tm1_f - q_t_f).abs().amax(dim=1, keepdim=True) < 1e-12  # [B,1]
    s_t_fwd = M_vec.unsqueeze(0) * (q_tp1_f - q_t_f) / dt
    s_t = torch.where(same, s_t_fwd, s_t)  # broadcast on feature dim
    # fallback if q_tm1 == q_t (rare)
    mask_same = torch.allclose(q_tm1_f, q_t_f)
    if mask_same:
        s_t = M_vec.unsqueeze(0) * (q_tp1_f - q_t_f) / dt
    # forward difference for s_{t+1}
    s_tp1 = M_vec.unsqueeze(0) * (q_tp1_f - q_t_f) / dt

    x_t   = torch.stack([q_t_f,   s_t],   dim=-1)
    x_tp1 = torch.stack([q_tp1_f, s_tp1], dim=-1)

    # aux passthrough
    aux = {}
    keys = set().union(*[set(b["aux"].keys()) for b in batch])
    for k in keys:
        vals = [b["aux"].get(k, torch.zeros_like(b["q_t"])) for b in batch]
        aux[k] = torch.stack(vals, dim=0).view(B, N)
    return x_t, x_tp1, aux

# ------------------------- training / evaluation helpers -------------------------

def mse_masked(a, b):
    return Fnn.mse_loss(a, b)

@torch.no_grad()
def rollout_rel_error_any(model, x0, coords, src, dst, eattr, dt, Tsteps,
                          M_data, eval_hodge=None, aux=None,
                          meshft_nsub=1, meshft_ckpt_threshold=32, meshft_ckpt_seg=16,
                          sponge_mode="off", gamma_bias=None):
    x = x0.clone()
    N = coords.shape[0]

    def energy(z):
        if eval_hodge is None:
            q, s = z[...,0], z[...,1]
            return 0.5*(q.pow(2).sum(dim=-1) + s.pow(2).sum(dim=-1))
        q, s = z[...,0], z[...,1]
        Bq  = q.index_select(1, dst) - q.index_select(1, src)
        WBq = eval_hodge.apply_W(Bq, src, dst, N)
        term_q = 0.5*(Bq*WBq).sum(dim=-1)
        term_s = 0.5*((s**2)/(eval_hodge.M_vec()+1e-12)).sum(dim=-1)
        return term_q + term_s

    e0 = energy(x)
    for _ in range(Tsteps):
        x = _meshft_step_adaptive(model, x, dt, aux or {}, meshft_nsub,
                                  threshold=meshft_ckpt_threshold, seg_size=meshft_ckpt_seg)
        if sponge_mode == "shared" and (gamma_bias is not None):
            x = apply_shared_sponge_step(x, gamma_bias, dt)
        if not torch.isfinite(x).all():
            return float("inf"), float("inf")

    ef = energy(x)
    drift = ((ef - e0).abs() / (e0.abs() + 1e-12)).mean().item()
    q, s = x[...,0], x[...,1]
    dqdt = s / (M_data + 1e-12)
    q_gt = q + dt * dqdt
    rel_final = (x.reshape(x.shape[0],-1) - torch.stack([q_gt,s],dim=-1).reshape(x.shape[0],-1)) \
                .norm(dim=-1) / (torch.stack([q_gt,s],dim=-1).reshape(x.shape[0],-1).norm(dim=-1) + 1e-12)
    return rel_final.mean().item(), drift

@torch.no_grad()
def evaluate_models(models, loader, coords, src, dst, eattr, dt, V0, eval_hodge,
                    steps_limit=50, rollout_T=10, meshft_nsub=1, device=torch.device("cpu"),
                    sponge_mode="off", gamma_bias=None, meshft_ckpt_threshold=0, meshft_ckpt_seg=0):
    name, (kind, net) = next(iter(models.items()))
    net.eval()
    agg = {"MSE_q":0.0,"MAE_q":0.0,"RelL2_x":0.0,"PSNR_q":0.0,"RollRel":0.0,"EnergyDrift":0.0,"count":0}
    it = iter(loader)
    for _ in range(steps_limit):
        try:
            x_t, x_tp1, aux = next(it)
        except StopIteration:
            break
        x_t   = x_t.to(device)
        x_tp1 = x_tp1.to(device)
        aux_d = {k:v.to(device) for k,v in aux.items()}

        # one-step
        x_pred = _meshft_step_adaptive(net, x_t, dt, aux_d, meshft_nsub,
                                       threshold=meshft_ckpt_threshold, seg_size=meshft_ckpt_seg)
        if sponge_mode == "shared" and (gamma_bias is not None):
            x_pred = apply_shared_sponge_step(x_pred, gamma_bias, dt)

        q_pred = x_pred[...,0]; q_tp1 = x_tp1[...,0]
        mse_q = Fnn.mse_loss(q_pred, q_tp1).item()
        mae_q = Fnn.l1_loss(q_pred, q_tp1).item()
        rel   = (x_pred.reshape(x_pred.shape[0],-1) - x_tp1.reshape(x_tp1.shape[0],-1)).norm(dim=-1) / \
                (x_tp1.reshape(x_tp1.shape[0],-1).norm(dim=-1) + 1e-12)
        ps    = psnr_q(q_pred, q_tp1)

        # rollout proxy
        rel_roll, drift = rollout_rel_error_any(net, x_t[:1], coords, src, dst, eattr, dt, rollout_T,
                                                V0, eval_hodge, aux=aux_d,
                                                meshft_nsub=meshft_nsub,
                                                meshft_ckpt_threshold=meshft_ckpt_threshold,
                                                meshft_ckpt_seg=meshft_ckpt_seg,
                                                sponge_mode=sponge_mode, gamma_bias=gamma_bias)

        agg["MSE_q"] += mse_q; agg["MAE_q"] += mae_q; agg["RelL2_x"] += rel.mean().item()
        agg["PSNR_q"] += ps;  agg["RollRel"] += rel_roll; agg["EnergyDrift"] += drift; agg["count"] += 1

    c = max(1, agg["count"])
    return {name:{k:(agg[k]/c if k!="count" else c) for k in agg if k!="count"}}

def pretty_print_metrics(title: str, metrics: Dict[str, Dict[str, float]]):
    keys = ["MSE_q","MAE_q","RelL2_x","PSNR_q","RollRel","EnergyDrift"]
    print(f"\n=== {title} ===")
    header = "Model".ljust(8) + " ".join([k.rjust(14) for k in keys])
    print(header)
    for name, m in metrics.items():
        row = name.ljust(12)
        for k in keys:
            if k in m:
                row += f"{m[k]:14.3e}" if k!="PSNR_q" else f"{m[k]:14.2f}"
            else:
                row += f"{'NA':>14}"
        print(row)
    print()

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer], epoch: int, args: argparse.Namespace, extra: Dict[str,Any]=None):
    obj = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": (optimizer.state_dict() if optimizer is not None else None),
        "args": vars(args),
        "extra": extra or {}
    }
    torch.save(obj, str(path))

def append_metrics_csv(csv_path: Optional[str], epoch: int, split_name: str, metrics: Dict[str, Dict[str,float]]):
    if not csv_path: return
    exists = os.path.exists(csv_path)
    keys = ["MSE_q","MAE_q","RelL2_x","PSNR_q","RollRel","EnergyDrift"]
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["epoch","split","model"] + keys)
        for model_name, m in metrics.items():
            row = [epoch, split_name, model_name] + [m.get(k, float("nan")) for k in keys]
            w.writerow(row)

# ------------------------- main runner -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str, default="hf://datasets/polymathic-ai/")
    ap.add_argument("--dataset", type=str, default="acoustic_scattering_discontinuous", choices=["acoustic_scattering_discontinuous"])
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val", help="validation split name; falls back if missing")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # training setup
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--val_steps", type=int, default=50)
    ap.add_argument("--roll_T", type=int, default=10)

    # model hparams
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)

    # I/O (kept compatible with your bash script)
    ap.add_argument("--out_dir", type=str, default="./runs")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default=None, help="override out_dir/run_name if set")
    ap.add_argument("--save_models", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--bench_csv", type=str, default=None, help="path to append CSV metrics")

    # legacy knobs (kept for CLI compatibility)
    ap.add_argument("--rayleigh_gamma", type=float, default=0.0)

    # MeshFT-Net Hodge learning
    ap.add_argument("--meshft_learn_hodge", type=int, default=1, help="Learn Hodge in MeshFT-Net (1) or keep theory Hodge (0).")
    ap.add_argument("--meshft_learn_M",     type=int, default=1, help="Learn node mass M (V0).")
    ap.add_argument("--meshft_learn_W",     type=int, default=1, help="Learn edge weight W (c^2 on edges).")
    ap.add_argument("--hodge_reg",       type=float, default=0.0, help="L2 reg weight on Hodge log-params (theta).")

    # MeshFT-Net local-MLP Hodge knobs
    ap.add_argument("--meshft_local_hodge", type=int, default=1, help="Use local MLPs to predict Hodge (M/W) from fields/geometry.")
    ap.add_argument("--meshft_node_hidden", type=int, default=32)
    ap.add_argument("--meshft_edge_hidden", type=int, default=32)
    ap.add_argument("--meshft_logscale_max_M", type=float, default=2.0, help="Max |log-scale| for node mass.")
    ap.add_argument("--meshft_logscale_max_W", type=float, default=2.0, help="Max |log-scale| for edge c^2.")

    # Local MLP Rayleigh dissipation
    ap.add_argument("--diss_local", type=int, default=1, help="Use local MLP Rayleigh dissipation on s.")
    ap.add_argument("--diss_hidden", type=int, default=32)

    # Fair comparison: drop all auxiliary fields for all models
    ap.add_argument("--no_aux", type=int, default=1,
                    help="If 1, load & feed no auxiliary fields (c, A, ...) to any model.")

    # MeshFT-Net CFL-safe substepping
    ap.add_argument("--meshft_cfl", type=float, default=0.5,
                    help="Target per-substep 2D CFL (<= ~1/sqrt(2) is stable).")
    ap.add_argument("--meshft_roll_substeps", type=int, default=0,
                    help="If >0, force this many MeshFT-Net substeps per dt in train/eval; "
                         "if 0, choose automatically from CFL.")

    ap.add_argument("--meshft_hodge_use_q", type=int, default=0,
                    help="If 0, ignore q in HodgeLocalMLP features (state-indep).")
    ap.add_argument("--meshft_hodge_use_s", type=int, default=0,
                    help="If 0, ignore s in HodgeLocalMLP features (state-indep).")

    ap.add_argument("--bc_x", type=str, default="reflect",  choices=["periodic","reflect","dirichlet0","open"])
    ap.add_argument("--bc_y", type=str, default="open",     choices=["periodic","reflect","dirichlet0","open"])
    ap.add_argument("--sponge_frac", type=float, default=0.08, help="fraction of Ly (top/bottom) for sponge")
    ap.add_argument("--sponge_gamma_max", type=float, default=2.0, help="max extra Rayleigh gamma at edge")
    ap.add_argument("--sponge_mode", type=str, default="shared",
                choices=["off","shared","meshft_only"],
                help="'shared' = apply sponge to all models' updates; "
                     "'meshft_only' = sponge only inside MeshFT-Net diss_op; "
                     "'off' = no sponge anywhere.")
    # MeshFT-Net substep execution controls
    ap.add_argument("--meshft_ckpt_seg", type=int, default=64,
                    help="Substep checkpoint segment size for MeshFT-Net; 0 disables.")
    ap.add_argument("--meshft_ckpt_threshold", type=int, default=64,
                    help="Use checkpointing only when substeps exceed this.")

    ap.add_argument("--meshft_diss_mode",
                    type=str,
                    default="none",
                    choices=["none", "rayleigh_local"],
                    help="Dissipation in MeshFT-Net split step: "
                        "'none' (no diss), 'rayleigh_local' (ds=-gamma*s)")

    ap.add_argument("--x_min", type=float, default=None)
    ap.add_argument("--x_max", type=float, default=None)
    ap.add_argument("--y_min", type=float, default=None)
    ap.add_argument("--y_max", type=float, default=None)

    args = ap.parse_args()
    set_seed(args.seed)

    if args.dataset == "acoustic_scattering_discontinuous":
        nx, ny = 256, 256
        x0_default, x1_default = -1.0, 1.0
        y0_default, y1_default = -1.0, 1.0
        field_q = "pressure"; dt_data = 2.0/101.0
        take_aux = True; aux_keys = ("c",)

    x0 = args.x_min if args.x_min is not None else x0_default
    x1 = args.x_max if args.x_max is not None else x1_default
    y0 = args.y_min if args.y_min is not None else y0_default
    y1 = args.y_max if args.y_max is not None else y1_default

    Lx = float(x1 - x0)
    Ly = float(y1 - y0)

    # Force-disable aux for fair comparison if requested
    if args.no_aux:
        take_aux = False
        aux_keys = tuple()

    # --- geometry on CPU, then move to device ---
    coords_cpu, src_cpu, dst_cpu, V0_cpu, V1inv_cpu = build_grid_mixed_bc(
        nx, ny, Lx=Lx, Ly=Ly, bc_x=args.bc_x, bc_y=args.bc_y, x0=x0, x1=x1, y0=y0, y1=y1
    )
    eattr_cpu = edge_attr_from_coords_bc(coords_cpu, src_cpu, dst_cpu)
    device = torch.device(args.device)
    coords = coords_cpu.to(device); src = src_cpu.to(device); dst = dst_cpu.to(device)
    eattr  = eattr_cpu.to(device);  V0  = V0_cpu.to(device);  V1inv = V1inv_cpu.to(device)

    # dataset adapters
    ds_train = WellPairDataset(args.base_path, args.dataset, args.split,
                               field_name_q=field_q, dt_data=dt_data,
                               take_aux=take_aux, aux_keys=aux_keys,
                               nx=nx, ny=ny, Lx=Lx, Ly=Ly)

    # validation split with fallbacks
    def _make_val_ds():
        for s in [args.val_split, "validation", "valid", "test", args.split]:
            try:
                _ds = WellPairDataset(args.base_path, args.dataset, s,
                                      field_name_q=field_q, dt_data=dt_data,
                                      take_aux=take_aux, aux_keys=aux_keys,
                                      nx=nx, ny=ny, Lx=Lx, Ly=Ly)
                if len(_ds) > 0:
                    return _ds, s
            except Exception:
                continue
        return ds_train, args.split
    ds_val, val_used = _make_val_ds()

    loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_pairs(b, V0_cpu, dt_data)
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_pairs(b, V0_cpu, dt_data)
    )

    # probe aux to assemble c^2 edges if available
    c2_edge = None
    present_aux = set()
    if not args.no_aux:
        try:
            _, _, aux_probe = next(iter(loader))
            present_aux = set(aux_probe.keys())
            if "c" in present_aux:
                c_vec = aux_probe["c"][0].to(device)  # [N]
                c2_edge = 0.5 * (c_vec[src]**2 + c_vec[dst]**2)
        except StopIteration:
            aux_probe = {}

    # THEORY Hodge for evaluation (fixed; for fair energy diagnostics)
    eval_hodge = HodgeTheory(V0, V1inv, c2=1.0, c2_edge=c2_edge).to(device)

    # --- MeshFT-Net (with dissipative op + local-MLP Hodge if enabled) ---
    if args.meshft_learn_hodge and args.meshft_local_hodge:
        # Local MLP Hodge (node/edge-wise), robust and expressive but still structure-preserving
        hodge_meshft = HodgeLocalMLP(
            V0, V1inv, coords, src, dst, eattr,
            learn_M=bool(args.meshft_learn_M),
            learn_W=bool(args.meshft_learn_W),
            node_hidden=args.meshft_node_hidden,
            edge_hidden=args.meshft_edge_hidden,
            node_layers=2, edge_layers=2,
            logscale_max_M=args.meshft_logscale_max_M,
            logscale_max_W=args.meshft_logscale_max_W,
            use_q=bool(args.meshft_hodge_use_q),
            use_s=bool(args.meshft_hodge_use_s),
        ).to(device)
    elif args.meshft_learn_hodge:
        # Global (diagonal) learned Hodge as a fallback
        if args.dataset == "gray_scott_reaction_diffusion":
            hodge_meshft = HodgeLearned(V0, V1inv, c2_edge_base=None,
                                    learn_M=bool(args.meshft_learn_M),
                                    learn_W=bool(args.meshft_learn_W)).to(device)
        else:
            hodge_meshft = HodgeLearned(V0, V1inv, c2_edge_base=c2_edge,
                                    learn_M=bool(args.meshft_learn_M),
                                    learn_W=bool(args.meshft_learn_W)).to(device)
    else:
        # Fixed theory Hodge
        if args.dataset == "gray_scott_reaction_diffusion":
            hodge_meshft = HodgeTheory(V0, V1inv, c2=1.0, c2_edge=None).to(device)
        else:
            hodge_meshft = HodgeTheory(V0, V1inv, c2=1.0, c2_edge=c2_edge).to(device)

    gamma_bias_cpu = torch.zeros(nx*ny, dtype=torch.float32)
    if args.bc_y == "open" and args.sponge_frac > 0.0 and args.sponge_gamma_max > 0.0:
        y = coords_cpu[:, 1]
        d = torch.minimum(y - y0, y1 - y)
        w = max(1e-12, args.sponge_frac * (y1 - y0))
        ramp = torch.clamp(1.0 - d / w, 0.0, 1.0)
        gamma_bias_cpu = args.sponge_gamma_max * (ramp**2)

    gamma_bias = gamma_bias_cpu.to(device)

    if args.sponge_mode == "shared":
        gamma_bias_for_meshft = None
    else:
        gamma_bias_for_meshft = gamma_bias

    # Dissipative operator
    if args.meshft_diss_mode == "none":
        diss_op = None
    elif args.meshft_diss_mode == "rayleigh_local":
        diss_op = RayleighDissLocalMLP(
            coords, hidden=args.diss_hidden, layers=2,
            min_gamma=max(0.0, args.rayleigh_gamma),
            gamma_bias=(gamma_bias_for_meshft if args.sponge_mode=="meshft_only" else None)
        ).to(device)

    meshft_net = MeshFTNet(src, dst, hodge_meshft, dissipative_op=diss_op).to(device)

    # optimizers
    def make_opt(params, lr=1e-3, wd=1e-6):
        ps = [p for p in params if p.requires_grad]
        return torch.optim.AdamW(ps, lr=lr, weight_decay=wd) if ps else None
    opt_meshft       = make_opt(meshft_net.parameters())

    # output directory
    stamp = args.run_name or f"{args.dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_s{args.seed}"
    run_dir = Path(args.save_dir) if args.save_dir else (Path(args.out_dir) / stamp)
    safe_makedirs(run_dir)
    with open(run_dir/"args.json","w") as f:
        json.dump(vars(args), f, indent=2)

    dt = float(dt_data)

    # Decide MeshFT-Net substeps from CFL (unless user overrides)
    meshft_nsub = args.meshft_roll_substeps if args.meshft_roll_substeps > 0 else _meshft_needed_substeps(dt, eattr, args.meshft_cfl)
    print(f"[MeshFT-Net] substeps per dt = {meshft_nsub} (CFL target={args.meshft_cfl})")

    # ------------------------- training loop -------------------------
    for ep in range(1, args.epochs+1):
        meshft_loss = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"[Epoch {ep}/{args.epochs}]")
        it_loader = iter(loader)
        for step_idx in pbar:
            try:
                x_t, x_tp1, aux = next(it_loader)
            except StopIteration:
                it_loader = iter(loader)
                x_t, x_tp1, aux = next(it_loader)

            x_t, x_tp1 = to_device(x_t, x_tp1, device=device)
            aux_d = {} if args.no_aux else {k: v.to(device) for k,v in aux.items()}

            # MeshFT-Net
            if opt_meshft is not None:
                opt_meshft.zero_grad(set_to_none=True)
                seg = (args.meshft_ckpt_seg
                    if (meshft_nsub >= args.meshft_ckpt_threshold and args.meshft_ckpt_seg > 0)
                    else 0)
                x_pred_meshft = _meshft_step_ckpt(meshft_net, x_t, dt, aux_d, meshft_nsub, seg_size=seg)
                if args.sponge_mode == "shared":
                    x_pred_meshft = apply_shared_sponge_step(x_pred_meshft, gamma_bias, dt)
                loss_meshft = vfield_mse(x_pred_meshft,  x_t, x_tp1, dt)
                if args.meshft_learn_hodge and hasattr(meshft_net.hodge, "reg_loss") and args.hodge_reg > 0.0:
                    loss_meshft = loss_meshft + args.hodge_reg * meshft_net.hodge.reg_loss()
                loss_meshft.backward()
                torch.nn.utils.clip_grad_norm_(meshft_net.parameters(), 1.0)
                opt_meshft.step()
                meshft_loss += loss_meshft.item()

            denom = step_idx + 1
            pbar.set_postfix(
                MeshFT_Net=(f"{meshft_loss/denom:.3e}" if opt_meshft is not None else "NA")
            )

        # ------------------------- validation / benchmarking -------------------------
        models_for_eval: Dict[str, Tuple[str, nn.Module]] = {
            "MeshFT-Net": ("meshft_net", meshft_net),
        }

        metrics = evaluate_models(
            models=models_for_eval,
            loader=val_loader,
            coords=coords, src=src, dst=dst, eattr=eattr,
            dt=dt, V0=V0, eval_hodge=eval_hodge,
            steps_limit=args.val_steps, rollout_T=args.roll_T,
            meshft_nsub=meshft_nsub, device=device,
            sponge_mode=args.sponge_mode,
            gamma_bias=(gamma_bias if args.sponge_mode=="shared" else None),
            meshft_ckpt_threshold=args.meshft_ckpt_threshold,
            meshft_ckpt_seg=args.meshft_ckpt_seg,
        )
        pretty_print_metrics(f"Validation (split='{val_used}')", metrics)

        # save metrics json
        (run_dir/ f"metrics_epoch_{ep:03d}.json").write_text(json.dumps(metrics, indent=2))

        # append CSV
        append_metrics_csv(args.bench_csv, ep, val_used, metrics)

        # save checkpoints
        if args.save_models and ((ep % args.save_every) == 0):
            save_checkpoint(run_dir/f"meshft_net_epoch_{ep:03d}.pth", meshft_net, opt_meshft, ep, args)

    # final evaluation & save
    models_for_eval = {"MeshFT-Net": ("meshft_net", meshft_net)}

    metrics = evaluate_models(
        models=models_for_eval,
        loader=val_loader,
        coords=coords, src=src, dst=dst, eattr=eattr,
        dt=dt, V0=V0, eval_hodge=eval_hodge,
        steps_limit=args.val_steps, rollout_T=args.roll_T,
        meshft_nsub=meshft_nsub, device=device,
        sponge_mode=args.sponge_mode,
        gamma_bias=(gamma_bias if args.sponge_mode=="shared" else None),
        meshft_ckpt_threshold=args.meshft_ckpt_threshold,
        meshft_ckpt_seg=args.meshft_ckpt_seg,
    )

    (run_dir/"metrics_final.json").write_text(json.dumps(metrics, indent=2))
    append_metrics_csv(args.bench_csv, args.epochs, val_used, metrics)
    pretty_print_metrics("Final Validation", metrics)

    if args.save_models:
        save_checkpoint(run_dir/"meshft_net_final.pth", meshft_net, opt_meshft, args.epochs, args)

    print(f"\nAll done. Artifacts are saved under: {run_dir.resolve()}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()