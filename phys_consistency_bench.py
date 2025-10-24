#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical-consistency benchmark for 2D wave dynamics models (MeshFT-Net / MGN / MGN-HP / HNN).

What this script adds (beyond accuracy/drift):
  - Dispersion relation check: estimate omega from predicted rollout and compare to ground-truth,
    and estimate wave speed c_hat = omega_hat / |k|.
  - Canonical consistency: check if p ≈ M * dq/dt (not explicitly supervised for black-box models).
  - PDE residual: relative norm of (M * q̈ + K q) using the COMMON theory Hodge (fair to all models).
  - Equipartition: time-averaged kinetic vs potential energy ratio ≈ 1 for plane waves.
  - Momentum conservation: time variation of sum(p) on the torus (should be ~constant).

Integrators (options):
  - MeshFT-Net: KDK (2nd, fixed)
  - MGN / MGN-HP: 'euler' (1st), 'rk2' (2nd), or 'kdk' (2nd, default) via --mgn_integrator / --mgnhp_integrator
  - HNN: 'se' (symplectic Euler, 1st) or 'kdk' (Störmer–Verlet, 2nd, default) via --hnn_integrator

Implementation notes:
  - We reuse the architecture and dataset forms from the prior analytic benchmark.
  - Long rollouts are executed on the chosen device, but the entire trajectory is offloaded to CPU
    to keep VRAM low. All physics metrics are computed on CPU using the theory Hodge buffers only.
  - The script writes a per-run CSV row (results.csv) and a JSON dump of detailed physics metrics.

Usage (example):
  python phys_consistency_bench.py --mesh grid --grid 32 32 --epochs 10 --train_size 4000 \
    --rollout_T 200 --dt 0.004 --lam_ham 0.05 --out_dir runs/phys_bench

"""

import os, math, argparse, json, csv, random, itertools, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# plotting (optional; only used if --make_plots is passed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- utils ----------------------------------

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

def _estimate_omega_phase(q_seq: torch.Tensor,
                          coords_cpu: torch.Tensor,
                          kvec_cpu: torch.Tensor,
                          dt: float) -> float:
    """
    Estimate angular frequency omega from a time series of the scalar field q(x,t)
    by projecting onto the known spatial Fourier mode exp(i k·x) and tracking
    phase increments over time.

    Args:
        q_seq:   [S, N] real tensor (CPU) of q at S time steps over N nodes.
        coords_cpu: [N, 2] CPU tensor of node coordinates.
        kvec_cpu:   [2] CPU tensor (or 1D array) of the target wavevector k.
        dt:      scalar time step.

    Returns:
        omega_hat (float): robust estimate in rad/s.

    Notes:
        - This avoids FFT frequency binning; resolution is set by dt only.
        - We weight phase differences by the instantaneous projection magnitude
          to reduce noise when the signal is weak.
    """
    q_seq = q_seq.to(dtype=torch.float64, device=coords_cpu.device)  # [S, N]
    kvec_cpu = kvec_cpu.to(dtype=torch.float64, device=coords_cpu.device)  # [2]
    xk = coords_cpu @ kvec_cpu                                  # [N]
    cos_xk = torch.cos(xk).unsqueeze(0)                         # [1,N]
    sin_xk = torch.sin(xk).unsqueeze(0)                         # [1,N]

    # Complex projection a_t = <q_t, e^{-i k·x}> = Σ_n q_t[n] (cos - i sin)
    a_real = (q_seq * cos_xk).sum(dim=1)                        # [S]
    a_imag = -(q_seq * sin_xk).sum(dim=1)                       # [S]
    phi = torch.atan2(a_imag, a_real)                           # [S] phase of a_t

    # Wrapped phase increments Δφ_t ∈ (-π, π]
    dphi = phi[1:] - phi[:-1]
    dphi = torch.remainder(dphi + math.pi, 2.0 * math.pi) - math.pi  # wrap to (-π, π]

    # Weights: use min(|a_t|, |a_{t+1}|) to discount low-SNR steps
    mag = torch.sqrt(a_real * a_real + a_imag * a_imag)         # [S]
    w = torch.minimum(mag[1:], mag[:-1])                        # [S-1]
    wsum = float(w.sum().item())

    if wsum <= 1e-12:
        # Fallback: unweighted mean of wrapped increments (should rarely happen)
        return float(dphi.mean().item() / dt)

    omega_hat = float((w * dphi).sum().item() / (dt * wsum))
    return abs(omega_hat)

# --- channel-wise std estimation & weighted loss (add) ---
@torch.no_grad()
def estimate_channel_std(loader, device):
    sq=0.0; sp=0.0; n=0
    for x0, x1, _ in loader:
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
                             eattr: torch.Tensor):        # [E,3] (dx,dy,|e|)
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

@torch.no_grad()
def estimate_omega_max_theory(V0, V1inv, src, dst, c2=1.0, iters=25):
    N = V0.numel()
    q = torch.randn(N, device=V0.device); q = q / (q.norm()+1e-12)
    lam = torch.tensor(0.0, device=V0.device)
    for _ in range(iters):
        Bq  = q[dst] - q[src]
        WBq = (c2 * V1inv) * Bq
        Kq  = torch.zeros_like(q); Kq.index_add_(0, dst, WBq); Kq.index_add_(0, src, -WBq)
        v   = Kq / (V0 + 1e-12)
        lam = (q * v).sum()
        q   = v / (v.norm()+1e-12)
    lam = lam.clamp_min(1e-12)
    return float(torch.sqrt(lam))

# ------------------------- discrete exterior calculus helpers -------------------------

def build_periodic_grid(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
    xs = torch.arange(nx, dtype=torch.float32) * (Lx / nx)
    ys = torch.arange(ny, dtype=torch.float32) * (Ly / ny)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    def nid(i,j): return i*ny + j
    src, dst, elen = [], [], []
    hx, hy = Lx/nx, Ly/ny
    for i in range(nx):
        for j in range(ny):
            a=nid(i,j); b=nid(i,(j+1)%ny)
            src.append(a); dst.append(b); elen.append(hy)
    for i in range(nx):
        for j in range(ny):
            a=nid(i,j); b=nid((i+1)%nx,j)
            src.append(a); dst.append(b); elen.append(hx)
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    elen= torch.tensor(elen,dtype=torch.float32)
    V0  = torch.full((nx*ny,), fill_value=hx*hy, dtype=torch.float32)
    return coords, src, dst, V0, elen

@torch.no_grad()
def B_times_q(src: torch.Tensor, dst: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return q[..., dst] - q[..., src]

def BT_times_e(src: torch.Tensor, dst: torch.Tensor, e: torch.Tensor, N: int) -> torch.Tensor:
    out = torch.zeros(*e.shape[:-1], N, dtype=e.dtype, device=e.device)
    out.index_add_(-1, dst, e); out.index_add_(-1, src, -e)
    return out

def build_edge_attr(coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    Lx = (coords[:,0].max()-coords[:,0].min()).item()
    Ly = (coords[:,1].max()-coords[:,1].min()).item()
    dv = coords[dst]-coords[src]
    if Lx>0 and Ly>0:
        dvx = dv[:,0] - torch.round(dv[:,0]/Lx)*Lx
        dvy = dv[:,1] - torch.round(dv[:,1]/Ly)*Ly
        dv  = torch.stack([dvx,dvy],dim=-1)
    elen = dv.norm(dim=-1, keepdim=True)
    return torch.cat([dv, elen], dim=-1)

# ------------------------- Hodges (theory + geom-MLP + learn) -------------------------

class HodgeTheory(nn.Module):
    """Fixed Hodge: M = V0, W = (c^2) * V1inv."""
    def __init__(self, V0: torch.Tensor, V1inv: torch.Tensor, c_speed: float = 1.0):
        super().__init__()
        self.register_buffer("V0", V0.clone())
        self.register_buffer("V1inv", V1inv.clone())
        self.c2 = float(c_speed)**2
    def M_vec(self): return self.V0
    def apply_W(self, e, src, dst, N): return (self.c2 * self.V1inv) * e

class HodgeGeomMLP(nn.Module):
    """Geometry-conditioned Hodge: predicts positive node-wise M scale and edge-wise W scale."""
    def __init__(self, coords, src, dst, V0, V1inv, hidden=64, layers=2, eps=1e-6):
        super().__init__()
        self.register_buffer("coords", coords.clone())
        self.register_buffer("src", src.clone()); self.register_buffer("dst", dst.clone())
        self.register_buffer("V0", V0.clone());  self.register_buffer("V1inv", V1inv.clone())
        self.eps = float(eps)
        # features
        with torch.no_grad():
            node = torch.cat([coords, V0.unsqueeze(-1)], dim=-1)
            dv = coords[dst]-coords[src]
            Lx = (coords[:,0].max()-coords[:,0].min()).item()
            Ly = (coords[:,1].max()-coords[:,1].min()).item()
            dvx = dv[:,0]-torch.round(dv[:,0]/max(Lx,1.0))*max(Lx,1.0)
            dvy = dv[:,1]-torch.round(dv[:,1]/max(Ly,1.0))*max(Ly,1.0)
            dv = torch.stack([dvx,dvy],dim=-1)
            elen = dv.norm(dim=-1, keepdim=True)
            edge = torch.cat([dv, elen], dim=-1)
            def _stdz(x):
                m=x.mean(0,keepdim=True); s=x.std(0,keepdim=True).clamp_min(1e-6)
                return (x-m)/s, m, s
            nf, nmu, nstd = _stdz(node); ef, emu, estd = _stdz(edge)
        self.register_buffer("node_raw", node); self.register_buffer("edge_raw", edge)
        self.register_buffer("node_mu", nmu);   self.register_buffer("node_std", nstd)
        self.register_buffer("edge_mu", emu);   self.register_buffer("edge_std", estd)
        def mlp(din, dout):
            dims=[din]+[hidden]*(layers-1)+[dout]; mods=[]
            for i in range(len(dims)-1):
                mods.append(nn.Linear(dims[i],dims[i+1]))
                if i<len(dims)-2: mods.append(nn.SiLU())
            return nn.Sequential(*mods)
        self.node_mlp = mlp(3,1); self.edge_mlp = mlp(3,1)

    def _stdz_node(self): return (self.node_raw - self.node_mu)/(self.node_std+1e-6)
    def _stdz_edge(self): return (self.edge_raw - self.edge_mu)/(self.edge_std+1e-6)

    def M_vec(self):
        m = F.softplus(self.node_mlp(self._stdz_node()).squeeze(-1))+self.eps
        return self.V0 * m
    def W_diag_vec(self):
        w = F.softplus(self.edge_mlp(self._stdz_edge()).squeeze(-1))+self.eps
        return self.V1inv * w
    def apply_W(self, e, src, dst, N):
        return self.W_diag_vec() * e

# ------------------------- MeshFT-Net DEC model -------------------------

class MeshFTNet(nn.Module):
    """Canonical DEC Hamiltonian model with KDK integrator."""
    def __init__(self, src, dst, hodge: nn.Module):
        super().__init__()
        self.src=src; self.dst=dst; self.hodge=hodge
        self.N = hodge.V0.numel(); self._nsub=1
        self.state_mode="canonical"
    def energy(self, x):
        q,p = x[...,0], x[...,1]
        M = self.hodge.M_vec()
        Bq = B_times_q(self.src, self.dst, q)
        WBq= self.hodge.apply_W(Bq, self.src, self.dst, self.N)
        pot = 0.5*(Bq*WBq).sum(dim=-1)
        kin = 0.5*((p**2)/(M+1e-12)).sum(dim=-1)
        return pot+kin
    def kdk_step(self, q,p,dt:float):
        M = self.hodge.M_vec(); Minv=1.0/(M+1e-12)
        Bq = B_times_q(self.src,self.dst,q); WBq=self.hodge.apply_W(Bq,self.src,self.dst,self.N)
        Kq = BT_times_e(self.src,self.dst,WBq,self.N)
        p_half = p - 0.5*dt*Kq
        q_new  = q + dt*(Minv*p_half)
        Bq_new = B_times_q(self.src,self.dst,q_new); WBq=self.hodge.apply_W(Bq_new,self.src,self.dst,self.N)
        Kq_new = BT_times_e(self.src,self.dst,WBq,self.N)
        p_new  = p_half - 0.5*dt*Kq_new
        return q_new, p_new
    def forward(self, x, dt:float):
        q,p = x[...,0], x[...,1]
        n=max(1,int(self._nsub)); dts=dt/n
        for _ in range(n):
            q,p = self.kdk_step(q,p,dts)
        return torch.stack([q,p],dim=-1)
    def vector_field(self, q: torch.Tensor, p: torch.Tensor):
        """Return the instantaneous Hamiltonian vector field (dq/dt, dp/dt)."""
        # M^{-1}
        M = self.hodge.M_vec()
        Minv = 1.0 / (M + 1e-12)

        # K q = B^T W B q
        Bq   = B_times_q(self.src, self.dst, q)
        WBq  = self.hodge.apply_W(Bq, self.src, self.dst, self.N)
        Kq   = BT_times_e(self.src, self.dst, WBq, self.N)

        # Canonical form: dq/dt = M^{-1} p, dp/dt = -K q
        dqdt = Minv * p
        dpdt = -Kq
        return dqdt, dpdt

# ------------------------- MGN / EnergyNet / HNN -------------------------

class MLP(nn.Module):
    def __init__(self, din, hidden, dout, layers=2):
        super().__init__()
        dims=[din]+[hidden]*(layers-1)+[dout]; mods=[]
        for i in range(len(dims)-1):
            mods.append(nn.Linear(dims[i],dims[i+1]))
            if i<len(dims)-2: mods.append(nn.SiLU())
        self.net=nn.Sequential(*mods)
    def forward(self,x): return self.net(x)

class GraphLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden):
        super().__init__()
        self.edge_mlp=MLP(2*node_dim+edge_dim, hidden, hidden, layers=2)
        self.node_mlp=MLP(node_dim+hidden, hidden, node_dim, layers=2)
    def forward(self,h,src,dst,eattr):
        hi=h[src]; hj=h[dst]
        m=self.edge_mlp(torch.cat([hi,hj,eattr],dim=-1))
        agg=torch.zeros_like(h); agg.index_add_(0,dst,m)
        return h + self.node_mlp(torch.cat([h,agg],dim=-1))

class MeshGraphNetVF(nn.Module):
    """v(x) predictor; integrates with explicit Euler."""
    def __init__(self, node_in=5, edge_in=3, hidden=64, layers=4, use_input_norm: bool = False):
        super().__init__()
        self.enc=MLP(node_in, hidden, hidden, layers=2)
        self.layers=nn.ModuleList([GraphLayer(hidden, edge_in, hidden) for _ in range(layers)])
        self.dec=MLP(hidden, hidden, 2, layers=2)

        self.use_input_norm = bool(use_input_norm)
        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))      # V0
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

    def forward(self, x_nodes, coords, src, dst, dt, eattr, node_extras=None):
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
            eattr_n  = (eattr  - self.mu_e ) / (self.std_e + 1e-6)
        else:
            coords_n, extras_n, eattr_n = coords, extras, eattr

        node_ctx = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([x_in, node_ctx], dim=-1)
        h=self.enc(h); B,N,H=h.shape

        offs=(torch.arange(B,device=h.device,dtype=src.dtype)*N).view(-1,1)
        src_b=(src.view(1,-1)+offs).reshape(-1); dst_b=(dst.view(1,-1)+offs).reshape(-1)
        eattr_b=eattr_n.unsqueeze(0).expand(B,-1,-1).reshape(-1,eattr_n.shape[-1])

        h_flat=h.reshape(B*N,-1)
        for gl in self.layers: h_flat=gl(h_flat,src_b,dst_b,eattr_b)
        v=self.dec(h_flat).view(B,N,2)
        return v

def mgn_step_kdk(model: Dict, x: torch.Tensor, coords: torch.Tensor,
                 src: torch.Tensor, dst: torch.Tensor, dt: float, eattr: torch.Tensor) -> torch.Tensor:
    """
    Component-split KDK for black-box v(x) = [v_q, v_p]:
      p_{n+1/2} = p_n + (dt/2) * v_p(q_n, p_n)
      q_{n+1}   = q_n +  dt    * v_q(q_n, p_{n+1/2})
      p_{n+1}   = p_{n+1/2} + (dt/2) * v_p(q_{n+1}, p_{n+1/2})
    """
    net = model["net"]
    extras = model.get("node_extras", None) 
    q, p = x[..., 0], x[..., 1]

    # Kick (half) at x_n
    v0    = net(x,                      coords, src, dst, dt, eattr, node_extras=extras)
    p_half = p + 0.5 * dt * v0[..., 1]
    x_mid = torch.stack([q, p_half], dim=-1)

    # Drift (full) using (q_n, p_{n+1/2})
    v_mid = net(x_mid,                  coords, src, dst, dt, eattr, node_extras=extras)
    q_new = q + dt * v_mid[..., 0]
    x_tmp = torch.stack([q_new, p_half], dim=-1)

    # Kick (half) at (q_{n+1}, p_{n+1/2})
    v_new = net(x_tmp,                  coords, src, dst, dt, eattr, node_extras=extras)
    p_new = p_half + 0.5 * dt * v_new[..., 1]

    return torch.stack([q_new, p_new], dim=-1)

class EnergyNet(nn.Module):
    """Node-wise energy aggregator -> scalar H(x)."""
    def __init__(self, node_in=5, edge_in=3, hidden=64, layers=4, use_input_norm: bool = False):
        super().__init__()
        self.enc=MLP(node_in, hidden, hidden, layers=2)
        self.layers=nn.ModuleList([GraphLayer(hidden, edge_in, hidden) for _ in range(layers)])
        self.dec_node=MLP(hidden, hidden, 1, layers=2)

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

    def forward(self, x_nodes, coords, src, dst, eattr, node_extras=None):
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
            eattr_n  = (eattr  - self.mu_e ) / (self.std_e + 1e-6)
        else:
            extras_n, eattr_n = extras, eattr

        node_ctx = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B,-1,-1)
        h = torch.cat([x_in, node_ctx], dim=-1)
        h=self.enc(h); B,N,H=h.shape

        offs=(torch.arange(B,device=h.device,dtype=src.dtype)*N).view(-1,1)
        src_b=(src.view(1,-1)+offs).reshape(-1); dst_b=(dst.view(1,-1)+offs).reshape(-1)
        eattr_b=eattr_n.unsqueeze(0).expand(B,-1,-1).reshape(-1,eattr_n.shape[-1])

        h_flat=h.reshape(B*N,-1)
        for gl in self.layers: h_flat=gl(h_flat,src_b,dst_b,eattr_b)
        e_node=self.dec_node(h_flat).view(B,N,1)
        return e_node.sum(dim=[1,2])

def apply_J_to_grad(g):
    gq, gp = g[...,0], g[...,1]
    return torch.stack([gp, -gq], dim=-1)

class _SeparableNodeEnergy(nn.Module):
    def __init__(self, hidden=64, layers=4, edge_in=3, use_input_norm: bool = False):
        super().__init__()
        self.enc=MLP(4, hidden, hidden, layers=2)
        self.layers=nn.ModuleList([GraphLayer(hidden, edge_in, hidden) for _ in range(layers)])
        self.dec_node=MLP(hidden, hidden, 1, layers=2)

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

    def forward(self, field_scalar, coords, src, dst, eattr, node_extras=None):
        B,N = field_scalar.shape

        s = field_scalar / (self.sigma_field + 1e-12) if self.use_input_norm else field_scalar

        extras = node_extras if node_extras is not None else coords.new_zeros(N,1)
        if self.use_input_norm:
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
            eattr_n  = (eattr  - self.mu_e ) / (self.std_e + 1e-6)
        else:
            coords_n, extras_n, eattr_n = coords, extras, eattr

        node_feat=torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B,-1,-1)
        h=torch.cat([s.unsqueeze(-1), node_feat], dim=-1)
        h=self.enc(h)

        offs=(torch.arange(B,device=dst.device,dtype=src.dtype)*N).view(-1,1)
        src_b=(src.view(1,-1)+offs).reshape(-1); dst_b=(dst.view(1,-1)+offs).reshape(-1)
        eattr_b=eattr_n.unsqueeze(0).expand(B,-1,-1).reshape(-1,eattr_n.shape[-1])

        h_flat=h.reshape(B*N,-1)
        for gl in self.layers: h_flat=gl(h_flat,src_b,dst_b,eattr_b)
        e_node=self.dec_node(h_flat).view(B,N,1)
        return e_node.sum(dim=[1,2])

class HNNSeparableSymplectic(nn.Module):
    """Separable HNN; symplectic Euler (kick–drift)."""
    def __init__(self, U_net, T_net, coords, src, dst, eattr=None, node_extras=None):
        super().__init__()
        self.U_net=U_net; self.T_net=T_net
        self.register_buffer("coords", coords.clone()); self.register_buffer("src", src.clone()); self.register_buffer("dst", dst.clone())
        self.eattr=eattr; self.node_extras=node_extras; self.state_mode="canonical"
        self.integrator = getattr(self, "integrator", "kdk")
    def energy(self, x):
        q,p=x[...,0], x[...,1]
        U=self.U_net(q, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        T=self.T_net(p, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        return U+T
    def _grad_U(self,q):
        q_req=q.detach().requires_grad_(True)
        U=self.U_net(q_req, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        (gq,)=torch.autograd.grad(U.sum(), q_req, create_graph=self.training)
        return gq
    def _grad_T(self,p):
        p_req=p.detach().requires_grad_(True)
        T=self.T_net(p_req, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        (gp,)=torch.autograd.grad(T.sum(), p_req, create_graph=self.training)
        return gp
    def _verlet_step(self, q: torch.Tensor, p: torch.Tensor, dt: float):
        # Kick (half)
        dUdq = self._grad_U(q)
        p_half = p - 0.5 * dt * dUdq
        # Drift (full)
        dTdp_half = self._grad_T(p_half)
        q_new = q + dt * dTdp_half
        # Kick (half)
        dUdq_new = self._grad_U(q_new)
        p_new = p_half - 0.5 * dt * dUdq_new
        return q_new, p_new
    def forward(self, x, dt: float):
        q, p = x[...,0], x[...,1]
        if self.integrator == "kdk":
            q, p = self._verlet_step(q, p, dt)
        else:  # 'se'
            dUdq = self._grad_U(q)
            p = p - dt * dUdq
            dTdp = self._grad_T(p)
            q = q + dt * dTdp
        out = torch.stack([q, p], dim=-1)
        if not self.training: out = out.detach()
        return out

# ------------------------- Analytic plane-wave dataset -------------------------

def sample_plane_params(Lx=1.0, Ly=1.0, c=1.0, kmax=4):
    kx = random.randint(1, kmax) * random.choice([-1,1])
    ky = random.randint(0, kmax) * random.choice([-1,1])
    if kx==0 and ky==0: ky=1
    kvec=np.array([2*np.pi*kx/Lx, 2*np.pi*ky/Ly], dtype=np.float32)
    omega = c * np.linalg.norm(kvec)
    phi = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.5, 1.5)
    return kvec.astype(np.float32), float(omega), float(phi), float(amp)

def plane_wave_q_and_v(coords: torch.Tensor, t: float, kvec, omega, phi, amp, device):
    x=coords.to(device)
    phase = x @ torch.tensor(kvec, device=device) - omega*t + phi
    q = amp * torch.sin(phase)
    v = - omega * amp * torch.cos(phase)
    return q, v

class PlaneWaveDataset(torch.utils.data.Dataset):
    """Pairs (x_t, x_{t+dt}) packed canonically: x=[q,p] with p = V0 * dq/dt."""
    def __init__(self, coords, V0, dt, size, c_wave=1.0, kmax=4, device="cpu"):
        super().__init__()
        self.coords=coords; self.V0=V0.to(device)
        self.dt=float(dt); self.size=int(size); self.c_wave=float(c_wave); self.kmax=int(kmax); self.device=device
        self.params=[sample_plane_params(Lx=float(coords[:,0].max()-coords[:,0].min()+1e-9),
                                         Ly=float(coords[:,1].max()-coords[:,1].min()+1e-9),
                                         c=self.c_wave, kmax=self.kmax) for _ in range(size)]
        self.t0=np.random.uniform(0, 2*np.pi, size).astype(np.float32)
    def __len__(self): return self.size
    def __getitem__(self, idx):
        kvec, omega, phi, amp = self.params[idx]
        t=float(self.t0[idx])
        q0,v0=plane_wave_q_and_v(self.coords, t, kvec, omega, phi, amp, self.device)
        q1,v1=plane_wave_q_and_v(self.coords, t+self.dt, kvec, omega, phi, amp, self.device)
        p0=self.V0*v0; p1=self.V0*v1
        x0=torch.stack([q0,p0],dim=-1); x1=torch.stack([q1,p1],dim=-1)
        meta=dict(kvec=torch.tensor(kvec), omega=omega, phi=phi, amp=amp, t=t)
        return x0, x1, meta

# ------------------------- training helpers -------------------------

def train_meshft(model: MeshFTNet, loader, dt, epochs:int, sigma_q: float, sigma_p: float):
    params=[p for p in model.parameters() if p.requires_grad]
    if not params: return
    opt=torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    for ep in range(epochs):
        for x0,x1,_ in loader:
            x0=x0.to(model.hodge.V0.device); x1=x1.to(model.hodge.V0.device)
            pred=model(x0, dt)
            # loss=weighted_mse(pred, x1, sigma_q, sigma_p)
            loss = F.mse_loss(pred, x1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

def train_mgn(model: Dict, loader, dt, epochs:int, sigma_q: float, sigma_p: float,
              lam_ham: float = 0.0, energy_net: EnergyNet = None):
    net=model["net"]; opt=model["_opt"]
    integrator = model.get("integrator", "euler")
    extras = model.get("node_extras", None)
    for ep in range(epochs):
        for x0,x1,_ in loader:
            x0=x0.to(net.enc.net[0].weight.device); x1=x1.to(net.enc.net[0].weight.device)

            if integrator == "kdk":
                x_pred = mgn_step_kdk(model, x0, model["coords"], model["src"], model["dst"], dt, model["eattr"])
                k1     = net(x0, model["coords"], model["src"], model["dst"], dt, model["eattr"], node_extras=extras)
            elif integrator == "rk2":
                k1   = net(x0, model["coords"], model["src"], model["dst"], dt, model["eattr"], node_extras=extras)
                x_mid = x0 + 0.5 * dt * k1
                k2   = net(x_mid, model["coords"], model["src"], model["dst"], dt, model["eattr"], node_extras=extras)
                x_pred = x0 + dt * k2
            else:  # euler
                v = net(x0, model["coords"], model["src"], model["dst"], dt, model["eattr"], node_extras=extras)
                x_pred = x0 + dt * v
                k1 = v

            # loss = weighted_mse(x_pred, x1, sigma_q, sigma_p)
            loss = F.mse_loss(x_pred, x1)
            if lam_ham > 0.0 and energy_net is not None:
                x0_req = x0.detach().requires_grad_(True)
                H = energy_net(x0_req, model["coords"], model["src"], model["dst"], model["eattr"],
                                node_extras=extras)
                (g,) = torch.autograd.grad(H.sum(), x0_req, create_graph=True)
                v_ham = apply_J_to_grad(g)
                loss_ham = F.mse_loss(k1, v_ham)
                loss = loss + lam_ham * loss_ham

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + (list(energy_net.parameters()) if energy_net is not None else []), 1.0)
            opt.step()

def train_hnn(model: HNNSeparableSymplectic, loader, dt, epochs:int, sigma_q: float, sigma_p: float):
    params=list(model.U_net.parameters())+list(model.T_net.parameters())
    opt=torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    for _ in range(epochs):
        for x0,x1,_ in loader:
            x0=x0.to(params[0].device); x1=x1.to(params[0].device)
            with torch.enable_grad():
                pred=model(x0, dt)
                # loss=weighted_mse(pred, x1, sigma_q, sigma_p)
                loss = F.mse_loss(pred, x1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

# ------------------------- rollout & physics metrics -------------------------

@torch.no_grad()
def collect_rollout(model_kind: str, model, coords, src, dst, eattr, V0, dt: float,
                    meta_batch: Dict[str,torch.Tensor], T: int, device: str):
    """
    Returns: traj_cpu [S,B,N,2] with S=T+1 on CPU.
    """
    B = len(meta_batch["omega"])
    # build initial batch x0 from meta
    x_list=[]
    for i in range(B):
        q0,v0=plane_wave_q_and_v(coords, float(meta_batch["t"][i]),
                                 meta_batch["kvec"][i].cpu().numpy(),
                                 float(meta_batch["omega"][i]),
                                 float(meta_batch["phi"][i]),
                                 float(meta_batch["amp"][i]), device)
        p0 = V0 * v0
        x_list.append(torch.stack([q0,p0],dim=-1))
    x = torch.stack(x_list, dim=0)  # [B,N,2] on device
    traj_cpu=[x.detach().cpu()]
    for k in range(T):
        if model_kind=="meshft_net":
            x = model(x, dt)
        elif model_kind=="hnn":
            with torch.enable_grad():
                x = model(x, dt)
            x = x.detach()
        else:
            integ = model.get("integrator", "euler")
            extras = model.get("node_extras", None)
            if integ == "kdk":
                x = mgn_step_kdk(model, x, coords, src, dst, dt, eattr)
            elif integ == "rk2":
                k1 = model["net"](x, coords, src, dst, dt, eattr, node_extras=extras)
                x  = x + dt * model["net"](x + 0.5 * dt * k1, coords, src, dst, dt, eattr,
                                           node_extras=extras)
            else:
                v  = model["net"](x, coords, src, dst, dt, eattr, node_extras=extras)
                x = x + dt * v

        traj_cpu.append(x.detach().cpu())
        if device.startswith("cuda") and (k%20==0):
            torch.cuda.empty_cache()
    traj_cpu = torch.stack(traj_cpu, dim=0)  # [S,B,N,2]
    return traj_cpu

def _energy_components_from_theory(q: torch.Tensor, s: torch.Tensor,
                                   V0: torch.Tensor, V1inv: torch.Tensor, c2: float,
                                   src: torch.Tensor, dst: torch.Tensor):
    # all CPU tensors
    B,N = q.shape
    Bq = q[:, dst] - q[:, src]    # [B,E]
    Wd = (c2 * V1inv).unsqueeze(0)  # [1,E]
    pot = 0.5 * (Bq * (Wd * Bq)).sum(dim=1)                # [B]
    kin = 0.5 * ((s**2) / (V0.unsqueeze(0)+1e-12)).sum(dim=1)  # [B]
    return kin, pot

def compute_phys_metrics(traj_cpu: torch.Tensor, meta_batch: Dict[str,torch.Tensor],
                         coords_cpu: torch.Tensor,
                         V0_cpu: torch.Tensor, V1inv_cpu: torch.Tensor, c_speed: float,
                         src_cpu: torch.Tensor, dst_cpu: torch.Tensor, dt: float) -> Dict[str,float]:
    """
    Compute dispersion / canonical consistency / PDE residual / equipartition / momentum conservation.
    All on CPU for robustness.
    """
    S,B,N,_ = traj_cpu.shape
    q = traj_cpu[...,0]  # [S,B,N]
    p = traj_cpu[...,1]  # [S,B,N]

    # --- 1) Dispersion via spatial projection + phase tracking (robust, high resolution) ---
    ome_true = torch.tensor([float(meta_batch["omega"][i]) for i in range(B)], dtype=torch.float64)
    k_norms  = torch.tensor([float(torch.linalg.norm(meta_batch["kvec"][i].detach().cpu())) for i in range(B)],
                             dtype=torch.float64)
    c_true   = torch.tensor([float(c_speed) for _ in range(B)], dtype=torch.float64)

    omega_hat_list = []
    for i in range(B):
        qi = q[:, i, :].to(dtype=torch.float64)                 # [S,N] (CPU)
        kvec_i = meta_batch["kvec"][i].detach().cpu().to(dtype=torch.float64)  # [2]
        omega_i = _estimate_omega_phase(qi, coords_cpu.to(dtype=torch.float64), kvec_i, dt)
        omega_hat_list.append(omega_i)
    omega_hat = torch.tensor(omega_hat_list, dtype=torch.float64)

    disp_rel_err = ((omega_hat - ome_true).abs() / (ome_true.abs() + 1e-12)).mean().item()
    c_hat = omega_hat / (k_norms + 1e-12)
    c_rel_err = ((c_hat - c_true).abs() / (c_true.abs()+1e-12)).mean().item()

    # --- 2) Canonical consistency: p ≈ M * dq/dt (central difference) ---
    dqdt = (q[2:,:,:] - q[:-2,:,:]) / (2.0*dt)   # [S-2,B,N]
    p_mid = 0.5 * (p[2:,:,:] + p[:-2,:,:])       # [S-2,B,N]
    target = (V0_cpu.unsqueeze(0).unsqueeze(0)) * dqdt
    num = ((p_mid - target)**2).sum()
    den = (target**2).sum().clamp_min(1e-12)
    canon_rel_mse = float((num/den).item())

    # --- 3) PDE residual: M q̈ + K q ≈ 0 with common theory Hodge ---
    qtt = (q[2:,:,:] - 2.0*q[1:-1,:,:] + q[:-2,:,:]) / (dt*dt)    # [S-2,B,N]
    def K_apply(q_now):  # q_now: [B,N]
        Bq = q_now[:, dst_cpu] - q_now[:, src_cpu]                 # [B,E]
        Wd = (c_speed**2) * V1inv_cpu.unsqueeze(0)                 # [1,E]
        WBq = Wd * Bq
        out = torch.zeros_like(q_now)
        out.index_add_(1, dst_cpu, WBq)
        out.index_add_(1, src_cpu, -WBq)
        return out
    res_num = 0.0; res_den = 0.0
    for t in range(S-2):
        q_t = q[1+t, :, :]
        Kq = K_apply(q_t)
        Mqtt = (V0_cpu.unsqueeze(0)) * qtt[t, :, :]
        r = Mqtt + Kq
        res_num += (r**2).sum().item()
        res_den += ((Mqtt**2).sum().item() + (Kq**2).sum().item()) + 1e-12
    pde_res_rel = float(res_num / res_den)

    # --- 4) Equipartition: <Kin>/<Pot> ≈ 1 ---
    kin_list=[]; pot_list=[]
    for s in range(S):
        kin, pot = _energy_components_from_theory(q[s,:,:], p[s,:,:], V0_cpu, V1inv_cpu, c_speed**2, src_cpu, dst_cpu)
        kin_list.append(kin); pot_list.append(pot)
    Kin = torch.stack(kin_list, dim=0)  # [S,B]
    Pot = torch.stack(pot_list, dim=0)  # [S,B]
    Kin_mu = Kin.mean(dim=0); Pot_mu = Pot.mean(dim=0)
    equip_err = ( (Kin_mu - Pot_mu).abs() / (Kin_mu + Pot_mu + 1e-12) ).mean().item()

    # --- 5) Momentum conservation: time variation of sum(p) ---
    sump = p.sum(dim=2)      # [S,B]
    denom = (p.abs().sum(dim=2).mean(dim=0) + 1e-12)  # [B]
    mom_var = ((sump.max(dim=0).values - sump.min(dim=0).values).abs() / denom).mean().item()

    return dict(
        disp_omega_rel_err=disp_rel_err,
        wave_c_rel_err=c_rel_err,
        canonical_rel_mse=canon_rel_mse,
        pde_res_rel=pde_res_rel,
        equipartition_rel_err=equip_err,
        momentum_conserv_var=mom_var
    )

# --- Learning adequacy diagnostics -------------------------------------------------
@torch.no_grad()
def _true_vector_field(eval_hodge, src, dst, x, state_mode="canonical"):
    """Compute ground-truth PDE vector field v_true(x) using the theory Hodge."""
    q, s = x[..., 0], x[..., 1]
    M = eval_hodge.M_vec()
    Minv = 1.0 / (M + 1e-12)
    Bq = B_times_q(src, dst, q)
    W_Bq = eval_hodge.apply_W(Bq, src, dst, q.shape[-1])
    Kq = BT_times_e(src, dst, W_Bq, q.shape[-1])
    if state_mode == "canonical":
        dqdt = Minv * s
        dsdt = -Kq
    else:
        dqdt = s
        dsdt = -(Minv * Kq)
    return torch.stack([dqdt, dsdt], dim=-1)

def _model_vector_field(model_kind, model, x, coords, src, dst, dt, eattr=None):
    """Get model's instantaneous vector field v_model(x) for all four model types."""
    if model_kind in ("mgn", "mgnhp"):
        return model["net"](x, coords, src, dst, dt, eattr, node_extras=model.get("node_extras", None))
    elif model_kind == "meshft_net":
        q, s = x[..., 0], x[..., 1]
        dqdt, dsdt = model.vector_field(q, s)
        return torch.stack([dqdt, dsdt], dim=-1)
    elif model_kind == "hnn":
        # Re-enable autograd inside no_grad context
        with torch.enable_grad():
            q, p = x[..., 0], x[..., 1]
            dUdq = model._grad_U(q)
            dTdp = model._grad_T(p)
            dqdt = dTdp
            dpdt = -dUdq
            v = torch.stack([dqdt, dpdt], dim=-1)
        return v.detach()
    else:
        raise ValueError("unknown model_kind")

def _cosine_and_rel_l2(v_pred, v_true):
    """Return (cosine_similarity_mean, relative_L2)."""
    vp = v_pred.reshape(v_pred.shape[0], -1)
    vt = v_true.reshape(v_true.shape[0], -1)
    num = (vp * vt).sum(dim=-1)
    den = (vp.norm(dim=-1) * vt.norm(dim=-1) + 1e-12)
    cos = (num / den).mean().item()
    rel = (vp - vt).norm(dim=-1) / (vt.norm(dim=-1) + 1e-12)
    return cos, rel.mean().item()

def _fit_amp_phase(q_field, coords, kvec):
    """
    Fit q(x) ≈ a*sin(k·x) + b*cos(k·x) by 2x2 least squares.
    Return (amplitude, phase) where a = A*cos(phi), b = A*sin(phi).
    """
    xk = coords @ torch.tensor(kvec, dtype=coords.dtype, device=coords.device)
    S = torch.sin(xk); C = torch.cos(xk)
    SS = (S*S).sum(); CC = (C*C).sum(); SC = (S*C).sum()
    SQ = (S*q_field).sum(); CQ = (C*q_field).sum()
    det = (SS*CC - SC*SC).clamp_min(1e-12)
    a = ( CC*SQ - SC*CQ) / det
    b = (-SC*SQ + SS*CQ) / det
    A = torch.sqrt(a*a + b*b)
    phi = torch.atan2(b, a)
    return float(A.item()), float(phi.item())

@torch.no_grad()
def compute_learning_diagnostics(model_kind, model, coords, src, dst, eval_hodge,
                                 meta_batch, dt, T_short=16, eattr=None, state_mode="canonical"):
    """
    Returns a dict with learning adequacy metrics:
      - vf_cosine, vf_rel_l2
      - short_roll_rel_mse@T_short, short_err_growth_rate
      - amp_rel_err, phase_abs_err_deg
    """
    device = coords.device
    B = len(meta_batch["t"])
    # Build initial batch x0 from meta (same as rollout code path)
    x0_list = []
    for i in range(B):
        q0, v0 = plane_wave_q_and_v(coords, float(meta_batch["t"][i]),
                                    meta_batch["kvec"][i].cpu().numpy(),
                                    float(meta_batch["omega"][i]),
                                    float(meta_batch["phi"][i]),
                                    float(meta_batch["amp"][i]), device)
        s0 = eval_hodge.V0 * v0 if state_mode == "canonical" else v0
        x0_list.append(torch.stack([q0, s0], dim=-1))
    x = torch.stack(x0_list, dim=0)

    # --- Vector-field alignment at x0 ---
    v_true = _true_vector_field(eval_hodge, src, dst, x, state_mode=state_mode)
    v_pred = _model_vector_field(model_kind, model, x, coords, src, dst, dt, eattr=eattr)
    vf_cos, vf_rel = _cosine_and_rel_l2(v_pred, v_true)

    # --- Short rollout error + early growth rate ---
    rel_hist = []
    z = x.clone()
    for k in range(T_short):
        # advance model
        if model_kind in ("mgn", "mgnhp"):
            integ = model.get("integrator", "euler")
            extras = model.get("node_extras", None)
            if integ == "kdk":
                z = mgn_step_kdk(model, z, coords, src, dst, dt, eattr)
            elif integ == "rk2":
                k1 = model["net"](z, coords, src, dst, dt, eattr, node_extras=extras)
                z  = z + dt * model["net"](z + 0.5*dt*k1, coords, src, dst, dt, eattr,
                                        node_extras=extras)
            else:
                v  = model["net"](z, coords, src, dst, dt, eattr, node_extras=extras)
                z = z + dt * v
        elif model_kind == "meshft_net":
            z = model(z, dt)
        else:  # hnn
            with torch.enable_grad():
                z = model(z, dt)
            z = z.detach()
        # GT now
        t_now = meta_batch["t"].to(device) + (k+1) * dt
        gt_list = []
        for i in range(B):
            qg, vg = plane_wave_q_and_v(coords, float(t_now[i]),
                                        meta_batch["kvec"][i].cpu().numpy(),
                                        float(meta_batch["omega"][i]),
                                        float(meta_batch["phi"][i]),
                                        float(meta_batch["amp"][i]), device)
            sg = eval_hodge.V0 * vg if state_mode == "canonical" else vg
            gt_list.append(torch.stack([qg, sg], dim=-1))
        gt = torch.stack(gt_list, dim=0)
        rel = (z - gt).reshape(B, -1).norm(dim=-1) / (gt.reshape(B, -1).norm(dim=-1) + 1e-12)
        rel_hist.append(rel)
    rel_hist = torch.stack(rel_hist, dim=0)                  # [T_short,B]
    short_rel = rel_hist[-1].mean().item()
    # simple log-slope over first half steps
    K = max(4, T_short // 2)
    y = torch.log(rel_hist[:K].mean(dim=1).clamp_min(1e-12)) # [K]
    x_idx = torch.arange(K, dtype=y.dtype, device=y.device)
    slope = float(((x_idx - x_idx.mean()) * (y - y.mean())).sum() /
                  ((x_idx - x_idx.mean()).pow(2).sum() + 1e-12))

    # --- Plane-wave amplitude/phase on the MODEL output at t_pred = t0 + T_short*dt ---
    # Fit on predicted q (z[...,0]) and compare to true amplitude A and
    # effective phase phi_eff = phi - omega * t_pred.
    t_pred = meta_batch["t"].to(device) + T_short * dt
    amp_errs, ph_errs = [], []
    for i in range(B):
        kvec_i = meta_batch["kvec"][i].cpu().numpy()
        A_hat, phi_hat = _fit_amp_phase(z[i, :, 0], coords, kvec_i)
        A_true = float(meta_batch["amp"][i])
        phi_eff = float(meta_batch["phi"][i] - meta_batch["omega"][i] * float(t_pred[i]))
        # wrap phase diff to [-pi, pi]
        dphi = ((phi_hat - phi_eff + math.pi) % (2*math.pi)) - math.pi
        amp_errs.append(abs(A_hat - A_true) / (abs(A_true) + 1e-12))
        ph_errs.append(abs(dphi) * 180.0 / math.pi)
    amp_rel_err = float(np.mean(amp_errs))
    phase_abs_err_deg = float(np.mean(ph_errs))

    return {
        "vf_cosine": vf_cos,
        "vf_rel_l2": vf_rel,
        "short_roll_rel_mse@T{}".format(T_short): short_rel,
        "short_err_growth_rate": slope,        # per step (log-error slope)
        "amp_rel_err": amp_rel_err,
        "phase_abs_err_deg": phase_abs_err_deg
    }

# ------------------------- main run (train -> metrics) -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs/phys_bench")
    ap.add_argument("--out_csv", type=str, default="runs/phys_bench/results.csv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)

    # mesh
    ap.add_argument("--mesh", type=str, default="grid", choices=["grid"])
    ap.add_argument("--grid", type=int, nargs=2, default=[32,32])
    ap.add_argument("--Lx", type=float, default=1.0); ap.add_argument("--Ly", type=float, default=1.0)

    # dynamics / data
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--c_speed", type=float, default=1.0, help="theory Hodge wave speed for evaluation")
    ap.add_argument("--c_wave", type=float, default=None, help="analytic plane-wave speed; default uses c_speed")
    ap.add_argument("--meshft_sigma_cfl", type=float, default=0.9,
                help="Substep guard: nsub = ceil(omega_max * dt / meshft_sigma_cfl)")

    ap.add_argument("--std_inputs", type=int, default=1,
                    help="standardize geometry inputs: coords/V0/edge_attr (0/1)")
    ap.add_argument("--std_state", type=int, default=1,
                    help="standardize state channels q,p using dataset std (0/1)")

    # train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--train_size", type=int, default=4000)
    ap.add_argument("--val_size", type=int, default=256)

    # models
    ap.add_argument("--meshft_hodge_mode", type=str, default="learn_geom", choices=["theory","learn_geom"])
    ap.add_argument("--mgn_hidden", type=int, default=64)
    ap.add_argument("--mgn_layers", type=int, default=4)
    ap.add_argument("--lam_ham", type=float, default=0.01)
    ap.add_argument("--hnn_hidden", type=int, default=64)
    ap.add_argument("--hnn_layers", type=int, default=4)

    # rollout
    ap.add_argument("--rollout_T", type=int, default=200)

    # --- Integrators (default = KDK for all) ---
    ap.add_argument("--mgn_integrator", type=str, default="kdk",
                    choices=["euler", "rk2", "kdk"],
                    help="Integrator for MGN")
    ap.add_argument("--mgnhp_integrator", type=str, default="kdk",
                    choices=["euler", "rk2", "kdk"],
                    help="Integrator for MGN-HP")
    ap.add_argument("--hnn_integrator", type=str, default="kdk",
                    choices=["se", "kdk"],  # 'kdk' == Störmer–Verlet
                    help="Integrator for HNN")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    if args.c_wave is None: args.c_wave = float(args.c_speed)
    seeds = args.seeds if (args.seeds and len(args.seeds)>0) else [args.seed]
    set_seed(seeds[0])
    device = torch.device(args.device)

    # mesh
    nx,ny = args.grid; Lx,Ly=args.Lx,args.Ly
    coords, src, dst, V0, elen = build_periodic_grid(nx, ny, Lx, Ly)
    V1inv = torch.ones_like(elen)  # grid: simple identity on edges
    eattr = build_edge_attr(coords, src, dst)

    coords_d, src_d, dst_d, V0_d, V1inv_d, eattr_d = to_device(coords, src, dst, V0, V1inv, eattr, device=device)

    # dataset
    train_ds = PlaneWaveDataset(coords_d, V0_d, dt=args.dt, size=args.train_size, c_wave=args.c_wave, kmax=args.kmax, device=device.type)
    val_ds   = PlaneWaveDataset(coords_d, V0_d, dt=args.dt, size=args.val_size,   c_wave=args.c_wave, kmax=args.kmax, device=device.type)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    sigma_q, sigma_p = estimate_channel_std(train_loader, device)

    mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e = compute_input_norm_stats(
        coords_d, V0_d.unsqueeze(-1), eattr_d
    )
    sigma_q_used = sigma_q if args.std_state else 1.0
    sigma_p_used = sigma_p if args.std_state else 1.0
    use_norm = bool(args.std_inputs or args.std_state)
    
    # evaluation Hodge for energy & PDE residual
    eval_hodge = HodgeTheory(V0_d, V1inv_d, c_speed=args.c_speed).to(device)

    # --- build models ---
    # MeshFT-Net
    if args.meshft_hodge_mode=="theory":
        hodge_tr = HodgeTheory(V0_d, V1inv_d, c_speed=args.c_speed).to(device)
    else:
        hodge_tr = HodgeGeomMLP(coords_d, src_d, dst_d, V0_d, V1inv_d, hidden=64, layers=2).to(device)
    meshft_net = MeshFTNet(src_d, dst_d, hodge_tr).to(device)

    # MGN & MGN-HP
    mgn_net   = MeshGraphNetVF(node_in=5, edge_in=eattr.shape[1], hidden=args.mgn_hidden,
                            layers=args.mgn_layers, use_input_norm=use_norm).to(device)
    mgn_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                            sigma_q=sigma_q_used, sigma_p=sigma_p_used)
    mgn = {"net": mgn_net,
        "_opt": torch.optim.AdamW(mgn_net.parameters(), lr=1e-3, weight_decay=1e-6),
        "coords": coords_d, "src": src_d, "dst": dst_d, "eattr": eattr_d,
        "integrator": args.mgn_integrator,
        "node_extras": V0_d.unsqueeze(-1)}

    mgnhp_net  = MeshGraphNetVF(node_in=5, edge_in=eattr.shape[1], hidden=args.mgn_hidden,
                                layers=args.mgn_layers, use_input_norm=use_norm).to(device)
    mgnhp_enet = EnergyNet(node_in=5, edge_in=eattr.shape[1], hidden=args.mgn_hidden,
                        layers=args.mgn_layers, use_input_norm=use_norm).to(device)

    mgnhp_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                sigma_q=sigma_q_used, sigma_p=sigma_p_used)
    mgnhp_enet.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                sigma_q=sigma_q_used, sigma_p=sigma_p_used)
    mgnhp = {"net": mgnhp_net, "energy_net": mgnhp_enet,
             "_opt": torch.optim.AdamW([
                 {"params": mgnhp_net.parameters(),  "lr": 1e-3},
                 {"params": mgnhp_enet.parameters(), "lr": 3e-4},
             ], weight_decay=1e-6),
             "coords": coords_d, "src": src_d, "dst": dst_d, "eattr": eattr_d,
             "integrator": args.mgnhp_integrator,
             "node_extras": V0_d.unsqueeze(-1)}   

    # HNN
    U_net = _SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                edge_in=eattr.shape[1], use_input_norm=use_norm).to(device)
    T_net = _SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                edge_in=eattr.shape[1], use_input_norm=use_norm).to(device)
    U_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e, sigma_field=(sigma_q if args.std_state else 1.0))
    T_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e, sigma_field=(sigma_p if args.std_state else 1.0))
    hnn = HNNSeparableSymplectic(U_net, T_net, coords_d, src_d, dst_d, eattr=eattr_d, node_extras=V0_d.unsqueeze(-1)).to(device)
    hnn.integrator = args.hnn_integrator

    # ===== Shared CFL gate from THEORY Hodge =====
    omega = estimate_omega_max_theory(eval_hodge.V0, eval_hodge.V1inv, src_d, dst_d, c2=args.c_speed**2)
    meshft_net._nsub = int(math.ceil(max(1.0, (omega * args.dt) / max(1e-12, args.meshft_sigma_cfl))))
    print(f"[MeshFT-Net:CFL] omega≈{omega:.3e}, dt={args.dt:.3e} -> nsub={meshft_net._nsub} (sigma={args.meshft_sigma_cfl})")

    # --- train ---
    print("[train] MeshFT-Net ...");   train_meshft(meshft_net,   train_loader, args.dt, args.epochs, sigma_q, sigma_p)
    print("[train] MGN ...");   train_mgn(mgn,   train_loader, args.dt, args.epochs, sigma_q, sigma_p, lam_ham=0.0)
    print("[train] MGN-HP ...");train_mgn(mgnhp, train_loader, args.dt, args.epochs, sigma_q, sigma_p, lam_ham=args.lam_ham, energy_net=mgnhp_enet)
    print("[train] HNN ...");   train_hnn(hnn,   train_loader, args.dt, args.epochs, sigma_q, sigma_p)

    # --- take a small meta batch for rollouts/metrics ---
    B = min(8, len(val_ds))
    meta_batch = {
        "t": torch.tensor([val_ds[i][2]["t"] for i in range(B)], device=device),
        "kvec": torch.stack([val_ds[i][2]["kvec"].to(device) for i in range(B)], dim=0),
        "omega": [val_ds[i][2]["omega"] for i in range(B)],
        "phi":   [val_ds[i][2]["phi"]   for i in range(B)],
        "amp":   [val_ds[i][2]["amp"]   for i in range(B)],
    }

    # --- collect trajectories (GPU -> CPU) ---
    traj_meshft   = collect_rollout("meshft_net",   meshft_net,   coords_d, src_d, dst_d, eattr_d, V0_d, args.dt, meta_batch, args.rollout_T, device.type)
    traj_mgn   = collect_rollout("mgn",   mgn,   coords_d, src_d, dst_d, eattr_d, V0_d, args.dt, meta_batch, args.rollout_T, device.type)
    traj_mgnhp = collect_rollout("mgnhp", mgnhp, coords_d, src_d, dst_d, eattr_d, V0_d, args.dt, meta_batch, args.rollout_T, device.type)
    traj_hnn   = collect_rollout("hnn",   hnn,   coords_d, src_d, dst_d, eattr_d, V0_d, args.dt, meta_batch, args.rollout_T, device.type)

    # --- compute physics metrics on CPU using the THEORY Hodge buffers ---
    V0_cpu    = eval_hodge.V0.detach().cpu()
    V1inv_cpu = eval_hodge.V1inv.detach().cpu()
    src_cpu   = src.detach().cpu()
    dst_cpu   = dst.detach().cpu()
    coords_cpu = coords.detach().cpu()

    phys_meshft   = compute_phys_metrics(traj_meshft,   meta_batch, coords_cpu, V0_cpu, V1inv_cpu, args.c_speed, src_cpu, dst_cpu, args.dt)
    phys_mgn   = compute_phys_metrics(traj_mgn,   meta_batch, coords_cpu, V0_cpu, V1inv_cpu, args.c_speed, src_cpu, dst_cpu, args.dt)
    phys_mgnhp = compute_phys_metrics(traj_mgnhp, meta_batch, coords_cpu, V0_cpu, V1inv_cpu, args.c_speed, src_cpu, dst_cpu, args.dt)
    phys_hnn   = compute_phys_metrics(traj_hnn,   meta_batch, coords_cpu, V0_cpu, V1inv_cpu, args.c_speed, src_cpu, dst_cpu, args.dt)

    # --- learning diagnostics (vector-field alignment, short-rollout, amp/phase) ---
    learn_meshft   = compute_learning_diagnostics("meshft_net",   meshft_net,   coords_d, src_d, dst_d, eval_hodge, meta_batch, args.dt, T_short=16, eattr=eattr_d)
    learn_mgn   = compute_learning_diagnostics("mgn",   mgn,   coords_d, src_d, dst_d, eval_hodge, meta_batch, args.dt, T_short=16, eattr=eattr_d)
    learn_mgnhp = compute_learning_diagnostics("mgnhp", mgnhp, coords_d, src_d, dst_d, eval_hodge, meta_batch, args.dt, T_short=16, eattr=eattr_d)
    learn_hnn   = compute_learning_diagnostics("hnn",   hnn,   coords_d, src_d, dst_d, eval_hodge, meta_batch, args.dt, T_short=16, eattr=eattr_d)

    # --- one-step validation numbers for context ---
    val_loader_it = iter(val_loader)
    x0b, x1b, _ = next(val_loader_it)
    x0b=x0b.to(device); x1b=x1b.to(device)
    with torch.no_grad():
        e1_meshft = F.mse_loss(meshft_net(x0b, args.dt), x1b).item()

        # MGN
        if args.mgn_integrator == "kdk":
            pred_mgn = mgn_step_kdk(mgn, x0b, coords_d, src_d, dst_d, args.dt, eattr_d)
        elif args.mgn_integrator == "rk2":
            k1 = mgn_net(x0b, coords_d, src_d, dst_d, args.dt, eattr_d, node_extras=mgn["node_extras"])
            pred_mgn = x0b + args.dt * mgn_net(x0b + 0.5 * args.dt * k1, coords_d, src_d, dst_d, args.dt,
                                       eattr_d, node_extras=mgn["node_extras"])
        else:
            v_mgn = mgn_net(x0b, coords_d, src_d, dst_d, args.dt, eattr_d, node_extras=mgn["node_extras"])
            pred_mgn = x0b + args.dt * v_mgn
        
        e1_mgn = F.mse_loss(pred_mgn, x1b).item()

        if args.mgnhp_integrator == "kdk":
            pred_hp = mgn_step_kdk(mgnhp, x0b, coords_d, src_d, dst_d, args.dt, eattr_d)
        elif args.mgnhp_integrator == "rk2":
            k1 = mgnhp_net(x0b, coords_d, src_d, dst_d, args.dt, eattr_d,
                       node_extras=mgnhp["node_extras"])
            pred_hp = x0b + args.dt * mgnhp_net(x0b + 0.5*args.dt*k1, coords_d, src_d, dst_d,
                                                args.dt, eattr_d, node_extras=mgnhp["node_extras"])
        else:
            v_hp  = mgnhp_net(x0b, coords_d, src_d, dst_d, args.dt, eattr_d,
                          node_extras=mgnhp["node_extras"])  
            pred_hp = x0b + args.dt * v_hp
        
        e1_hp = F.mse_loss(pred_hp, x1b).item()

        with torch.enable_grad():
            hnn.integrator = args.hnn_integrator
            pred_hnn = hnn(x0b, args.dt)
        e1_hnn = F.mse_loss(pred_hnn, x1b).item()

    # --- write summary CSV row ---
    row = dict(
        mesh=args.mesh, grid=f"{nx}x{ny}", dt=args.dt, kmax=args.kmax, c_speed=args.c_speed, c_wave=args.c_wave,
        epochs=args.epochs, train_size=args.train_size, rollout_T=args.rollout_T,
        std_inputs=int(args.std_inputs),
        std_state=int(args.std_state),
        e1_meshft=e1_meshft, e1_mgn=e1_mgn, e1_mgnhp=e1_hp, e1_hnn=e1_hnn,

        # learning diagnostics (MeshFT-Net)
        meshft_vf_cosine=learn_meshft["vf_cosine"],
        meshft_vf_rel_l2=learn_meshft["vf_rel_l2"],
        meshft_short_roll_rel_mse=learn_meshft["short_roll_rel_mse@T16"],
        meshft_short_err_growth=learn_meshft["short_err_growth_rate"],
        meshft_amp_rel_err=learn_meshft["amp_rel_err"],
        meshft_phase_abs_err_deg=learn_meshft["phase_abs_err_deg"],

        # learning diagnostics (MGN)
        mgn_vf_cosine=learn_mgn["vf_cosine"],
        mgn_vf_rel_l2=learn_mgn["vf_rel_l2"],
        mgn_short_roll_rel_mse=learn_mgn["short_roll_rel_mse@T16"],
        mgn_short_err_growth=learn_mgn["short_err_growth_rate"],
        mgn_amp_rel_err=learn_mgn["amp_rel_err"],
        mgn_phase_abs_err_deg=learn_mgn["phase_abs_err_deg"],

        # learning diagnostics (MGN-HP)
        mgnhp_vf_cosine=learn_mgnhp["vf_cosine"],
        mgnhp_vf_rel_l2=learn_mgnhp["vf_rel_l2"],
        mgnhp_short_roll_rel_mse=learn_mgnhp["short_roll_rel_mse@T16"],
        mgnhp_short_err_growth=learn_mgnhp["short_err_growth_rate"],
        mgnhp_amp_rel_err=learn_mgnhp["amp_rel_err"],
        mgnhp_phase_abs_err_deg=learn_mgnhp["phase_abs_err_deg"],

        # learning diagnostics (HNN)
        hnn_vf_cosine=learn_hnn["vf_cosine"],
        hnn_vf_rel_l2=learn_hnn["vf_rel_l2"],
        hnn_short_roll_rel_mse=learn_hnn["short_roll_rel_mse@T16"],
        hnn_short_err_growth=learn_hnn["short_err_growth_rate"],
        hnn_amp_rel_err=learn_hnn["amp_rel_err"],
        hnn_phase_abs_err_deg=learn_hnn["phase_abs_err_deg"],

        # physics metrics (means over B)
        meshft_disp_omega_rel_err=phys_meshft["disp_omega_rel_err"],
        mgn_disp_omega_rel_err=phys_mgn["disp_omega_rel_err"],
        mgnhp_disp_omega_rel_err=phys_mgnhp["disp_omega_rel_err"],
        hnn_disp_omega_rel_err=phys_hnn["disp_omega_rel_err"],

        meshft_wave_c_rel_err=phys_meshft["wave_c_rel_err"],
        mgn_wave_c_rel_err=phys_mgn["wave_c_rel_err"],
        mgnhp_wave_c_rel_err=phys_mgnhp["wave_c_rel_err"],
        hnn_wave_c_rel_err=phys_hnn["wave_c_rel_err"],

        meshft_canonical_rel_mse=phys_meshft["canonical_rel_mse"],
        mgn_canonical_rel_mse=phys_mgn["canonical_rel_mse"],
        mgnhp_canonical_rel_mse=phys_mgnhp["canonical_rel_mse"],
        hnn_canonical_rel_mse=phys_hnn["canonical_rel_mse"],

        meshft_pde_res_rel=phys_meshft["pde_res_rel"],
        mgn_pde_res_rel=phys_mgn["pde_res_rel"],
        mgnhp_pde_res_rel=phys_mgnhp["pde_res_rel"],
        hnn_pde_res_rel=phys_hnn["pde_res_rel"],

        meshft_equipartition_rel_err=phys_meshft["equipartition_rel_err"],
        mgn_equipartition_rel_err=phys_mgn["equipartition_rel_err"],
        mgnhp_equipartition_rel_err=phys_mgnhp["equipartition_rel_err"],
        hnn_equipartition_rel_err=phys_hnn["equipartition_rel_err"],

        meshft_momentum_conserv_var=phys_meshft["momentum_conserv_var"],
        mgn_momentum_conserv_var=phys_mgn["momentum_conserv_var"],
        mgnhp_momentum_conserv_var=phys_mgnhp["momentum_conserv_var"],
        hnn_momentum_conserv_var=phys_hnn["momentum_conserv_var"],
    )

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    is_new = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if is_new: w.writerow(list(row.keys()))
        w.writerow([row[k] for k in row.keys()])

    # also dump a JSON with per-model details
    details = {
        "MeshFT-Net": {**phys_meshft, **learn_meshft},
        "MGN":        {**phys_mgn,    **learn_mgn},
        "MGN-HP":     {**phys_mgnhp,  **learn_mgnhp},
        "HNN":        {**phys_hnn,    **learn_hnn},
        "meta": {"B": len(meta_batch["omega"]), "T": args.rollout_T, "dt": args.dt},
    }
    with open(os.path.join(args.out_dir, "phys_consistency_summary.json"), "w") as f:
        json.dump(details, f, indent=2)

    print("\n=== Physical consistency (mean over batch) ===")
    for name, d in [("MeshFT-Net", phys_meshft), ("MGN", phys_mgn), ("MGN-HP", phys_mgnhp), ("HNN", phys_hnn)]:
        print(f"{name:6s} | disp ω err={d['disp_omega_rel_err']:.3e}  c err={d['wave_c_rel_err']:.3e}  "
              f"canon={d['canonical_rel_mse']:.3e}  PDE={d['pde_res_rel']:.3e}  "
              f"equip={d['equipartition_rel_err']:.3e}  momVar={d['momentum_conserv_var']:.3e}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()