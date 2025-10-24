#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytic plane-wave benchmark on general 2D meshes (grid or Delaunay) with FIXED missing-data masks:
- Dataset uses CANONICAL momentum by default: x = [q, p] with p = M * dq/dt.
- Models: MGN, MGN+Hamiltonian-penalty (MGN-HP), MeshFT-Net (DEC-based).
- Integrators:
-   MeshFT-Net: KDK (2nd, fixed)
-   MGN/MGN-HP: Euler (1st), RK2-midpoint (2nd), or KDK (2nd, default) via --mgn_integrator / --mgnhp_integrator
-   HNN: symplectic Euler (1st) or Störmer–Verlet/KDK (2nd, default) via --hnn_integrator
- Mesh: regular periodic grid OR random-point Delaunay triangulation.
- W Hodge: diagonal, or SPD via node-coupled off-diagonals (learnable case).
- Hodge mode: 'theory' (fixed), 'learn' (free trainable), or 'learn_geom' (geometry-conditioned).
- Data-size / missing-rate sweeps with publication-quality plots.
- Fixed missing-data masks per run and per split (train/val), shared across models.
- Robustness when normalization is OFF:
    * clamp tiny dual areas / ultra-short edges on Delaunay
    * optional global speed^2 calibration to match target wave speed
    * KDK stability guard using an estimate of the maximum angular frequency
    * rollout NaN/Inf guards so CSV does not contain NaNs

Notes:
- Canonical dataset: M_data is the node dual area V0 (barycentric for Delaunay; cell area for grids).
  Thus p = V0 * dq/dt at each node.
- 'Theory' Hodge: M = V0, W = (c_speed^2) * V1inv (no normalization), canonical dynamics:
    dq/dt = M^{-1} p,   dp/dt = -K q,   K = B^T W B
- 'Learn' Hodge: M and W are learned SPD (optionally with node-coupled off-diagonals for W).
- Missing-data masks:
  * Built ONCE per split with a fixed seed, then reused across all epochs/batches/models.
  * Evaluation metrics can optionally apply the same mask (see --mask_apply_to_val).

Fairness & evaluation:
- For energy-based diagnostics (e.g., drift) and rollout relative errors, we evaluate with a COMMON
  physical energy induced by the THEORY Hodge (M=V0, W=c^2 V1inv) across all models (MeshFT-Net/MGN/HP).
  This provides an apples-to-apples comparison independent of what the model learned internally.
"""

import argparse, math, os, csv, random, itertools, json
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# optional triangulators
_HAVE_SCIPY = False
try:
    from scipy.spatial import Delaunay as _SciPyDelaunay
    _HAVE_SCIPY = True
except Exception:
    pass
_HAVE_MPL_TRI = False
try:
    import matplotlib.tri as _mtri
    _HAVE_MPL_TRI = True
except Exception:
    pass

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

def psnr(pred: torch.Tensor, target: torch.Tensor, peak: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-20: return 99.0
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)

def to_device(*tensors, device="cpu"):
    return [t.to(device) for t in tensors]

def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Mask shape: [B,N,C] or None."""
    if mask is None:
        return F.mse_loss(pred, tgt)
    diff2 = (pred - tgt)**2
    num = (diff2 * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den

def _cot_at(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float = 1e-12) -> float:
    u = b - a
    v = c - a
    cross = u[0]*v[1] - u[1]*v[0]
    dot   = (u * v).sum()
    return float(dot) / (abs(float(cross)) + eps)

@torch.no_grad()
def cotangent_W_from_tris(coords: torch.Tensor,
                          src: torch.Tensor, dst: torch.Tensor,
                          simplices: np.ndarray) -> torch.Tensor:
    """
    Cotangent weights on a torus using minimum-image distances.
    For each triangle (i,j,k) with base indices, compute side lengths under
    periodic minimum-image convention, then cotangents via:
        cot(α_i) = (|ki|^2 + |ij|^2 - |jk|^2) / (4 * Area)
    Edge weight: W_e = 0.5 * (cot α + cot β) for the two incident angles.
    """
    device = coords.device
    E = src.numel()
    W = torch.zeros(E, dtype=torch.float32, device=device)

    # Edge index map (undirected, src<dst as built)
    ed2idx = {(int(a) if int(a) < int(b) else int(b),
               int(b) if int(a) < int(b) else int(a)): k
              for k, (a, b) in enumerate(zip(src.tolist(), dst.tolist()))}

    # Domain sizes (infer from coords bbox)
    Lx = (coords[:, 0].max() - coords[:, 0].min()).item()
    Ly = (coords[:, 1].max() - coords[:, 1].min()).item()
    Lx = float(Lx) if Lx > 0 else 1.0
    Ly = float(Ly) if Ly > 0 else 1.0

    def _min_image_len(i: int, j: int) -> torch.Tensor:
        pi = coords[i]; pj = coords[j]
        dx = pj[0] - pi[0]; dy = pj[1] - pi[1]
        dx = dx - torch.round(dx / Lx) * Lx
        dy = dy - torch.round(dy / Ly) * Ly
        return torch.sqrt(dx*dx + dy*dy + 1e-30)

    for tri in simplices:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        # side lengths (minimum-image)
        lij = _min_image_len(i, j)
        ljk = _min_image_len(j, k)
        lki = _min_image_len(k, i)

        # Heron's formula for area (robust on obtuse triangles)
        s = 0.5 * (lij + ljk + lki)
        A2 = torch.clamp(s*(s-lij)*(s-ljk)*(s-lki), min=1e-24)
        A = torch.sqrt(A2)

        # Cotangents at i, j, k
        # Opposite sides: at i -> a=|jk|, adjacent: b=|ki|, c=|ij|
        cot_i = (lki*lki + lij*lij - ljk*ljk) / (4.0 * A + 1e-30)
        cot_j = (lij*lij + ljk*ljk - lki*lki) / (4.0 * A + 1e-30)
        cot_k = (ljk*ljk + lki*lki - lij*lij) / (4.0 * A + 1e-30)

        # Accumulate 0.5*cot(angle-opposite-edge) to edge weights
        e_jk = (min(j, k), max(j, k)); W[ed2idx[e_jk]] += 0.5 * cot_i
        e_ik = (min(i, k), max(i, k)); W[ed2idx[e_ik]] += 0.5 * cot_j
        e_ij = (min(i, j), max(i, j)); W[ed2idx[e_ij]] += 0.5 * cot_k

    # Enforce non-negativity up to tiny numerical noise
    W = torch.clamp(W, min=0.0)
    return W

@torch.no_grad()
def _prepare_hodge_for_dt(hodge, src, dst, dt, target_c2=1.0, guard=True):
    if hasattr(hodge, "calibrate_speed2"):
        hodge.calibrate_speed2(target_c2)
    if guard and getattr(hodge, "log_speed2", None) is not None:
        try:
            omega = _estimate_omega_max(src, dst, hodge, iters=15)
            if omega * dt > 1.0:
                with torch.no_grad():
                    scale = (1.0 / (omega * dt))**2
                    hodge.log_speed2.add_(math.log(scale))
        except Exception:
            pass

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_energy_series_csv(E_hist: torch.Tensor, dt: float, out_dir: str, fname: str):
    """
    Save energy time series to CSV.
    E_hist: [S, B] tensor where S = (#steps done + 1), B = batch size.
    CSV columns: step, time, E_mean, E_median, E_std, E_0, E_1, ..., E_{B-1}
    """
    _ensure_dir(out_dir)
    S, B = int(E_hist.shape[0]), int(E_hist.shape[1])
    path = os.path.join(out_dir, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["step", "time", "E_mean", "E_median", "E_std"] + [f"E_{i}" for i in range(B)]
        w.writerow(header)
        for s in range(S):
            Ei = E_hist[s].detach().cpu().numpy().astype(np.float64)
            row = [s, s * float(dt), float(np.nanmean(Ei)), float(np.nanmedian(Ei)), float(np.nanstd(Ei))]
            row += [float(x) for x in Ei.tolist()]
            w.writerow(row)

# ---- channel-wise std estimation & weighted losses ----

@torch.no_grad()
def estimate_channel_std(loader, device):
    sq = 0.0; sp = 0.0; n = 0
    for _, x1, _ in loader:
        x1 = x1.to(device)
        sq += (x1[..., 0]**2).sum().item()
        sp += (x1[..., 1]**2).sum().item()
        n  += x1.shape[0] * x1.shape[1]
    sigma_q = (sq / max(1, n))**0.5 + 1e-8
    sigma_p = (sp / max(1, n))**0.5 + 1e-8
    return float(sigma_q), float(sigma_p)

@torch.no_grad()
def compute_input_norm_stats(coords: torch.Tensor,
                             node_extras: torch.Tensor,   # [N,1] (V0)
                             eattr: torch.Tensor):        # [E,3] (dx,dy,|e|)
    def _stats(x: torch.Tensor):
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        return mu, std
    mu_xy,  std_xy  = _stats(coords)       # [1,2]
    mu_ex,  std_ex  = _stats(node_extras)  # [1,1]
    mu_e,   std_e   = _stats(eattr)        # [1,3] (edge_in_dim)
    return mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e

def masked_weighted_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor,
                        sigma_q: float, sigma_p: float) -> torch.Tensor:
    """Channel-wise weighted MSE; mask can be None."""
    dq2 = ((pred[..., 0] - tgt[..., 0]) / sigma_q) ** 2
    dp2 = ((pred[..., 1] - tgt[..., 1]) / sigma_p) ** 2
    if mask is None:
        return (dq2 + dp2).mean()
    num = dq2 * mask[..., 0] + dp2 * mask[..., 1]
    den = (mask[..., 0] + mask[..., 1]).sum().clamp_min(1.0)
    return num.sum() / den

@torch.no_grad()
def energy_power_violation_pi(eval_hodge: nn.Module, src, dst, x: torch.Tensor,
                              model_kind: str, model, coords, dt: float,
                              eattr=None, state_mode: str = "canonical") -> float:
    """
    Π = <∇H_theory(x), v_model(x)>
    """
    q, s = x[..., 0], x[..., 1]               # [B,N]
    M = eval_hodge.M_vec()                    # [N]
    Minv = 1.0 / (M + 1e-12)

    # grad H_theory = [Kq, Minv*s]
    Bq  = B_times_q(src, dst, q)
    Wd  = (eval_hodge.c2 * eval_hodge.V1inv).unsqueeze(0)   # [1,E]
    WBq = Wd * Bq
    Kq  = torch.zeros_like(q)
    Kq.index_add_(-1, dst, WBq); Kq.index_add_(-1, src, -WBq)
    grad_q, grad_s = Kq, Minv.unsqueeze(0) * s              # [B,N]

    # v_model(x)
    if model_kind in ("mgn", "mgnhp"):
        v = model["net"](x, coords, src, dst, dt, eattr,
                         node_extras=model.get("node_extras", None))
    elif model_kind == "meshft_net":
        dqdt, dsdt = model.vector_field(q, s); v = torch.stack([dqdt, dsdt], dim=-1)
    else:  # hnn
        with torch.enable_grad():
            dUdq = model._grad_U_wrt_q(q); dTdp = model._grad_T_wrt_p(s)
            v = torch.stack([dTdp, -dUdq], dim=-1)
        v = v.detach()

    vq, vs = v[..., 0], v[..., 1]
    pi = (grad_q * vq + grad_s * vs).sum(dim=1)   # [B]
    return float(pi.mean().item())

@torch.no_grad()
def drift_slope_from_series(E_hist: torch.Tensor, dt: float) -> float:
    device = E_hist.device
    dtype = E_hist.dtype
    Em = E_hist.to(dtype=dtype).mean(dim=1)

    t = torch.arange(E_hist.shape[0], dtype=dtype, device=device)
    dt_t = torch.as_tensor(dt, dtype=dtype, device=device)
    t = t * dt_t

    t0 = t - t.mean()
    e0 = Em - Em.mean()
    eps = torch.tensor(1e-12, dtype=dtype, device=device)
    slope = (t0.mul(e0).sum() / (t0.square().sum() + eps)).item()
    return float(slope)

# ------------------------- mesh & DEC -------------------------

def build_periodic_grid(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
    """Regular periodic grid."""
    xs = torch.arange(nx, dtype=torch.float32) * (Lx / nx)
    ys = torch.arange(ny, dtype=torch.float32) * (Ly / ny)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)  # [N,2]
    def node_id(i, j): return i * ny + j

    src, dst, elen = [], [], []
    hx, hy = Lx / nx, Ly / ny
    # horizontal edges
    for i in range(nx):
        for j in range(ny):
            a = node_id(i, j); b = node_id(i, (j + 1) % ny)
            src.append(a); dst.append(b); elen.append(hy)
    # vertical edges
    for i in range(nx):
        for j in range(ny):
            a = node_id(i, j); b = node_id((i + 1) % nx, j)
            src.append(a); dst.append(b); elen.append(hx)

    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)
    elen = torch.tensor(elen, dtype=torch.float32)
    N = nx * ny
    node_area = torch.full((N,), fill_value=(hx * hy), dtype=torch.float32)
    return coords, src, dst, node_area, elen

def _tri_area(p, q, r):
    return 0.5 * abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

def build_delaunay_mesh(n_points: int, Lx: float = 1.0, Ly: float = 1.0, seed: int = 0):
    """
    Periodic Delaunay on a torus [0,Lx) x [0,Ly) via 3x3 tiling:
      1) Sample N base points in the central tile.
      2) Tile them on a 3x3 grid of shifts {-1,0,1}*{Lx,Ly}.
      3) Run Delaunay on all 9N points.
      4) KEEP only triangles whose centroid lies inside the central tile.
      5) Map triangle vertices back to base indices (periodic identification).
    Returns:
      coords[N,2] in [0,Lx) x [0,Ly),
      src[E], dst[E]  (undirected unique edges, src<dst),
      V0[N] (barycentric dual areas on the torus),
      elen[E] (edge length under minimum-image convention),
      simplices[T,3] (triangles as base node indices; orientation arbitrary).
    """
    rng = np.random.RandomState(seed)
    base = np.stack([rng.rand(n_points)*Lx, rng.rand(n_points)*Ly], axis=1)  # [N,2]

    # 3x3 tiling
    tiles = []
    orig  = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            shift = np.array([ix*Lx, iy*Ly], dtype=np.float64)
            tiles.append(base + shift[None, :])
            orig.append(np.arange(n_points, dtype=np.int64))
    P_all = np.concatenate(tiles, axis=0)     # [9N,2]
    O_all = np.concatenate(orig,  axis=0)     # [9N]

    # Delaunay on the tiled cloud
    if _HAVE_SCIPY:
        tri = _SciPyDelaunay(P_all)
        T_all = tri.simplices
    elif _HAVE_MPL_TRI:
        tri = _mtri.Triangulation(P_all[:,0], P_all[:,1])
        T_all = tri.triangles
    else:
        raise ImportError("Delaunay requires SciPy or Matplotlib (tri).")

    # Keep triangles whose centroid is inside the central tile (not modded)
    keep = []
    for a, b, c in T_all:
        cent = (P_all[a] + P_all[b] + P_all[c]) / 3.0
        if (0.0 <= cent[0] < Lx) and (0.0 <= cent[1] < Ly):
            keep.append((int(a), int(b), int(c)))

    # Map kept triangles to base indices and deduplicate by vertex set
    tri_keys = set()
    tris_base = []
    for a, b, c in keep:
        ia, ib, ic = int(O_all[a]), int(O_all[b]), int(O_all[c])
        key = tuple(sorted((ia, ib, ic)))
        if (key[0] != key[1]) and (key[1] != key[2]) and (key[0] != key[2]):
            if key not in tri_keys:
                tri_keys.add(key)
                tris_base.append([ia, ib, ic])
    if len(tris_base) == 0:
        raise RuntimeError("No triangles kept for periodic Delaunay.")

    # Build unique undirected edges from triangles (base indices)
    edset = set()
    for i, j, k in tris_base:
        for u, v in ((i, j), (j, k), (k, i)):
            if u == v: 
                continue
            if u > v: 
                u, v = v, u
            edset.add((u, v))
    ed_sorted = sorted(list(edset))
    src, dst = zip(*ed_sorted)
    src = torch.tensor(src, dtype=torch.long)
    dst = torch.tensor(dst, dtype=torch.long)

    # Torch coordinates in the base cell
    coords = torch.tensor(base, dtype=torch.float32)
    N = coords.shape[0]

    # Minimum-image delta helper (vectorized for a single pair)
    def _min_image_delta(pj: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        dx = pj[0] - pi[0]; dy = pj[1] - pi[1]
        dx = dx - torch.round(dx / Lx) * Lx
        dy = dy - torch.round(dy / Ly) * Ly
        return torch.stack([dx, dy], dim=-1)

    # Edge lengths under minimum-image convention
    elen = torch.empty(len(ed_sorted), dtype=torch.float32)
    for e, (i, j) in enumerate(ed_sorted):
        dv = _min_image_delta(coords[j], coords[i])
        elen[e] = torch.linalg.norm(dv)

    # Node dual areas V0 via Heron's formula on periodic triangles
    V0 = torch.zeros(N, dtype=torch.float32)
    for i, j, k in tris_base:
        # side lengths (minimum-image)
        lij = torch.linalg.norm(_min_image_delta(coords[j], coords[i]))
        ljk = torch.linalg.norm(_min_image_delta(coords[k], coords[j]))
        lki = torch.linalg.norm(_min_image_delta(coords[i], coords[k]))
        s = 0.5 * (lij + ljk + lki)
        A = torch.sqrt(torch.clamp(s*(s-lij)*(s-ljk)*(s-lki), min=1e-20))
        share = (A / 3.0).to(V0.dtype)
        V0[i] += share; V0[j] += share; V0[k] += share

    # Gentle robustness floors (scale-free)
    v0_floor = torch.quantile(V0, 0.001).clamp_min(1e-12)
    V0 = V0.clamp_min(v0_floor)
    e_floor = torch.quantile(elen, 0.001).clamp_min(1e-9)
    elen = elen.clamp_min(e_floor)

    simplices = np.asarray(tris_base, dtype=np.int64)
    return coords, src, dst, V0, elen, simplices

@torch.no_grad()
def B_times_q(src: torch.Tensor, dst: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Apply 1-form incidence B to node scalar q: (B q)_e = q[dst] - q[src]."""
    return q[..., dst] - q[..., src]  # supports [B,N] or [N]

def BT_times_e(src: torch.Tensor, dst: torch.Tensor, e: torch.Tensor, N: int) -> torch.Tensor:
    """Apply B^T to edge scalar e. Accumulates contributions to nodes."""
    out = torch.zeros(*e.shape[:-1], N, dtype=e.dtype, device=e.device)
    out.index_add_(-1, dst, e)   # +e to dst
    out.index_add_(-1, src, -e)  # -e to src
    return out

@torch.no_grad()
def observed_order_dt(model_kind: str, model, coords, src, dst, eval_hodge,
                      meta_batch, dt: float, eattr=None, alpha_shared: float = 1.0,
                      state_mode: str = "canonical") -> float:

    device = coords.device
    B = len(meta_batch["t"])
    x = []
    for i in range(B):
        q0, v0 = plane_wave_q_and_v(coords, float(meta_batch["t"][i]),
                                    meta_batch["kvec"][i].cpu().numpy(),
                                    float(meta_batch["omega"][i]),
                                    float(meta_batch["phi"][i]),
                                    float(meta_batch["amp"][i]), device)
        s0 = (eval_hodge.V0 * v0) if state_mode == "canonical" else v0
        x.append(torch.stack([q0, s0], dim=-1))
    x0 = torch.stack(x, dim=0)

    def step_once(z, h):
        if model_kind == "meshft_net":
            return model(z, h)
        elif model_kind == "hnn":
            with torch.enable_grad():
                out = model(z, alpha_shared * h)
            return out.detach()
        else:
            if model.get("integrator","euler") == "kdk":
                return mgn_step_kdk(model, z, coords, src, dst, h, alpha_shared)
            elif model.get("integrator","euler") == "rk2":
                k1 = model["net"](z, coords, src, dst, h, eattr,
                                  node_extras=model.get("node_extras", None))
                z_mid = z + 0.5 * alpha_shared * h * k1
                k2 = model["net"](z_mid, coords, src, dst, h, eattr,
                                  node_extras=model.get("node_extras", None))
                return z + alpha_shared * h * k2
            else:
                v = model["net"](z, coords, src, dst, h, eattr,
                                 node_extras=model.get("node_extras", None))
                return z + alpha_shared * h * v

    # dt
    x_dt = step_once(x0, dt)
    # dt/2 
    x_h  = step_once(x0, 0.5*dt)
    x_dt2 = step_once(x_h, 0.5*dt)
    t1 = meta_batch["t"].to(device) + alpha_shared * dt
    gt1 = []
    for i in range(B):
        qg, vg = plane_wave_q_and_v(coords, float(t1[i]),
                                    meta_batch["kvec"][i].cpu().numpy(),
                                    float(meta_batch["omega"][i]),
                                    float(meta_batch["phi"][i]),
                                    float(meta_batch["amp"][i]), device)
        sg = (eval_hodge.V0 * vg) if state_mode == "canonical" else vg
        gt1.append(torch.stack([qg, sg], dim=-1))
    gt1 = torch.stack(gt1, dim=0)

    err_dt  = phys_rel_error_from_hodge(eval_hodge, src, dst, x_dt,  gt1, coords.shape[0], state_mode).mean().item() + 1e-16
    err_dt2 = phys_rel_error_from_hodge(eval_hodge, src, dst, x_dt2, gt1, coords.shape[0], state_mode).mean().item() + 1e-16

    p = math.log(err_dt/err_dt2) / math.log(2.0)
    return float(p)

# ------------------------- Hodge blocks -------------------------

class HodgeBlockLearnable(nn.Module):
    """
    Learn 0-form and 1-form Hodge stars (SPD).
    W can be:
      - diagonal:        W = diag(w)
      - node-coupled:    W = diag(w) + C^T diag(gamma) C,  C_{n,e}=1 if edge e touches node n
    Includes geometry injection; normalization is optional (OFF by default).
    """
    def __init__(self,
                 V0: torch.Tensor,
                 V1inv: torch.Tensor,
                 eps: float = 1e-6,
                 use_speed_scalar: bool = False,
                 w_structure: str = "diag",
                 offdiag_init: float = -6.0,
                 normalize: bool = False):
        super().__init__()
        assert w_structure in ("diag", "offdiag")
        self.register_buffer("V0", V0.clone())
        self.register_buffer("V1inv", V1inv.clone())
        self.eps = eps
        self.use_speed_scalar = use_speed_scalar
        self.w_structure = w_structure
        self.normalize = normalize

        self.log_m = nn.Parameter(torch.zeros_like(V0))      # [N]
        self.log_w = nn.Parameter(torch.zeros_like(V1inv))   # [E]
        if self.use_speed_scalar:
            self.log_speed2 = nn.Parameter(torch.zeros(()))  # scalar (global c^2)
        else:
            self.register_parameter("log_speed2", None)

        if self.w_structure == "offdiag":
            init = torch.full_like(V0, float(offdiag_init))
            self.log_gamma = nn.Parameter(init)              # [N]
        else:
            self.register_parameter("log_gamma", None)

    def M_vec(self) -> torch.Tensor:
        M = F.softplus(self.log_m) + self.eps
        M = self.V0 * M
        if self.normalize:
            M = M / (M.mean() + 1e-12)
        return M

    def W_diag_vec(self) -> torch.Tensor:
        Wd = F.softplus(self.log_w) + self.eps
        Wd = self.V1inv * Wd
        if self.use_speed_scalar and self.log_speed2 is not None:
            Wd = Wd * torch.exp(self.log_speed2)
        if self.normalize:
            Wd = Wd / (Wd.mean() + 1e-12)
        return Wd

    def apply_W(self, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, Nnodes: int) -> torch.Tensor:
        Wd = self.W_diag_vec()
        y = Wd * e
        if self.w_structure == "offdiag":
            gamma = F.softplus(self.log_gamma) + self.eps  # [N] >= eps
            s = torch.zeros(*e.shape[:-1], Nnodes, dtype=e.dtype, device=e.device)
            s.index_add_(-1, src, e)
            s.index_add_(-1, dst, e)
            t = gamma * s
            y = y + t[..., src] + t[..., dst]
        return y

    @torch.no_grad()
    def calibrate_speed2(self, target_c2: float = 1.0):
        """
        Set global speed^2 so that mean(W)/mean(M) ~= target_c2.
        No-op if normalization is ON or speed scalar is disabled.
        """
        if not self.use_speed_scalar or self.log_speed2 is None or self.normalize:
            return
        Wd = (F.softplus(self.log_w) + self.eps) * self.V1inv
        M  = (F.softplus(self.log_m) + self.eps) * self.V0
        c2_est = (Wd.mean() / (M.mean() + 1e-12)).clamp_min(1e-12).item()
        self.log_speed2.data = torch.tensor(math.log(max(1e-12, target_c2 / c2_est)),
                                            dtype=self.log_speed2.dtype,
                                            device=self.log_speed2.device)

class HodgeBlockTheory(nn.Module):
    """Fixed (non-trainable) Hodge: M = V0, W = (c_speed^2) * V1inv * exp(log_speed2) if enabled."""
    def __init__(self, V0: torch.Tensor, V1inv: torch.Tensor,
                 c_speed: float = 1.0, use_speed_scalar: bool = True):
        super().__init__()
        self.register_buffer("V0", V0.clone())
        self.register_buffer("V1inv", V1inv.clone())
        self.c2 = float(c_speed) ** 2
        self.use_speed_scalar = bool(use_speed_scalar)
        if self.use_speed_scalar:
            # kept as a parameter in case speed-guard wants to adjust it
            self.log_speed2 = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_speed2", None)

    def M_vec(self) -> torch.Tensor:
        return self.V0

    def apply_W(self, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, Nnodes: int) -> torch.Tensor:
        Wd = self.c2 * self.V1inv
        if getattr(self, "log_speed2", None) is not None:
            Wd = Wd * torch.exp(self.log_speed2)
        return Wd * e

    @torch.no_grad()
    def calibrate_speed2(self, target_c2: float = 1.0):
        if getattr(self, "log_speed2", None) is None:
            return
        M  = self.V0
        Wg = self.c2 * self.V1inv
        c2_est = (Wg.mean() / (M.mean() + 1e-12)).clamp_min(1e-12).item()
        self.log_speed2.data = torch.tensor(math.log(max(1e-12, target_c2 / c2_est)),
                                            dtype=M.dtype, device=M.device)

class HodgeBlockGeomMLP(nn.Module):
    """
    Geometry-conditioned Hodge:
      - Predicts M (node-wise) and W_diag (edge-wise) from geometry features only.
      - Node features: (x, y, V0)
      - Edge features: (dx, dy, |e|) under periodic minimum-image convention
      - Optional off-diagonal coupling: W = diag(Wd) + C^T diag(gamma) C
    This keeps the Hodge identifiable from mesh geometry even when data is missing.
    """
    def __init__(self,
                 coords: torch.Tensor,            # [N,2]
                 src: torch.Tensor, dst: torch.Tensor,
                 V0: torch.Tensor,                # [N]
                 V1inv: torch.Tensor,             # [E]
                 eps: float = 1e-6,
                 use_speed_scalar: bool = False,
                 w_structure: str = "diag",       # "diag" or "offdiag"
                 normalize: bool = False,
                 hidden: int = 64,
                 layers: int = 2,
                 use_sn: bool = False):
        super().__init__()
        assert w_structure in ("diag", "offdiag")
        self.eps = float(eps)
        self.w_structure = w_structure
        self.normalize = bool(normalize)
        self.use_speed_scalar = bool(use_speed_scalar)

        # Geometry buffers (constant w.r.t. learning)
        self.register_buffer("coords", coords.clone())
        self.register_buffer("src", src.clone())
        self.register_buffer("dst", dst.clone())
        self.register_buffer("V0", V0.clone())
        self.register_buffer("V1inv", V1inv.clone())

        # Domain size (for minimum-image edge features)
        Lx = (coords[:, 0].max() - coords[:, 0].min()).item()
        Ly = (coords[:, 1].max() - coords[:, 1].min()).item()
        self.register_buffer("Lx", torch.tensor(max(Lx, 1.0), dtype=coords.dtype))
        self.register_buffer("Ly", torch.tensor(max(Ly, 1.0), dtype=coords.dtype))

        # Feature construction (node: [x,y,V0], edge: [dx,dy,|e|])
        with torch.no_grad():
            node_feats = torch.cat([self.coords, self.V0.unsqueeze(-1)], dim=-1)  # [N,3]
            dv = self.coords[self.dst] - self.coords[self.src]
            dvx = dv[:, 0] - torch.round(dv[:, 0] / self.Lx) * self.Lx
            dvy = dv[:, 1] - torch.round(dv[:, 1] / self.Ly) * self.Ly
            dv = torch.stack([dvx, dvy], dim=-1)
            elen = dv.norm(dim=-1, keepdim=True)
            edge_feats = torch.cat([dv, elen], dim=-1)                            # [E,3]

            # Light standardization to stabilize training
            def _stdz(x):
                m = x.mean(dim=0, keepdim=True)
                s = x.std(dim=0, keepdim=True).clamp_min(1e-6)
                return (x - m) / s, m, s
            nf, nmu, nstd = _stdz(node_feats)
            ef, emu, estd = _stdz(edge_feats)

        self.register_buffer("node_feats_mu", nmu)
        self.register_buffer("node_feats_std", nstd)
        self.register_buffer("edge_feats_mu", emu)
        self.register_buffer("edge_feats_std", estd)
        # Store the *raw* (unstandardized) features; we re-standardize on-the-fly to the current device/dtype
        self.register_buffer("node_feats_raw", node_feats)
        self.register_buffer("edge_feats_raw", edge_feats)

        # Small MLPs to map geometry -> positive scales (via softplus)
        def mlp(in_dim, out_dim=1):
            dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
            mods = []
            for i in range(len(dims)-1):
                lin = nn.Linear(dims[i], dims[i+1])
                if use_sn: lin = nn.utils.spectral_norm(lin)
                mods.append(lin)
                if i < len(dims)-2:
                    mods.append(nn.SiLU())
            return nn.Sequential(*mods)

        self.node_mlp  = mlp(3, 1)  # -> log_m scale proxy
        self.edge_mlp  = mlp(3, 1)  # -> log_w scale proxy
        if self.w_structure == "offdiag":
            self.gamma_mlp = mlp(3, 1)  # node-coupled off-diagonal strength
        else:
            self.gamma_mlp = None

        if self.use_speed_scalar:
            self.log_speed2 = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_speed2", None)

    def _stdz_node(self):
        x = self.node_feats_raw
        return (x - self.node_feats_mu) / (self.node_feats_std + 1e-6)

    def _stdz_edge(self):
        x = self.edge_feats_raw
        return (x - self.edge_feats_mu) / (self.edge_feats_std + 1e-6)

    def M_vec(self) -> torch.Tensor:
        # Positive node-wise mass from geometry; inject physical V0
        m_raw = self.node_mlp(self._stdz_node()).squeeze(-1)
        m_pos = F.softplus(m_raw) + self.eps
        M = self.V0 * m_pos
        if self.normalize:
            M = M / (M.mean() + 1e-12)
        return M

    def W_diag_vec(self) -> torch.Tensor:
        # Positive edge-wise weight from geometry; inject physical V1inv
        w_raw = self.edge_mlp(self._stdz_edge()).squeeze(-1)
        w_pos = F.softplus(w_raw) + self.eps
        Wd = self.V1inv * w_pos
        if self.use_speed_scalar and self.log_speed2 is not None:
            Wd = Wd * torch.exp(self.log_speed2)
        if self.normalize:
            Wd = Wd / (Wd.mean() + 1e-12)
        return Wd

    def apply_W(self, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, Nnodes: int) -> torch.Tensor:
        Wd = self.W_diag_vec()
        y = Wd * e
        if self.w_structure == "offdiag":
            g_raw = self.gamma_mlp(self._stdz_node()).squeeze(-1)   # [N]
            gamma = F.softplus(g_raw) + self.eps                    # >= eps
            s = torch.zeros(*e.shape[:-1], Nnodes, dtype=e.dtype, device=e.device)
            s.index_add_(-1, src, e)
            s.index_add_(-1, dst, e)
            t = gamma * s
            y = y + t[..., src] + t[..., dst]
        return y

    @torch.no_grad()
    def calibrate_speed2(self, target_c2: float = 1.0):
        """Set global speed^2 so that mean(W)/mean(M) ~= target_c2 (if enabled)."""
        if not self.use_speed_scalar or self.log_speed2 is None or self.normalize:
            return
        Wd = self.W_diag_vec().detach()
        M  = self.M_vec().detach()
        c2_est = (Wd.mean() / (M.mean() + 1e-12)).clamp_min(1e-12).item()
        self.log_speed2.data = torch.tensor(
            math.log(max(1e-12, target_c2 / c2_est)),
            dtype=self.log_speed2.dtype, device=self.log_speed2.device
        )

# ------------------------- MeshFT-Net (DEC-based) -------------------------

class MeshFTNet(nn.Module):
    """
    DEC Hamiltonian model.
    state_mode:
      - "canonical": x=[q,p], dq/dt = M^{-1}p, dp/dt = -K q
      - "velocity" : x=[q,v], dq/dt = v,       dv/dt = -M^{-1}K q
    Time stepping: KDK (single W-application per kick).
    """
    def __init__(self, src, dst, hodge_module: nn.Module, state_mode: str = "canonical"):
        super().__init__()
        assert state_mode in ("velocity", "canonical")
        self.src = src; self.dst = dst
        self.N = hodge_module.V0.numel()
        self.hodge = hodge_module
        self.state_mode = state_mode
        self._target_c2 = 1.0  # can be set by caller
        self._nsub = 1

    def energy(self, x):
        q, s = x[...,0], x[...,1]
        M = self.hodge.M_vec()
        Bq = B_times_q(self.src, self.dst, q)
        W_Bq = self.hodge.apply_W(Bq, self.src, self.dst, self.N)
        term_q = 0.5 * (Bq * W_Bq).sum(dim=-1)
        if self.state_mode == "canonical":
            term_s = 0.5 * ((s**2) / (M + 1e-12)).sum(dim=-1)
        else:
            term_s = 0.5 * (M * (s**2)).sum(dim=-1)
        return term_s + term_q

    def vector_field(self, q, s):
        M = self.hodge.M_vec()
        Minv = 1.0 / (M + 1e-12)
        Bq = B_times_q(self.src, self.dst, q)
        W_Bq = self.hodge.apply_W(Bq, self.src, self.dst, self.N)
        Kq = BT_times_e(self.src, self.dst, W_Bq, self.N)
        if self.state_mode == "canonical":
            dqdt = Minv * s
            dsdt = - Kq
        else:
            dqdt = s
            dsdt = - (Minv * Kq)
        return dqdt, dsdt

    def kdk_step(self, q, s, dt: float):
        M = self.hodge.M_vec()
        Minv = 1.0 / (M + 1e-12)
        # Kick 1
        Bq = B_times_q(self.src, self.dst, q)
        W_Bq = self.hodge.apply_W(Bq, self.src, self.dst, self.N)
        Kq = BT_times_e(self.src, self.dst, W_Bq, self.N)
        if self.state_mode == "canonical":
            s_half = s - (dt * 0.5) * Kq
            q_new  = q + dt * (Minv * s_half)
        else:
            s_half = s - (dt * 0.5) * (Minv * Kq)
            q_new  = q + dt * s_half
        # Kick 2
        Bq_new = B_times_q(self.src, self.dst, q_new)
        W_Bq_new = self.hodge.apply_W(Bq_new, self.src, self.dst, self.N)
        Kq_new = BT_times_e(self.src, self.dst, W_Bq_new, self.N)
        if self.state_mode == "canonical":
            s_new = s_half - (dt * 0.5) * Kq_new
        else:
            s_new = s_half - (dt * 0.5) * (Minv * Kq_new)
        return q_new, s_new

    def forward(self, x, dt: float):
        # Substep KDK to satisfy stability on irregular meshes.
        q, s = x[..., 0], x[..., 1]
        n = max(1, int(getattr(self, "_nsub", 1)))
        dts = dt / n
        for _ in range(n):
            q, s = self.kdk_step(q, s, dts)
        return torch.stack([q, s], dim=-1)

# ------------------------- Tiny MGN (message-passing) -------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2, act=nn.SiLU, use_sn=False):
        super().__init__()
        dims = [in_dim] + [hidden]*(layers-1) + [out_dim]
        mods = []
        for i in range(len(dims)-1):
            lin = nn.Linear(dims[i], dims[i+1])
            if use_sn:
                lin = nn.utils.spectral_norm(lin)
            mods.append(lin)
            if i < len(dims)-2:
                mods.append(act())
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class GraphLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, use_sn=False):
        super().__init__()
        self.edge_mlp = MLP(2*node_dim + edge_dim, hidden, hidden, layers=2, use_sn=use_sn)
        self.node_mlp = MLP(node_dim + hidden, hidden, node_dim, layers=2, use_sn=use_sn)
    def forward(self, h, src, dst, eattr=None):
        if eattr is None:
            eattr = torch.zeros(dst.numel(), 1, device=h.device, dtype=h.dtype)
        hi = h[src]; hj = h[dst]
        m_ij = self.edge_mlp(torch.cat([hi, hj, eattr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_ij)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h

class MeshGraphNetVF(nn.Module):
    """
    Predicts vector field v(x); integrator x_{t+1} = x_t + dt * v(x_t).
    Node features = [x_nodes (q,s), coords (x,y), optional node_extras (e.g., V0)].
    Edge features = provided eattr (e.g., dx, dy, |e|).
    """
    def __init__(self, in_dim=5, hidden=64, layers=4, out_dim=2, edge_in_dim=3, use_sn=False, use_input_norm: bool = False):
        super().__init__()
        self.edge_in_dim = edge_in_dim
        self.use_input_norm = bool(use_input_norm)
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
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_dim=edge_in_dim, hidden=hidden, use_sn=use_sn) for _ in range(layers)])
        self.dec = mlp(hidden, out_dim, 2)

        self.register_buffer("mu_xy",     torch.zeros(1, 2))
        self.register_buffer("std_xy",    torch.ones(1, 2))
        self.register_buffer("mu_ex",     torch.zeros(1, 1))  # V0
        self.register_buffer("std_ex",    torch.ones(1, 1))
        self.register_buffer("mu_e",      torch.zeros(1, self.edge_in_dim))
        self.register_buffer("std_e",     torch.ones(1, self.edge_in_dim))
        self.register_buffer("sigma",     torch.ones(2))    

    @torch.no_grad()
    def set_normalization(self, mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                          sigma_q: float = 1.0, sigma_s: float = 1.0):
        self.mu_xy.copy_(mu_xy.to(self.mu_xy))
        self.std_xy.copy_(std_xy.to(self.std_xy))
        self.mu_ex.copy_(mu_ex.to(self.mu_ex))
        self.std_ex.copy_(std_ex.to(self.std_ex))
        self.mu_e.copy_(mu_e.to(self.mu_e))
        self.std_e.copy_(std_e.to(self.std_e))
        self.sigma[0] = float(sigma_q)
        self.sigma[1] = float(sigma_s)

    def forward(self, x_nodes, coords, src, dst, dt=None, eattr=None, node_extras: torch.Tensor=None):
        B, N, _ = x_nodes.shape
        if self.use_input_norm:
            x_in = x_nodes.clone()
            x_in[..., 0] = x_in[..., 0] / (self.sigma[0] + 1e-12)  # q
            x_in[..., 1] = x_in[..., 1] / (self.sigma[1] + 1e-12)  # s (p or v)
        else:
            x_in = x_nodes

        extras = node_extras if node_extras is not None else coords.new_zeros(N, 0)
        if self.use_input_norm and extras.numel() > 0:
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
        else:
            coords_n, extras_n = coords, extras

        Hnode = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([x_in, Hnode], dim=-1)
        h = self.enc(h)

        # Edge attributes
        ea0 = eattr if eattr is not None else torch.ones(dst.numel(), self.edge_in_dim, device=h.device, dtype=h.dtype)
        if self.use_input_norm:
            ea0 = (ea0 - self.mu_e) / (self.std_e + 1e-6)

        eattr_b = ea0.unsqueeze(0).expand(B, -1, -1).reshape(-1, ea0.shape[-1])

        offs = (torch.arange(B, device=dst.device, dtype=dst.dtype) * N).view(-1, 1)
        src_b = (src.view(1, -1) + offs).reshape(-1)
        dst_b = (dst.view(1, -1) + offs).reshape(-1)

        h_flat = h.reshape(B*N, -1)
        for gl in self.layers:
            h_flat = gl(h_flat, src_b, dst_b, eattr_b)
        v = self.dec(h_flat).view(B, N, -1)
        return v

def mgn_step_kdk(model: Dict, x: torch.Tensor, coords: torch.Tensor,
                 src: torch.Tensor, dst: torch.Tensor, dt: float, alpha: float) -> torch.Tensor:
    """
    KDK with component-wise splitting using the learned vector field:
      s_{n+1/2} = s_n + (dt/2)*v_s(q_n, s_n)
      q_{n+1}   = q_n +  dt   *v_q(q_n, s_{n+1/2})
      s_{n+1}   = s_{n+1/2} + (dt/2)*v_s(q_{n+1}, s_{n+1/2})
    """
    net = model["net"]
    # Kick (half) on s    @ x_n
    v0 = net(x, coords, src, dst, dt,
             eattr=model.get("eattr", None),
             node_extras=model.get("node_extras", None))
    q = x[..., 0]; s = x[..., 1]
    s_half = s + (dt * alpha * 0.5) * v0[..., 1]
    x_mid = torch.stack([q, s_half], dim=-1)

    # Drift (full) on q   @ (q_n, s_{n+1/2})
    v_mid = net(x_mid, coords, src, dst, dt,
                eattr=model.get("eattr", None),
                node_extras=model.get("node_extras", None))
    q_new = q + (dt * alpha) * v_mid[..., 0]
    x_tmp = torch.stack([q_new, s_half], dim=-1)

    # Kick (half) on s    @ (q_{n+1}, s_{n+1/2})
    v_new = net(x_tmp, coords, src, dst, dt,
                eattr=model.get("eattr", None),
                node_extras=model.get("node_extras", None))
    s_new = s_half + (dt * alpha * 0.5) * v_new[..., 1]
    return torch.stack([q_new, s_new], dim=-1)

# ------------------------- EnergyNet for HP (no Hodge/DEC needed) -------------------------

def apply_J_to_grad(gradH: torch.Tensor, state_mode: str = "canonical",
                    M_data: torch.Tensor = None) -> torch.Tensor:
    """
    Map gradH = [dH/dq, dH/ds] to the Hamiltonian vector field J∇H.
    We use the canonical symplectic form by default:
      canonical: x=[q,p] -> [dq/dt, dp/dt] = [ dH/dp, - dH/dq ]
      velocity : x=[q,v] -> [dq/dt, dv/dt] = [ dH/dv, - dH/dq ]
    If you need mass-weighted J for 'velocity', adapt here and pass M_data.
    """
    gq, gs = gradH[..., 0], gradH[..., 1]
    dqdt = gs
    dsdt = -gq
    return torch.stack([dqdt, dsdt], dim=-1)

class EnergyNet(nn.Module):
    """
    Graph-based scalar Hamiltonian H(x). Aggregates per-node features with message passing, then sums.
    Node features = [x_nodes (q,s), coords (x,y), optional node_extras (e.g., V0)].
    """
    def __init__(self, node_in_dim=5, edge_in_dim=3, hidden=64, layers=4, use_sn=False, use_input_norm: bool = False):
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
        self.enc = mlp(node_in_dim, hidden, 2)
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_dim=edge_in_dim, hidden=hidden, use_sn=use_sn) for _ in range(layers)])
        self.dec_node = mlp(hidden, 1, 2)
        self.use_input_norm = bool(use_input_norm)

        self.register_buffer("mu_xy",  torch.zeros(1, 2))
        self.register_buffer("std_xy", torch.ones(1, 2))
        self.register_buffer("mu_ex",  torch.zeros(1, 1))
        self.register_buffer("std_ex", torch.ones(1, 1))
        self.register_buffer("mu_e",   torch.zeros(1, edge_in_dim))
        self.register_buffer("std_e",  torch.ones(1, edge_in_dim))
        self.register_buffer("sigma",  torch.ones(2)) 

    @torch.no_grad()
    def set_normalization(self, mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                          sigma_q: float = 1.0, sigma_s: float = 1.0):
        self.mu_xy.copy_(mu_xy.to(self.mu_xy))
        self.std_xy.copy_(std_xy.to(self.std_xy))
        self.mu_ex.copy_(mu_ex.to(self.mu_ex))
        self.std_ex.copy_(std_ex.to(self.std_ex))
        self.mu_e.copy_(mu_e.to(self.mu_e))
        self.std_e.copy_(std_e.to(self.std_e))
        self.sigma[0] = float(sigma_q)
        self.sigma[1] = float(sigma_s)

    def forward(self, x_nodes, coords, src, dst, eattr=None, node_extras: torch.Tensor=None):
        B, N, _ = x_nodes.shape
        if self.use_input_norm:
            x_in = x_nodes.clone()
            x_in[..., 0] = x_in[..., 0] / (self.sigma[0] + 1e-12)
            x_in[..., 1] = x_in[..., 1] / (self.sigma[1] + 1e-12)
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
        else:
            x_in = x_nodes
            coords_n = coords

        extras = node_extras if node_extras is not None else coords.new_zeros(N, 0)
        if self.use_input_norm and extras.numel()>0:
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
        else:
            extras_n = extras

        Hnode = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([x_in, Hnode], dim=-1)
        h = self.enc(h)

        ea0 = eattr if eattr is not None else torch.ones(dst.numel(), 3, device=h.device, dtype=h.dtype)
        if self.use_input_norm:
            ea0 = (ea0 - self.mu_e) / (self.std_e + 1e-6)
        eattr_b = ea0.unsqueeze(0).expand(B, -1, -1).reshape(-1, ea0.shape[-1])

        offs = (torch.arange(B, device=dst.device, dtype=dst.dtype) * N).view(-1, 1)
        src_b = (src.view(1, -1) + offs).reshape(-1)
        dst_b = (dst.view(1, -1) + offs).reshape(-1)

        h_flat = h.reshape(B*N, -1)
        for gl in self.layers:
            h_flat = gl(h_flat, src_b, dst_b, eattr_b)
        e_node = self.dec_node(h_flat).view(B, N, 1)
        return e_node.sum(dim=[1, 2])

def build_edge_attr(coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Edge features: (dx, dy, |e|) with periodic minimum-image convention."""
    # infer domain size from bbox (coords are in [0,L) by construction)
    Lx = (coords[:, 0].max() - coords[:, 0].min()).item()
    Ly = (coords[:, 1].max() - coords[:, 1].min()).item()
    dv = coords[dst] - coords[src]  # [E,2]
    if Lx > 0 and Ly > 0:
        dvx = dv[:, 0] - torch.round(dv[:, 0] / Lx) * Lx
        dvy = dv[:, 1] - torch.round(dv[:, 1] / Ly) * Ly
        dv = torch.stack([dvx, dvy], dim=-1)
    elen = dv.norm(dim=-1, keepdim=True)          # [E,1]
    return torch.cat([dv, elen], dim=-1)          # [E,3]

# ------------------------- HNN-like MGN -------------------------

class _SeparableNodeEnergy(nn.Module):
    """
    Generic per-node energy network that takes a SINGLE scalar field (q OR p)
    plus geometric node/edge context, does message passing, and sums to a scalar.
    We will instantiate two copies: U_net (uses q) and T_net (uses p).
    Node features: [scalar_field, x, y, V0]; Edge features: [dx, dy, |e|].
    """
    def __init__(self, hidden=64, layers=4, edge_in_dim=3, use_sn=False, use_input_norm: bool=False):
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
        # scalar + x + y + V0 = 4
        self.use_input_norm = bool(use_input_norm)
        self.enc = mlp(4, hidden, 2)
        self.layers = nn.ModuleList([GraphLayer(hidden, edge_dim=edge_in_dim, hidden=hidden, use_sn=use_sn) for _ in range(layers)])
        self.dec_node = mlp(hidden, 1, 2)   # per-node energy -> sum

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

    def forward(self, field_scalar, coords, src, dst, eattr=None, node_extras: torch.Tensor=None):
        B, N = field_scalar.shape
        s = field_scalar / (self.sigma_field + 1e-12) if self.use_input_norm else field_scalar

        extras = node_extras if node_extras is not None else coords.new_zeros(N, 0)
        if self.use_input_norm and extras.numel()>0:
            coords_n = (coords - self.mu_xy) / (self.std_xy + 1e-6)
            extras_n = (extras - self.mu_ex) / (self.std_ex + 1e-6)
        else:
            coords_n, extras_n = coords, extras

        node_feat = torch.cat([coords_n, extras_n], dim=-1).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([s.unsqueeze(-1), node_feat], dim=-1)
        h = self.enc(h)

        ea0 = eattr if eattr is not None else torch.ones(dst.numel(), 3, device=h.device, dtype=h.dtype)
        if self.use_input_norm:
            ea0 = (ea0 - self.mu_e) / (self.std_e + 1e-6)
        eattr_b = ea0.unsqueeze(0).expand(B, -1, -1).reshape(-1, ea0.shape[-1])

        offs = (torch.arange(B, device=dst.device, dtype=dst.dtype) * N).view(-1, 1)
        src_b = (src.view(1, -1) + offs).reshape(-1)
        dst_b = (dst.view(1, -1) + offs).reshape(-1)

        h_flat = h.reshape(B*N, -1)
        for gl in self.layers:
            h_flat = gl(h_flat, src_b, dst_b, eattr_b)
        e_node = self.dec_node(h_flat).view(B, N, 1)
        return e_node.sum(dim=[1, 2])


class HNNSeparableSymplectic(nn.Module):
    """
    Pure HNN with separable H(q,p)=U(q)+T(p) (canonical).
    Time stepping: Symplectic Euler (kick–drift):
      p_{n+1} = p_n - dt * dU/dq(q_n)
      q_{n+1} = q_n + dt * dT/dp(p_{n+1})
    """
    def __init__(self, U_net: _SeparableNodeEnergy, T_net: _SeparableNodeEnergy,
                 coords: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                 eattr: torch.Tensor = None, node_extras: torch.Tensor = None):
        super().__init__()
        self.U_net = U_net
        self.T_net = T_net
        self.register_buffer("coords", coords.clone())
        self.register_buffer("src", src.clone())
        self.register_buffer("dst", dst.clone())
        self.eattr = eattr
        self.node_extras = node_extras
        self.state_mode = "canonical"
        self.integrator = getattr(self, "integrator", "kdk")

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Return H(x) = U(q) + T(p) for a batch: shape [B]."""
        q, p = x[..., 0], x[..., 1]
        U = self.U_net(q, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        T = self.T_net(p, self.coords, self.src, self.dst, self.eattr, self.node_extras)
        return U + T

    def _grad_U_wrt_q(self, q: torch.Tensor) -> torch.Tensor:
        """Compute dU/dq(q) with create_graph=True to allow learning."""
        q_req = q.detach().requires_grad_(True)
        U = self.U_net(q_req, self.coords, self.src, self.dst, self.eattr, self.node_extras)  # [B]
        (gq,) = torch.autograd.grad(U.sum(), q_req, create_graph=self.training)
        return gq  # [B,N]

    def _grad_T_wrt_p(self, p: torch.Tensor) -> torch.Tensor:
        """Compute dT/dp(p) with create_graph=True to allow learning."""
        p_req = p.detach().requires_grad_(True)
        T = self.T_net(p_req, self.coords, self.src, self.dst, self.eattr, self.node_extras)  # [B]
        (gp,) = torch.autograd.grad(T.sum(), p_req, create_graph=self.training)
        return gp  # [B,N]

    def _verlet_step(self, q: torch.Tensor, p: torch.Tensor, dt: float):
        # Kick (half) at q_n
        gq = self._grad_U_wrt_q(q)          # dU/dq(q_n)
        p_half = p - 0.5 * dt * gq
        # Drift (full)
        dTdp_half = self._grad_T_wrt_p(p_half)
        q_new = q + dt * dTdp_half
        # Kick (half) at q_{n+1}
        gq_new = self._grad_U_wrt_q(q_new)
        p_new = p_half - 0.5 * dt * gq_new
        return q_new, p_new

    def forward(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        q, p = x[..., 0], x[..., 1]
        n = max(1, int(getattr(self, "_nsub", 1)))
        dts = dt / n

        if self.integrator in ("kdk", "verlet"):
            for _ in range(n):
                q, p = self._verlet_step(q, p, dts)
            return torch.stack([q, p], dim=-1)

        # 'se' fallback
        for _ in range(n):
            dUdq = self._grad_U_wrt_q(q)
            p = p - dts * dUdq
            dTdp = self._grad_T_wrt_p(p)
            q = q + dts * dTdp
        return torch.stack([q, p], dim=-1)


# ------------------------- Analytic plane-wave data (CANONICAL by default) -------------------------

def sample_plane_params(nx, ny, Lx=1.0, Ly=1.0, c=1.0, kmax=4):
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
    state_mode_data:
      - "canonical": x=[q, p] with p = M_data * dq/dt
      - "velocity" : x=[q, v] with v = dq/dt
    """
    def __init__(self, nx, ny, dt, size, coords, M_data: torch.Tensor,
                 Lx=1.0, Ly=1.0, c_wave=1.0, kmax=4, device="cpu",
                 state_mode_data: str = "canonical"):
        super().__init__()
        assert state_mode_data in ("canonical","velocity")
        self.nx, self.ny, self.dt = nx, ny, dt
        self.size = size
        self.coords = coords
        self.M_data = M_data.to(device)  # [N]
        self.Lx, self.Ly, self.c_wave = Lx, Ly, c_wave
        self.kmax = kmax
        self.device = device
        self.state_mode_data = state_mode_data
        self.params = [sample_plane_params(nx, ny, Lx, Ly, c_wave, kmax) for _ in range(size)]
        self.t0 = np.random.uniform(0, 2*np.pi, size).astype(np.float32)

    def __len__(self): return self.size

    def _pack(self, q, v):
        if self.state_mode_data == "canonical":
            p = self.M_data * v
            return torch.stack([q, p], dim=-1)
        else:
            return torch.stack([q, v], dim=-1)

    def __getitem__(self, idx):
        kvec, omega, phi, amp = self.params[idx]
        t = float(self.t0[idx])
        q0, v0 = plane_wave_q_and_v(self.coords, t, kvec, omega, phi, amp, self.device)
        q1, v1 = plane_wave_q_and_v(self.coords, t + self.dt, kvec, omega, phi, amp, self.device)
        x0 = self._pack(q0, v0)
        x1 = self._pack(q1, v1)
        meta = dict(kvec=torch.tensor(kvec), omega=omega, phi=phi, amp=amp, t=t)
        return x0, x1, meta

# ------------------------- fixed missing-data masks -------------------------

def build_fixed_obs_mask(coords: torch.Tensor, miss_ratio: float, mode: str,
                         grid_stride: int, nx: int, ny: int, Lx: float, Ly: float,
                         seed: int, device: str) -> torch.Tensor:
    """
    Build a FIXED node-wise mask [N,2] for a split (train/val).
    - random: Bernoulli(1-miss_ratio) per node with given seed.
    - grid: keep nodes whose (bin_x % grid_stride == 0 and bin_y % grid_stride == 0).
    """
    N = coords.shape[0]
    if miss_ratio <= 1e-12 and mode == "random":
        keep = torch.ones(N, 1, dtype=torch.float32)
    elif mode == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        u = torch.rand(N, 1, generator=g)  # CPU generator for determinism
        keep = (u >= miss_ratio).float()
    else:
        hx = Lx / max(1, nx); hy = Ly / max(1, ny)
        gx = torch.clamp((coords[:, 0] / hx).floor().long(), min=0, max=nx-1)
        gy = torch.clamp((coords[:, 1] / hy).floor().long(), min=0, max=ny-1)
        sel = ((gx % grid_stride) == 0) & ((gy % grid_stride) == 0)
        keep = sel.float().unsqueeze(-1)
    # duplicate to 2 channels (q & p/v)
    mask_2c = keep.repeat(1, 2).to(device)
    return mask_2c  # [N,2]

def expand_mask_for_batch(mask_2c: torch.Tensor, B: int) -> torch.Tensor:
    """Expand fixed [N,2] mask to [B,N,2] without copying."""
    return mask_2c.unsqueeze(0).expand(B, -1, -1)

# ------------------------- energy & rollout -------------------------

def energy_proxy(x: torch.Tensor) -> torch.Tensor:
    """Fallback Euclidean proxy (for vanilla MGN) — not used for drift when a theory Hodge is provided."""
    q, s = x[..., 0], x[..., 1]
    return 0.5 * (q.pow(2).sum(dim=-1) + s.pow(2).sum(dim=-1))

def energy_hamiltonian_meshft(model: "MeshFTNet", x: torch.Tensor) -> torch.Tensor:
    return model.energy(x)

def energy_from_hodge(hodge: nn.Module, src, dst, x: torch.Tensor, N: int, state_mode: str="canonical") -> torch.Tensor:
    q, s = x[..., 0], x[..., 1]
    M = hodge.M_vec()
    Bq = B_times_q(src, dst, q)
    W_Bq = hodge.apply_W(Bq, src, dst, N)
    term_q = 0.5 * (Bq * W_Bq).sum(dim=-1)
    if state_mode == "canonical":
        term_s = 0.5 * ((s**2) / (M + 1e-12)).sum(dim=-1)
    else:
        term_s = 0.5 * (M * (s**2)).sum(dim=-1)
    return term_s + term_q

def phys_rel_error_from_hodge(hodge: nn.Module, src, dst, x: torch.Tensor, y: torch.Tensor,
                              N: int, state_mode: str="canonical") -> torch.Tensor:
    """
    Relative error under the quadratic form induced by (M,W):
      ||x-y||_H / ||y||_H, where ||z||_H^2 = 2 * Energy_H(z).
    """
    Ez = energy_from_hodge(hodge, src, dst, x - y, N, state_mode=state_mode)
    Ey = energy_from_hodge(hodge, src, dst, y,     N, state_mode=state_mode)
    rel = torch.sqrt(Ez / (Ey + 1e-12))
    return rel

@torch.no_grad()
def _estimate_omega_max(src, dst, hodge: nn.Module, iters: int = 20):
    """
    Power iteration on A = M^{-1} K with K = B^T W B to estimate the largest eigenvalue λ.
    Then ω_max ≈ sqrt(λ).
    """
    N = hodge.V0.numel()
    q = torch.randn(N, device=hodge.V0.device, dtype=hodge.V0.dtype)
    q = q / (q.norm() + 1e-12)
    lam = torch.tensor(0.0, device=q.device, dtype=q.dtype)
    for _ in range(iters):
        Bq = B_times_q(src, dst, q)
        W_Bq = hodge.apply_W(Bq, src, dst, N)
        z = BT_times_e(src, dst, W_Bq, N)              # K q
        v = z / (hodge.M_vec() + 1e-12)                # M^{-1} K q
        lam = (q * v).sum()
        vnorm = v.norm() + 1e-12
        q = v / vnorm
    lam = lam.clamp_min(1e-12)
    return float(torch.sqrt(lam))

@torch.no_grad()
def rollout_eval(model_kind: str, model, src, dst, coords, dt: float, batch_meta, T: int,
                 device: str, eval_hodge: nn.Module = None, M_data: torch.Tensor = None,
                 return_energy_series: bool = False):
    """
    Roll out and compute (relF, relMedian, energy drift, steps_done[, E_hist]).

    If return_energy_series=True, also returns E_hist [steps_done+1, B] with
    per-step energies computed using eval_hodge when provided (common physical energy).
    """
    assert M_data is not None, "M_data (V0) must be provided for canonical packaging."

    # Resolve state_mode safely
    state_mode = model.get("state_mode", "canonical") if isinstance(model, dict) else getattr(model, "state_mode", "canonical")

    # Build initial batch
    B = len(batch_meta["t"])
    x = []
    for i in range(B):
        q0, v0 = plane_wave_q_and_v(
            coords, float(batch_meta["t"][i]),
            batch_meta["kvec"][i].cpu().numpy(),
            float(batch_meta["omega"][i]),
            float(batch_meta["phi"][i]),
            float(batch_meta["amp"][i]), device
        )
        s0 = (M_data * v0) if state_mode == "canonical" else v0
        x.append(torch.stack([q0, s0], dim=-1))
    x = torch.stack(x, dim=0)

    # Energy compute helper (prefer common theory Hodge if provided)
    def _energy_now(z: torch.Tensor) -> torch.Tensor:
        if eval_hodge is not None:
            return energy_from_hodge(eval_hodge, src, dst, z, N=coords.shape[0], state_mode=state_mode)
        elif model_kind == "meshft_net":
            was = model.training; model.eval()
            e = energy_hamiltonian_meshft(model, z)
            if was: model.train()
            return e
        elif model_kind == "hnn":
            was = model.training; model.eval()
            e = model.energy(z)
            if was: model.train()
            return e
        else:
            return energy_proxy(z)

    # Initial energy
    e0 = _energy_now(x)                     # [B]
    E_hist = [e0]                           # list of [B]

    rel_hist = []; steps_done = 0
    for k in range(T):
        # Advance one step
        if model_kind == "meshft_net":
            x = model(x, dt)
        elif model_kind == "hnn":
            with torch.enable_grad():
                x = model(x, dt)
            x = x.detach()
        else:
            integrator = model.get("integrator", "euler")
            if integrator == "kdk":
                x = mgn_step_kdk(model, x, coords, src, dst, dt, 1.0) 
            elif integrator == "rk2":
                k1 = model["net"](x, coords, src, dst, dt, eattr=model.get("eattr", None),
                                node_extras=model.get("node_extras", None))
                x_mid = x + 0.5 * dt * k1
                k2 = model["net"](x_mid, coords, src, dst, dt, eattr=model.get("eattr", None),
                                node_extras=model.get("node_extras", None))
                x = x + dt * k2
            else:
                v = model["net"](x, coords, src, dst, dt, eattr=model.get("eattr", None),
                                node_extras=model.get("node_extras", None))
                x = x + dt * v

        # Non-finite guard
        if not torch.isfinite(x).all():
            bad = torch.full((B,), float("inf"), device=device)
            rel_hist.append(bad)
            steps_done += 1
            break

        # Ground truth at current time
        t_now = batch_meta["t"].to(device).clone() + (k+1) * dt
        gt = []
        for i in range(B):
            qg, vg = plane_wave_q_and_v(
                coords, float(t_now[i]),
                batch_meta["kvec"][i].cpu().numpy(),
                float(batch_meta["omega"][i]),
                float(batch_meta["phi"][i]),
                float(batch_meta["amp"][i]), device
            )
            sg = (M_data * vg) if state_mode == "canonical" else vg
            gt.append(torch.stack([qg, sg], dim=-1))
        gt = torch.stack(gt, dim=0)

        # Relative error (common physical norm if available)
        if eval_hodge is not None:
            rel = phys_rel_error_from_hodge(eval_hodge, src, dst, x, gt, N=coords.shape[0], state_mode=state_mode)
        else:
            rel = (x - gt).reshape(B, -1).norm(dim=-1) / (gt.reshape(B, -1).norm(dim=-1) + 1e-12)
        rel_hist.append(rel); steps_done += 1

        # Energy at this step
        E_hist.append(_energy_now(x))

    rel_hist = torch.stack(rel_hist, dim=0) if len(rel_hist) > 0 else torch.zeros(1, B, device=device)
    rel_final = rel_hist[-1].mean().item()
    rel_median = rel_hist.median().item()

    # Drift vs. initial energy
    ef = _energy_now(x)
    if (not torch.isfinite(e0).all()) or (not torch.isfinite(ef).all()):
        drift = float("inf")
    else:
        drift = (ef - e0).abs() / (e0.abs() + 1e-12)
        drift = drift.mean().item()

    if return_energy_series:
        E_hist = torch.stack(E_hist, dim=0)   # [steps_done+1, B]
        return rel_final, rel_median, drift, steps_done, E_hist
    else:
        return rel_final, rel_median, drift, steps_done

# ------------------------- training & eval (uses FIXED masks if provided) -------------------------

def train_one_epoch_meshft(model: MeshFTNet, loader, dt, device, nx, ny, coords, Lx, Ly,
                        miss_ratio=0.0, miss_mode="random", grid_stride=2, use_bar=True,
                        use_weighted_loss=True, sigma_q=1.0, sigma_p=1.0,
                        fixed_mask_2c: torch.Tensor = None):
    if getattr(model, "_opt", None) is None:
        return 0.0
    model.train(); opt = model._opt; loss_meter = 0.0
    it = tqdm(loader, desc="[MeshFT-Net train]", leave=False) if use_bar else loader
    for x0, x1, _ in it:
        x0, x1 = to_device(x0, x1, device=device)
        pred = model(x0, dt)
        if fixed_mask_2c is not None:
            mask = expand_mask_for_batch(fixed_mask_2c, x0.shape[0])
        else:
            # fallback random mask (not recommended)
            mask = build_fixed_obs_mask(coords, miss_ratio, miss_mode, grid_stride, nx, ny, Lx, Ly, seed=0, device=device)
            mask = expand_mask_for_batch(mask, x0.shape[0])
        loss = masked_weighted_mse(pred, x1, mask, sigma_q, sigma_p) if use_weighted_loss else masked_mse(pred, x1, mask)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); loss_meter += loss.item()
        if use_bar: it.set_postfix(loss=f"{loss.item():.3e}")
    return loss_meter / max(1, len(loader))

@torch.no_grad()
def eval_one_epoch_meshft(model: MeshFTNet, loader, dt, device, use_bar=True, fixed_mask_2c: torch.Tensor = None, apply_mask_to_metrics: bool = True):
    model.eval()
    mse = ps = 0.0; c = 0
    it = tqdm(loader, desc="[MeshFT-Net val]", leave=False) if use_bar else loader
    for x0, x1, _ in it:
        x0, x1 = to_device(x0, x1, device=device)
        pred = model(x0, dt)
        if apply_mask_to_metrics and fixed_mask_2c is not None:
            mask = expand_mask_for_batch(fixed_mask_2c, x0.shape[0])
            m = masked_mse(pred, x1, mask).item()
            mse += m
            ps  += (99.0 if m <= 1e-20 else 20.0 * math.log10(1.0) - 10.0 * math.log10(m))
        else:
            mse += F.mse_loss(pred, x1).item()
            ps  += psnr(pred, x1)
        c += 1
    return mse/max(1,c), ps/max(1,c)

def train_one_epoch_mgn(model: Dict, loader, dt, device, nx, ny, coords, Lx, Ly,
                        lam_ham=0.0, src=None, dst=None,
                        miss_ratio=0.0, miss_mode="random", grid_stride=2, use_bar=True,
                        use_weighted_loss=True, sigma_q=1.0, sigma_p=1.0,
                        fixed_mask_2c: torch.Tensor = None,
                        # --- extras for HP stabilizers ---
                        eval_hodge: nn.Module = None, n_nodes: int = None,
                        state_mode: str = "canonical",
                        ham_energy_consistency: bool = False, ham_energy_cons_w: float = 0.05):
    """
    For lam_ham==0: vanilla MGN training (no structural constraint).
    For lam_ham>0: Hamiltonian penalty using EnergyNet.
    """
    if model.get("_opt", None) is None:
        return 0.0
    net = model["net"]; enet = model.get("energy_net", None)
    net.train()
    if enet is not None: enet.train()
    loss_meter = 0.0
    it = tqdm(loader, desc="[MGN train]" if lam_ham==0 else "[MGN-HP train]", leave=False) if use_bar else loader
    for x0, x1, _ in it:
        x0, x1 = to_device(x0, x1, device=device)
        k1 = net(x0, model["coords"], src, dst, dt,
                 eattr=model.get("eattr", None),
                 node_extras=model.get("node_extras", None))
        integrator = model.get("integrator", "euler")

        if integrator == "kdk":
            x_pred = mgn_step_kdk(model, x0, model["coords"], src, dst, dt, 1.0)
        elif integrator == "rk2":
            x_mid = x0 + 0.5 * dt * k1
            k2 = net(x_mid, model["coords"], src, dst, dt,
                    eattr=model.get("eattr", None),
                    node_extras=model.get("node_extras", None))
            x_pred = x0 + dt * k2
        else:
            x_pred = x0 + dt * k1

        # data-fit loss
        if fixed_mask_2c is not None:
            mask = expand_mask_for_batch(fixed_mask_2c, x0.shape[0])
        else:
            mask = build_fixed_obs_mask(coords, miss_ratio, miss_mode, grid_stride, nx, ny, Lx, Ly, seed=0, device=device)
            mask = expand_mask_for_batch(mask, x0.shape[0])
        loss_fit = masked_weighted_mse(x_pred, x1, mask, sigma_q, sigma_p) if use_weighted_loss else masked_mse(x_pred, x1, mask)
        loss = loss_fit

        # Hamiltonian penalty via EnergyNet (no Hodge)
        if lam_ham > 0.0 and enet is not None:
            x0_req = x0.detach().requires_grad_(True)
            H = enet(x0_req, model["coords"], src, dst,
                     eattr=model.get("eattr", None),
                     node_extras=model.get("node_extras", None))
            g = torch.autograd.grad(H.sum(), x0_req, create_graph=True)[0]
            state_mode_local = model.get("state_mode", state_mode)
            v_ham = apply_J_to_grad(g, state_mode=state_mode_local)
            # main penalty: J∇H
            loss_ham = F.mse_loss(k1, v_ham)
            loss = loss + lam_ham * loss_ham
            # additional weak penalty: ΔH≃0
            if ham_energy_consistency and (eval_hodge is not None) and (n_nodes is not None):
                dH = energy_from_hodge(eval_hodge, src, dst, x_pred, N=n_nodes, state_mode=state_mode_local) - \
                     energy_from_hodge(eval_hodge, src, dst, x0,     N=n_nodes, state_mode=state_mode_local)
                loss = loss + ham_energy_cons_w * (dH**2).mean()

        params = list(net.parameters()) + (list(enet.parameters()) if enet is not None else [])
        opt = model["_opt"]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); loss_meter += loss.item()
        if use_bar: it.set_postfix(loss=f"{loss.item():.3e}", fit=f"{loss_fit.item():.3e}")
    return loss_meter / max(1, len(loader))

@torch.no_grad()
def eval_one_epoch_mgn(model: Dict, loader, dt, device, src=None, dst=None, use_bar=True, fixed_mask_2c: torch.Tensor = None, apply_mask_to_metrics: bool = True):
    net = model["net"]; net.eval()
    if model.get("energy_net", None) is not None: model["energy_net"].eval()
    mse = ps = 0.0; c = 0
    it = tqdm(loader, desc="[MGN val]", leave=False) if use_bar else loader
    for x0, x1, _ in it:
        x0, x1 = to_device(x0, x1, device=device)
        integrator = model.get("integrator", "euler")

        if integrator == "kdk":
            x_pred = mgn_step_kdk(model, x0, model["coords"], src, dst, dt, 1.0)
        elif integrator == "rk2":
            k1 = net(x0, model["coords"], src, dst, dt,
                    eattr=model.get("eattr", None),
                    node_extras=model.get("node_extras", None))
            x_mid = x0 + 0.5 * dt * k1
            k2 = net(x_mid, model["coords"], src, dst, dt,
                    eattr=model.get("eattr", None),
                    node_extras=model.get("node_extras", None))
            x_pred = x0 + dt * k2
        else:
            v = net(x0, model["coords"], src, dst, dt,
                    eattr=model.get("eattr", None),
                    node_extras=model.get("node_extras", None))
            x_pred = x0 + dt * v

        if apply_mask_to_metrics and fixed_mask_2c is not None:
            mask = expand_mask_for_batch(fixed_mask_2c, x0.shape[0])
            m = masked_mse(x_pred, x1, mask).item()
            mse += m
            ps  += (99.0 if m <= 1e-20 else 20.0 * math.log10(1.0) - 10.0 * math.log10(m))
        else:
            mse += F.mse_loss(x_pred, x1).item(); ps += psnr(x_pred, x1)
        c += 1
    return mse/max(1,c), ps/max(1,c)

def train_one_epoch_hnn(model: HNNSeparableSymplectic, loader, dt, device,
                        nx, ny, coords, Lx, Ly, miss_ratio=0.0, miss_mode="random",
                        grid_stride=2, use_bar=True,
                        use_weighted_loss=True, sigma_q=1.0, sigma_p=1.0,
                        fixed_mask_2c: torch.Tensor = None):
    if getattr(model, "_opt", None) is None:
        return 0.0
    model.train(); opt = model._opt; loss_meter = 0.0
    it = tqdm(loader, desc="[HNN train]", leave=False) if use_bar else loader
    for x0, x1, _ in it:
        x0, x1 = to_device(x0, x1, device=device)
        with torch.enable_grad():
            pred = model(x0, dt)
            if fixed_mask_2c is not None:
                mask = expand_mask_for_batch(fixed_mask_2c, x0.shape[0])
            else:
                mask = build_fixed_obs_mask(coords, miss_ratio, miss_mode, grid_stride, nx, ny, Lx, Ly, seed=0, device=device)
                mask = expand_mask_for_batch(mask, x0.shape[0])
            loss = (masked_weighted_mse(pred, x1, mask, sigma_q, sigma_p)
                    if use_weighted_loss else masked_mse(pred, x1, mask))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.U_net.parameters()) + list(model.T_net.parameters()), 1.0)
        opt.step(); loss_meter += loss.item()
        if use_bar: it.set_postfix(loss=f"{loss.item():.3e}")
    return loss_meter / max(1, len(loader))

@torch.no_grad()
def eval_one_epoch_hnn(model: HNNSeparableSymplectic, loader, dt, device,
                       use_bar=True, fixed_mask_2c: torch.Tensor = None, apply_mask_to_metrics: bool = True):
    model.eval()
    mse = ps = 0.0; c = 0
    it = tqdm(loader, desc="[HNN val]", leave=False) if use_bar else loader
    for x0, x1, _ in it:
        x0, x1 = to_device(x0, x1, device=device)
        with torch.enable_grad():
            pred = model(x0, dt)
        pred = pred.detach()
        if apply_mask_to_metrics and fixed_mask_2c is not None:
            mask = expand_mask_for_batch(fixed_mask_2c, x0.shape[0])
            m = masked_mse(pred, x1, mask).item()
            mse += m
            ps  += (99.0 if m <= 1e-20 else 20.0 * math.log10(1.0) - 10.0 * math.log10(m))
        else:
            mse += F.mse_loss(pred, x1).item()
            ps  += psnr(pred, x1)
        c += 1
    return mse/max(1,c), ps/max(1,c)

# ------------------------- helpers to build hodge & optim -------------------------

def make_hodge(mode: str, V0: torch.Tensor, V1inv: torch.Tensor,
               use_speed_scalar: bool, w_structure: str, offdiag_init: float,
               normalize: bool, c_speed: float,
               # --- new for geometry-conditioned mode ---
               coords: torch.Tensor = None, src: torch.Tensor = None, dst: torch.Tensor = None,
               geom_hidden: int = 64, geom_layers: int = 2, geom_use_sn: bool = False):
    """
    Returns (hodge_module, trainable_params)
    mode:
      - "theory"     : fixed Hodge (no learnable params)
      - "learn"      : free per-node/edge parameters (log_m/log_w/log_gamma)
      - "learn_geom" : predict Hodge from geometry via small MLPs
    """
    if mode == "theory":
        h = HodgeBlockTheory(V0, V1inv, c_speed=c_speed, use_speed_scalar=use_speed_scalar)
        return h, []
    elif mode == "learn":
        h = HodgeBlockLearnable(V0, V1inv,
                                use_speed_scalar=use_speed_scalar,
                                w_structure=w_structure,
                                offdiag_init=offdiag_init,
                                normalize=normalize)
        params = list(h.parameters())
        return h, params
    elif mode == "learn_geom":
        assert coords is not None and src is not None and dst is not None, \
            "coords/src/dst are required for 'learn_geom' mode."
        h = HodgeBlockGeomMLP(coords, src, dst, V0, V1inv,
                              use_speed_scalar=use_speed_scalar,
                              w_structure=w_structure,
                              normalize=normalize,
                              hidden=geom_hidden, layers=geom_layers,
                              use_sn=geom_use_sn)
        params = list(h.parameters())
        return h, params
    else:
        raise ValueError("hodge mode must be 'learn', 'learn_geom', or 'theory'")

def make_optimizer_if_any(params, lr=1e-3, wd=1e-6):
    """Return AdamW over trainable params, or None if no params."""
    params = [p for p in params if p.requires_grad]
    if len(params) == 0:
        return None
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

# ------------------------- one configuration run -------------------------

def run_one_config(args, nx, ny, coords, src, dst, V0, V1inv, train_size: int, miss_ratio: float, seed: int, Lx: float, Ly: float):
    set_seed(seed)
    device = args.device

    # Dataset (CANONICAL by default): p = M_data * dq/dt
    M_data = V0.clone().to(device)
    train_ds = PlaneWaveDataset(nx, ny, dt=args.dt, size=train_size, coords=coords,
                                M_data=M_data, Lx=Lx, Ly=Ly, c_wave=args.c_wave, kmax=args.kmax,
                                device=device, state_mode_data=args.data_state_mode)
    val_ds   = PlaneWaveDataset(nx, ny, dt=args.dt, size=args.val_size, coords=coords,
                                M_data=M_data, Lx=Lx, Ly=Ly, c_wave=args.c_wave, kmax=args.kmax,
                                device=device, state_mode_data=args.data_state_mode)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Fixed masks (shared across models). Seeds default to seed+100 / seed+200.
    mask_seed_tr = args.mask_seed_train if args.mask_seed_train is not None else int(seed) + 100
    mask_seed_va = args.mask_seed_val   if args.mask_seed_val   is not None else int(seed) + 200
    fixed_mask_train = build_fixed_obs_mask(coords, miss_ratio, args.miss_mode, args.grid_stride, nx, ny, Lx, Ly, mask_seed_tr, device)
    fixed_mask_val   = build_fixed_obs_mask(coords, miss_ratio, args.miss_mode, args.grid_stride, nx, ny, Lx, Ly, mask_seed_va, device)

    # channel scales for weighted loss
    if args.use_weighted_loss:
        sigma_q, sigma_p = estimate_channel_std(train_loader, device)
    else:
        sigma_q = sigma_p = 1.0

    # Build common evaluation Hodge (theory) for fair energy-based metrics
    eval_hodge = HodgeBlockTheory(V0.to(device), V1inv.to(device), c_speed=args.c_speed, use_speed_scalar=False).to(device)

    # Build geometric edge features once
    eattr = build_edge_attr(coords, src, dst).to(device)  # [E,3]
    node_extras = V0.unsqueeze(-1).to(device)             # [N,1]  <-- add V0 as node feature
    mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e = compute_input_norm_stats(coords, node_extras, eattr)

    # ---------------- MeshFT-Net ----------------
    use_speed_scalar_meshft = bool(args.meshft_use_speed_scalar)
    meshft_hodge, _ = make_hodge(
        args.meshft_hodge_mode, V0, V1inv,
        use_speed_scalar=use_speed_scalar_meshft,
        w_structure=args.meshft_w_structure,
        offdiag_init=args.offdiag_init,
        normalize=bool(args.normalize_hodge),
        c_speed=args.c_speed,
        # pass geometry if learn_geom
        coords=coords, src=src, dst=dst,
        geom_hidden=args.meshft_geom_hidden, geom_layers=args.meshft_geom_layers,
        geom_use_sn=bool(args.use_spectral_norm)
    )
    meshft_hodge = meshft_hodge.to(device)
    meshft_net = MeshFTNet(src, dst, meshft_hodge, state_mode=args.state_mode).to(device)
    meshft_net._target_c2 = float(args.c_wave**2)

    # Decide whether to use guarded calibration based on use_speed_scalar_meshft
    # - If True: allow guard to downscale c^2 to satisfy CFL and freeze log_speed2.
    # - If False: keep behavior without global rescaling (guard=False).
    _guard_flag = use_speed_scalar_meshft

    # 1) Initial calibration (before training) with branching on guard
    _prepare_hodge_for_dt(
        meshft_net.hodge, src, dst, args.dt,
        target_c2=args.c_wave**2,
        guard=_guard_flag
    )

    # Freeze the global speed scalar during training to avoid train/eval mismatch
    if use_speed_scalar_meshft and getattr(meshft_net.hodge, "log_speed2", None) is not None:
        meshft_net.hodge.log_speed2.requires_grad_(False)

    # 2) Compute substeps AFTER (possibly guarded) calibration
    try:
        omega = _estimate_omega_max(src, dst, meshft_net.hodge, iters=25)
        meshft_net._nsub = int(math.ceil(max(1.0, (omega * args.dt) / 1.0)))
        print(f"[MeshFT-Net:init] omega_max≈{omega:.3e}, dt={args.dt:.3e} -> substeps={meshft_net._nsub}")
    except Exception as e:
        meshft_net._nsub = 1
        print(f"[MeshFT-Net:init] omega_max estimation failed ({e}); using substeps=1.")

    meshft_params = list(meshft_net.parameters())
    meshft_net._opt = make_optimizer_if_any(meshft_params, lr=1e-3, wd=1e-6)

    for _ in range(args.epochs if meshft_net._opt is not None else 0):
        _ = train_one_epoch_meshft(
            meshft_net, train_loader, args.dt, device, nx, ny, coords, Lx, Ly,
            miss_ratio, args.miss_mode, args.grid_stride, args.progress=="bars",
            use_weighted_loss=bool(args.use_weighted_loss), sigma_q=sigma_q, sigma_p=sigma_p,
            fixed_mask_2c=fixed_mask_train
        )

    meshft_val_mse, meshft_val_psnr = eval_one_epoch_meshft(
        meshft_net, val_loader, args.dt, device, args.progress=="bars",
        fixed_mask_2c=(fixed_mask_val if args.mask_apply_to_val else None),
        apply_mask_to_metrics=bool(args.mask_apply_to_val)
    )

    # meta for rollout (unchanged)
    meta_batch = {
        "t": torch.tensor([m["t"] for _,_,m in [val_ds[i] for i in range(min(8, len(val_ds)))]], device=device),
        "kvec": torch.stack([val_ds[i][2]["kvec"] for i in range(min(8, len(val_ds)))], dim=0).to(device),
        "omega": [val_ds[i][2]["omega"] for i in range(min(8, len(val_ds)))],
        "phi":   [val_ds[i][2]["phi"]   for i in range(min(8, len(val_ds)))],
        "amp":   [val_ds[i][2]["amp"]   for i in range(min(8, len(val_ds)))],
    }
    
    meshft_relF, meshft_relM, meshft_drift, meshft_steps, meshft_E = rollout_eval(
        "meshft_net", meshft_net, src, dst, coords, args.dt, meta_batch, args.rollout_T,
        device, eval_hodge=eval_hodge, M_data=V0.clone().to(device),
        return_energy_series=True
    )
    meshft_drift_slope = drift_slope_from_series(meshft_E, args.dt)
    if args.save_energy_csv:
        tag = f"meshft_mesh-{args.mesh}_grid-{nx}x{ny}_seed-{seed}_tr{train_size}_mr-{miss_ratio:.2f}_dt-{args.dt}_T-{args.rollout_T}"
        save_energy_series_csv(meshft_E, args.dt, args.energy_csv_dir, f"{tag}.csv")

    # ---------------- MGN (baseline) ----------------
    mgn_net = MeshGraphNetVF(in_dim=5, hidden=args.mgn_hidden, layers=args.mgn_layers, out_dim=2,
                             edge_in_dim=eattr.shape[1], use_sn=bool(args.use_spectral_norm), use_input_norm=bool(args.std_inputs or args.std_state)).to(device)

    sigma_q_used = sigma_q if args.std_state else 1.0
    sigma_p_used = sigma_p if args.std_state else 1.0
    mgn_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                            sigma_q=sigma_q_used, sigma_s=sigma_p_used)

    mgn = {"net": mgn_net, "coords": coords, "eattr": eattr, "state_mode": args.state_mode,
           "node_extras": node_extras}
    mgn["integrator"] = args.mgn_integrator
    mgn["_opt"] = make_optimizer_if_any(list(mgn_net.parameters()), lr=1e-3, wd=1e-6)

    # ---- Common CFL gate alpha from theory omega_max (no step splitting) ----

    for ep in range(args.epochs if mgn["_opt"] is not None else 0):
        _ = train_one_epoch_mgn(
            mgn, train_loader, args.dt, device, nx, ny, coords, Lx, Ly,
            lam_ham=0.0, src=src, dst=dst,
            miss_ratio=miss_ratio, miss_mode=args.miss_mode, grid_stride=args.grid_stride,
            use_bar=(args.progress=="bars"),
            use_weighted_loss=bool(args.use_weighted_loss), sigma_q=sigma_q, sigma_p=sigma_p,
            fixed_mask_2c=fixed_mask_train
        )
    mgn_val_mse, mgn_val_psnr = eval_one_epoch_mgn(mgn, val_loader, args.dt, device, src=src, dst=dst, use_bar=(args.progress=="bars"),
                                                   fixed_mask_2c=(fixed_mask_val if args.mask_apply_to_val else None),
                                                   apply_mask_to_metrics=bool(args.mask_apply_to_val))
    
    mgn_relF, mgn_relM, mgn_drift, mgn_steps, mgn_E = rollout_eval(
        "mgn", mgn, src, dst, coords, args.dt, meta_batch, args.rollout_T,
        device, eval_hodge=eval_hodge, M_data=M_data, return_energy_series=True
    )
    mgn_drift_slope = drift_slope_from_series(mgn_E, args.dt)
    if args.save_energy_csv:
        tag = f"mgn_mesh-{args.mesh}_grid-{nx}x{ny}_seed-{seed}_tr{train_size}_mr-{miss_ratio:.2f}_dt-{args.dt}_T-{args.rollout_T}"
        save_energy_series_csv(mgn_E, args.dt, args.energy_csv_dir, f"{tag}.csv")

    # ---------------- MGN-HP (Hamiltonian penalty) ----------------
    mgnhp_net  = MeshGraphNetVF(in_dim=5, hidden=args.mgn_hidden, layers=args.mgn_layers, out_dim=2,
                                edge_in_dim=eattr.shape[1], use_sn=bool(args.use_spectral_norm), use_input_norm=bool(args.std_inputs or args.std_state)).to(device)
    mgnhp_enet = EnergyNet(node_in_dim=5, edge_in_dim=eattr.shape[1],
                           hidden=args.mgn_hidden, layers=args.mgn_layers, use_sn=bool(args.use_spectral_norm), use_input_norm=bool(args.std_inputs or args.std_state)).to(device)

    mgnhp_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                sigma_q=sigma_q_used, sigma_s=sigma_p_used)
    mgnhp_enet.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e,
                                sigma_q=sigma_q_used, sigma_s=sigma_p_used)

    mgnhp = {"net": mgnhp_net, "energy_net": mgnhp_enet, "coords": coords, "eattr": eattr,
             "state_mode": args.state_mode, "node_extras": node_extras}
    mgnhp["integrator"] = args.mgnhp_integrator
    mgnhp_params = list(mgnhp_net.parameters()) + list(mgnhp_enet.parameters())
    mgnhp["_opt"] = make_optimizer_if_any(mgnhp_params, lr=1e-3, wd=1e-6)
    lam = max(args.lam_ham, 1e-2) if args.lam_ham > 0 else 0.0

    for ep in range(args.epochs if mgnhp["_opt"] is not None else 0):
        lam_eff = lam * (min(1.0, (ep+1)/max(1,args.lam_warmup_epochs)) if lam>0 else 0.0)
        _ = train_one_epoch_mgn(
            mgnhp, train_loader, args.dt, device, nx, ny, coords, Lx, Ly,
            lam_ham=lam_eff, src=src, dst=dst,
            miss_ratio=miss_ratio, miss_mode=args.miss_mode, grid_stride=args.grid_stride,
            use_bar=(args.progress=="bars"),
            use_weighted_loss=bool(args.use_weighted_loss), sigma_q=sigma_q, sigma_p=sigma_p,
            fixed_mask_2c=fixed_mask_train,
            eval_hodge=eval_hodge, n_nodes=coords.shape[0], state_mode=args.state_mode,
            ham_energy_consistency=bool(args.ham_energy_consistency), ham_energy_cons_w=float(args.ham_energy_cons_w)
        )
    mgnhp_val_mse, mgnhp_val_psnr = eval_one_epoch_mgn(mgnhp, val_loader, args.dt, device, src=src, dst=dst, use_bar=(args.progress=="bars"),
                                                       fixed_mask_2c=(fixed_mask_val if args.mask_apply_to_val else None),
                                                       apply_mask_to_metrics=bool(args.mask_apply_to_val))
    
    mgnhp_relF, mgnhp_relM, mgnhp_drift, mgnhp_steps, mgnhp_E = rollout_eval(
        "mgn", mgnhp, src, dst, coords, args.dt, meta_batch, args.rollout_T,
        device, eval_hodge=eval_hodge, M_data=M_data, return_energy_series=True
    )
    mgnhp_drift_slope = drift_slope_from_series(mgnhp_E, args.dt)
    if args.save_energy_csv:
        tag = f"mgnhp_mesh-{args.mesh}_grid-{nx}x{ny}_seed-{seed}_tr{train_size}_mr-{miss_ratio:.2f}_dt-{args.dt}_T-{args.rollout_T}"
        save_energy_series_csv(mgnhp_E, args.dt, args.energy_csv_dir, f"{tag}.csv")

    # ---------------- PURE HNN (separable, canonical, symplectic leapfrog) ----------------
    if int(args.hnn_enable) == 1 and args.state_mode == "canonical":
        U_net = _SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                     edge_in_dim=eattr.shape[1], use_sn=bool(args.use_spectral_norm), use_input_norm=bool(args.std_inputs or args.std_state)).to(device)
        T_net = _SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                     edge_in_dim=eattr.shape[1], use_sn=bool(args.use_spectral_norm), use_input_norm=bool(args.std_inputs or args.std_state)).to(device)

        U_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e, sigma_field=(sigma_q if args.std_state else 1.0))
        T_net.set_normalization(mu_xy, std_xy, mu_ex, std_ex, mu_e, std_e, sigma_field=(sigma_p if args.std_state else 1.0))

        hnn = HNNSeparableSymplectic(U_net, T_net, coords, src, dst, eattr=eattr, node_extras=node_extras).to(device)
        hnn.integrator = args.hnn_integrator
        hnn._opt = make_optimizer_if_any(list(U_net.parameters()) + list(T_net.parameters()), lr=1e-3, wd=1e-6)

        try:
            omega_h = _estimate_omega_max(src, dst, eval_hodge, iters=25)
            hnn._nsub = int(math.ceil(max(1.0, omega_h * args.dt))) 
            print(f"[HNN:init] omega_max≈{omega_h:.3e}, dt={args.dt:.3e} -> substeps={hnn._nsub}")
        except Exception as e:
            hnn._nsub = 1
            print(f"[HNN:init] omega_max estimation failed ({e}); using substeps=1.")

        for _ in range(args.epochs if hnn._opt is not None else 0):
            _ = train_one_epoch_hnn(
                hnn, train_loader, args.dt, device, nx, ny, coords, Lx, Ly,
                miss_ratio, args.miss_mode, args.grid_stride, args.progress=="bars",
                use_weighted_loss=bool(args.use_weighted_loss), sigma_q=sigma_q, sigma_p=sigma_p,
                fixed_mask_2c=fixed_mask_train
            )

        hnn_val_mse, hnn_val_psnr = eval_one_epoch_hnn(
            hnn, val_loader, args.dt, device, use_bar=(args.progress=="bars"),
            fixed_mask_2c=(fixed_mask_val if args.mask_apply_to_val else None),
            apply_mask_to_metrics=bool(args.mask_apply_to_val)
        )

        
        hnn_relF, hnn_relM, hnn_drift, hnn_steps, hnn_E = rollout_eval(
            "hnn", hnn, src, dst, coords, args.dt, meta_batch, args.rollout_T,
            device, eval_hodge=eval_hodge, M_data=V0.clone().to(device), return_energy_series=True
        )
        hnn_drift_slope = drift_slope_from_series(hnn_E, args.dt)
        if args.save_energy_csv:
            tag = f"hnn_mesh-{args.mesh}_grid-{nx}x{ny}_seed-{seed}_tr{train_size}_mr-{miss_ratio:.2f}_dt-{args.dt}_T-{args.rollout_T}"
            save_energy_series_csv(hnn_E, args.dt, args.energy_csv_dir, f"{tag}.csv")

    else:
        hnn_val_mse = hnn_val_psnr = hnn_relF = hnn_relM = hnn_drift = float("nan")
        hnn_steps = 0

    # --- build initial batch x0 once for Π ---
    B0 = len(meta_batch["t"])
    x0_for_pi = []
    for i in range(B0):
        q0, v0 = plane_wave_q_and_v(coords, float(meta_batch["t"][i]),
                                    meta_batch["kvec"][i].cpu().numpy(),
                                    float(meta_batch["omega"][i]),
                                    float(meta_batch["phi"][i]),
                                    float(meta_batch["amp"][i]), device)
        s0 = (V0 * v0) if args.state_mode == "canonical" else v0
        x0_for_pi.append(torch.stack([q0, s0], dim=-1))
    x0_for_pi = torch.stack(x0_for_pi, dim=0)

    pi_meshft   = energy_power_violation_pi(eval_hodge, src, dst, x0_for_pi, "meshft_net",   meshft_net,   coords, args.dt)
    pi_mgn   = energy_power_violation_pi(eval_hodge, src, dst, x0_for_pi, "mgn",   mgn,   coords, args.dt, eattr=eattr)
    pi_mgnhp = energy_power_violation_pi(eval_hodge, src, dst, x0_for_pi, "mgnhp", mgnhp, coords, args.dt, eattr=eattr)
    if int(args.hnn_enable) == 1 and args.state_mode == "canonical":
        pi_hnn = energy_power_violation_pi(eval_hodge, src, dst, x0_for_pi, "hnn", hnn, coords, args.dt, eattr=eattr)
    else:
        pi_hnn = float("nan")

    # --- observed order wrt dt (shared gate alpha) ---
    ord_meshft   = observed_order_dt("meshft_net",   meshft_net,   coords, src, dst, eval_hodge, meta_batch, args.dt, eattr=eattr, alpha_shared=1.0, state_mode=args.state_mode)
    ord_mgn   = observed_order_dt("mgn",   mgn,   coords, src, dst, eval_hodge, meta_batch, args.dt, eattr=eattr, alpha_shared=1.0, state_mode=args.state_mode)
    ord_mgnhp = observed_order_dt("mgnhp", mgnhp, coords, src, dst, eval_hodge, meta_batch, args.dt, eattr=eattr, alpha_shared=1.0, state_mode=args.state_mode)
    ord_hnn   = observed_order_dt("hnn",   hnn,   coords, src, dst, eval_hodge, meta_batch, args.dt, eattr=eattr, alpha_shared=1.0, state_mode=args.state_mode) \
                if int(args.hnn_enable)==1 and args.state_mode=="canonical" else float("nan")

    row = dict(
        seed=seed, mesh=args.mesh, grid=f"{nx}x{ny}", npoints=(coords.shape[0] if args.mesh=="delaunay" else nx*ny),
        dt=args.dt, epochs=args.epochs, batch=args.batch_size,
        train_size=train_size, val_size=args.val_size, rollout_T=args.rollout_T,
        std_inputs=int(args.std_inputs),
        std_state=int(args.std_state),
        miss_ratio=miss_ratio, miss_mode=args.miss_mode, grid_stride=args.grid_stride,
        use_weighted_loss=int(args.use_weighted_loss),
        state_mode=args.state_mode, data_state_mode=args.data_state_mode,
        meshft_hodge_mode=args.meshft_hodge_mode,
        meshft_w_structure=args.meshft_w_structure,
        offdiag_init=args.offdiag_init, normalize_hodge=int(args.normalize_hodge),
        c_speed=args.c_speed, c_wave=args.c_wave,
        meshft_nsub=int(getattr(meshft_net, "_nsub", 1)),
        mask_seed_train=mask_seed_tr, mask_seed_val=mask_seed_va, mask_apply_to_val=int(args.mask_apply_to_val),
        sigma_q=sigma_q, sigma_p=sigma_p,
        meshft_mse=meshft_val_mse, meshft_psnr=meshft_val_psnr, meshft_relF=meshft_relF, meshft_relM=meshft_relM, meshft_drift=meshft_drift, meshft_steps=meshft_steps,
        mgn_mse=mgn_val_mse, mgn_psnr=mgn_val_psnr, mgn_relF=mgn_relF, mgn_relM=mgn_relM, mgn_drift=mgn_drift, mgn_steps=mgn_steps,
        mgnhp_mse=mgnhp_val_mse, mgnhp_psnr=mgnhp_val_psnr, mgnhp_relF=mgnhp_relF, mgnhp_relM=mgnhp_relM, mgnhp_drift=mgnhp_drift, mgnhp_steps=mgnhp_steps,
        hnn_mse=hnn_val_mse, hnn_psnr=hnn_val_psnr,
        hnn_relF=hnn_relF, hnn_relM=hnn_relM, hnn_drift=hnn_drift, hnn_steps=hnn_steps,
        lam_ham=lam,
        # energy production rate (Π)
        meshft_pi=pi_meshft, mgn_pi=pi_mgn, mgnhp_pi=pi_mgnhp, hnn_pi=pi_hnn,
        # drift slope
        meshft_drift_slope=meshft_drift_slope, mgn_drift_slope=mgn_drift_slope,
        mgnhp_drift_slope=mgnhp_drift_slope, hnn_drift_slope=hnn_drift_slope if (int(args.hnn_enable)==1 and args.state_mode=="canonical") else float("nan"),
        # observed order p
        meshft_order=ord_meshft, mgn_order=ord_mgn, mgnhp_order=ord_mgnhp, hnn_order=ord_hnn,
    )
    return row

# ------------------------- plotting helpers -------------------------

def _aggregate(rows: List[dict], xkey: str, ykey_map: Dict[str,str], cond: Dict[str,object]):
    def ok(r):
        for k,v in cond.items():
            if k==xkey: continue
            if r.get(k) != v: return False
        return True
    filt = [r for r in rows if ok(r)]
    xs_sorted = sorted(sorted({r[xkey] for r in filt}), key=lambda z: float(z))
    res = {lab: ([],[]) for lab in ykey_map.keys()}
    for x in xs_sorted:
        bucket = [r for r in filt if float(r[xkey])==float(x)]
        for lab, ykey in ykey_map.items():
            vals = [float(r[ykey]) for r in bucket]
            mu = float(np.mean(vals)) if len(vals)>0 else float("nan")
            sd = float(np.std(vals))  if len(vals)>0 else float("nan")
            res[lab][0].append(mu); res[lab][1].append(sd)
    return xs_sorted, res

def plot_curves(xs, curves, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(4.0,3.0))
    markers = {"MeshFT-Net":"o","MGN":"s","MGN-HP":"^"}
    for lab, (mu, sd) in curves.items():
        plt.errorbar(xs, mu, yerr=sd, marker=markers.get(lab,"o"), linewidth=1.5, capsize=2.5, label=lab)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.2); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(outpath); plt.close()

# ------------------------- main -------------------------

def parse_list(s: str, typ=float):
    if s is None or s.strip()=="":
        return []
    return [typ(x) for x in s.split(",")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs/analytic_bench")
    ap.add_argument("--out_csv", type=str, default="runs/analytic_bench/results.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Mesh options
    ap.add_argument("--mesh", type=str, default="delaunay", choices=["grid","delaunay"])
    ap.add_argument("--grid", type=int, nargs=2, default=[32, 32], help="for grid mesh or binning in masking")
    ap.add_argument("--npoints", type=int, default=None, help="only for mesh=delaunay; default uses prod(grid)")
    ap.add_argument("--Lx", type=float, default=1.0); ap.add_argument("--Ly", type=float, default=1.0)

    # Dynamics / training
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--train_size", type=int, default=2000)
    ap.add_argument("--val_size", type=int, default=256)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--progress", type=str, default="bars", choices=["bars","none"])

    ap.add_argument("--std_inputs", type=int, default=1,
                    help="standardize geometry features: coords/V0/edge_attr (0/1)")
    ap.add_argument("--std_state", type=int, default=1,
                    help="standardize state channels q,s by dataset std (0/1)")

    # Wave speeds
    ap.add_argument("--c_speed", type=float, default=1.0, help="sets W=(c_speed^2)*V1inv in theory Hodge")
    ap.add_argument("--c_wave", type=float, default=None, help="analytic plane-wave speed; default matches c_speed")

    # State conventions
    ap.add_argument("--state_mode", type=str, default="canonical", choices=["canonical","velocity"],
                    help="how models interpret channel-1")
    ap.add_argument("--data_state_mode", type=str, default="canonical", choices=["canonical","velocity"],
                    help="how dataset constructs channel-1 (default canonical p=M dq/dt)")

    # MGN hyperparams
    ap.add_argument("--mgn_hidden", type=int, default=64)
    ap.add_argument("--mgn_layers", type=int, default=4)

    # Hamiltonian penalty (EnergyNet-based; no Hodge)
    ap.add_argument("--lam_ham", type=float, default=0.0)

    # Long rollout
    ap.add_argument("--rollout_T", type=int, default=200)
    # ---- Stabilization knobs (no integrator replacement) ----
    ap.add_argument("--use_spectral_norm", type=int, default=0, help="Apply spectral_norm to all Linear layers (0/1)")
    # ---- HP-specific stabilizers ----
    ap.add_argument("--lam_warmup_epochs", type=int, default=3, help="warm up ham penalty over first K epochs")
    ap.add_argument("--ham_energy_consistency", type=int, default=1, help="add theory energy consistency penalty (0/1)")
    ap.add_argument("--ham_energy_cons_w", type=float, default=0.1, help="weight of energy consistency (multiplied by lam_ham)")

    # Missingness / sweeps
    ap.add_argument("--miss_ratio", type=float, default=0.0, help="0..1")
    ap.add_argument("--miss_mode", type=str, default="random", choices=["random","grid"])
    ap.add_argument("--grid_stride", type=int, default=2)
    ap.add_argument("--sweep_train_sizes", type=str, default="")
    ap.add_argument("--sweep_miss_ratios", type=str, default="")
    ap.add_argument("--make_plots", action="store_true")
    ap.add_argument("--plot_ext", type=str, default="pdf", choices=["pdf","svg","png"])

    # fairness / Hodge options for MeshFT-Net
    ap.add_argument("--use_weighted_loss", type=int, default=0)
    ap.add_argument("--normalize_hodge", type=int, default=0, help="normalize M/W by mean; OFF for canonical recommended")
    ap.add_argument("--meshft_hodge_mode", type=str, default="learn_geom", choices=["learn","learn_geom","theory"])
    ap.add_argument("--meshft_geom_hidden", type=int, default=64)
    ap.add_argument("--meshft_geom_layers", type=int, default=2)
    ap.add_argument("--meshft_use_speed_scalar", type=int, default=0)
    ap.add_argument("--meshft_w_structure", type=str, default="diag", choices=["diag","offdiag"])
    ap.add_argument("--offdiag_init", type=float, default=-6.0)

    # FIXED mask options
    ap.add_argument("--mask_seed_train", type=int, default=None, help="fixed random seed for train mask (random mode)")
    ap.add_argument("--mask_seed_val",   type=int, default=None, help="fixed random seed for val mask (random mode)")
    ap.add_argument("--mask_apply_to_val", type=int, default=1, help="apply mask to val metrics (1=yes)")

    # --- Energy trace CSV options ---
    ap.add_argument("--save_energy_csv", type=int, default=0,
                    help="Save per-step energy time series to CSV for each model (0/1)")
    ap.add_argument("--energy_csv_dir", type=str, default=None,
                    help="Directory for energy time-series CSVs; defaults to <out_dir>/energy_traces")

    # HNN-like MGN
    ap.add_argument("--hnn_enable", type=int, default=1, help="Enable pure HNN (separable) branch (0/1)")
    ap.add_argument("--hnn_hidden", type=int, default=64)
    ap.add_argument("--hnn_layers", type=int, default=4)

    # --- Integrator options (unify to KDK by default) ---
    ap.add_argument("--mgn_integrator", type=str, default="kdk",
                    choices=["euler", "rk2", "kdk"],
                    help="MGN/MGN-HP integrator")
    ap.add_argument("--mgnhp_integrator", type=str, default="kdk",
                    choices=["euler", "rk2", "kdk"],
                    help="MGN-HP integrator")
    ap.add_argument("--hnn_integrator", type=str, default="kdk",
                    choices=["se", "kdk"],  # 'kdk' == Störmer–Verlet
                    help="HNN integrator ('se'=symplectic-Euler, 'kdk'=Störmer–Verlet)")
    

    args = ap.parse_args()

    if args.energy_csv_dir is None:
        args.energy_csv_dir = os.path.join(args.out_dir, "energy_traces")

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = args.seeds if (args.seeds is not None and len(args.seeds)>0) else [args.seed]

    device = args.device
    nx, ny = args.grid
    Lx, Ly = args.Lx, args.Ly
    if args.c_wave is None:
        args.c_wave = float(args.c_speed)  # align analytic wave speed with theory W scaling by default

    # ----- build mesh -----
    if args.mesh == "grid":
        coords, src, dst, V0, elen = build_periodic_grid(nx, ny, Lx, Ly)
        V1inv = torch.ones_like(elen)
    else:
        npts = args.npoints if (args.npoints and args.npoints>0) else int(nx*ny)
        coords, src, dst, V0, elen, simplices = build_delaunay_mesh(npts, Lx, Ly, seed=args.seed)
        V1inv = cotangent_W_from_tris(coords, src, dst, simplices)

    coords = coords.to(device); src = src.to(device); dst = dst.to(device)
    V0 = V0.to(device); V1inv = V1inv.to(device)

    list_train = parse_list(args.sweep_train_sizes, int) or [args.train_size]
    list_miss  = parse_list(args.sweep_miss_ratios, float) or [args.miss_ratio]

    rows = []
    for seed, trsz, mr in itertools.product(seeds, list_train, list_miss):
        print(f"\n=== RUN mesh={args.mesh} seed={seed} train_size={trsz} miss_ratio={mr} mode={args.miss_mode} (state={args.state_mode}, data={args.data_state_mode}) ===")
        row = run_one_config(args, nx, ny, coords, src, dst, V0, V1inv, trsz, mr, seed, Lx, Ly)
        rows.append(row)
        is_new = not os.path.exists(args.out_csv)
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if is_new: w.writerow(list(row.keys()))
            w.writerow([row[k] for k in row.keys()])

    if args.make_plots:
        with open(os.path.join(args.out_dir, "sweep_raw.json"), "w") as f:
            json.dump(rows, f, indent=2)

        uniq_mr = sorted({float(r["miss_ratio"]) for r in rows})
        for mr in uniq_mr:
            xs, curves = _aggregate(rows, xkey="train_size",
                                    ykey_map={"MeshFT-Net":"meshft_mse", "MGN":"mgn_mse", "MGN-HP":"mgnhp_mse"},
                                    cond={"miss_ratio":mr, "grid":f"{nx}x{ny}"})
            outp = os.path.join(args.out_dir, f"mse_vs_trainsize_mr{mr:.2f}.{args.plot_ext}")
            plot_curves(xs, curves, xlabel="Train size (#pairs)", ylabel="Validation MSE",
                        title=f"Accuracy vs data size  (missing={mr:.2f}, {args.miss_mode}, mesh={args.mesh})", outpath=outp)

            xs, curves = _aggregate(rows, xkey="train_size",
                                    ykey_map={"MeshFT-Net":"meshft_psnr", "MGN":"mgn_psnr", "MGN-HP":"mgnhp_psnr"},
                                    cond={"miss_ratio":mr, "grid":f"{nx}x{ny}"})
            outp = os.path.join(args.out_dir, f"psnr_vs_trainsize_mr{mr:.2f}.{args.plot_ext}")
            plot_curves(xs, curves, xlabel="Train size (#pairs)", ylabel="Validation PSNR (dB)",
                        title=f"PSNR vs data size  (missing={mr:.2f}, {args.miss_mode}, mesh={args.mesh})", outpath=outp)

        uniq_tr = sorted({int(r["train_size"]) for r in rows})
        for tr in uniq_tr:
            xs, curves = _aggregate(rows, xkey="miss_ratio",
                                    ykey_map={"MeshFT-Net":"meshft_mse", "MGN":"mgn_mse", "MGN-HP":"mgnhp_mse"},
                                    cond={"train_size":tr, "grid":f"{nx}x{ny}"})
            outp = os.path.join(args.out_dir, f"mse_vs_missing_tr{tr}.{args.plot_ext}")
            plot_curves(xs, curves, xlabel="Missing rate (fraction)", ylabel="Validation MSE",
                        title=f"Accuracy vs missing (train={tr}, mesh={args.mesh})", outpath=outp)

            xs, curves = _aggregate(rows, xkey="miss_ratio",
                                    ykey_map={"MeshFT-Net":"meshft_psnr", "MGN":"mgn_psnr", "MGN-HP":"mgnhp_psnr"},
                                    cond={"train_size":tr, "grid":f"{nx}x{ny}"})
            outp = os.path.join(args.out_dir, f"psnr_vs_missing_tr{tr}.{args.plot_ext}")
            plot_curves(xs, curves, xlabel="Missing rate (fraction)", ylabel="Validation PSNR (dB)",
                        title=f"PSNR vs missing (train={tr}, mesh={args.mesh})", outpath=outp)

        print(f"[plots] saved to {args.out_dir}")

    last = rows[-1]
    print("=== Last-run Summary ===")
    print(f"(mesh={last['mesh']} seed={last['seed']} train={last['train_size']} miss={last['miss_ratio']})")
    print(f"MeshFT-Net : MSE={last['meshft_mse']:.3e} PSNR={last['meshft_psnr']:.2f}  relF={last['meshft_relF']:.3e} med={last['meshft_relM']:.3e} drift={last['meshft_drift']:.3e}")
    print(f"MGN    : MSE={last['mgn_mse']:.3e} PSNR={last['mgn_psnr']:.2f}  relF={last['mgn_relF']:.3e} med={last['mgn_relM']:.3e} drift={last['mgn_drift']:.3e}")
    print(f"MGN-HP : MSE={last['mgnhp_mse']:.3e} PSNR={last['mgnhp_psnr']:.2f} relF={last['mgnhp_relF']:.3e} med={last['mgnhp_relM']:.3e} drift={last['mgnhp_drift']:.3e} (lam={last['lam_ham']})")
    print("HNN    : " f"MSE={last.get('hnn_mse', float('nan')):.3e} " f"PSNR={last.get('hnn_psnr', float('nan')):.2f}  " f"relF={last.get('hnn_relF', float('nan')):.3e} "
        f"med={last.get('hnn_relM', float('nan')):.3e} " f"drift={last.get('hnn_drift', float('nan')):.3e}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()