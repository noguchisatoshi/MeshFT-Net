#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute-cost benchmark for MeshFT-Net (DEC), MGN, MGN-HP, HNN.
Measures per-step inference & training times, peak memory, and param counts
under the same synthetic plane-wave setup as the original script.

Usage examples:
  python cost_bench.py --ab-module analytic_bench.py
  python cost_bench.py --device cuda --mesh grid --grid 32 32 --iters 100 --warmup 20
"""
import argparse, importlib.util, os, sys, time, statistics, math, csv
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# ------------------------- dynamic import of the original file -------------------------
def _import_ab(module_path_or_name: str):
    # If user passed a filename like analytic_bench.py, import from path
    if module_path_or_name.endswith(".py") and os.path.exists(module_path_or_name):
        spec = importlib.util.spec_from_file_location("ab", module_path_or_name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ab"] = mod
        spec.loader.exec_module(mod)
        return mod
    # Else treat as module name (without .py)
    name = module_path_or_name.replace(".py", "")
    if name in sys.modules:
        return sys.modules[name]
    return __import__(name, fromlist=["*"])

# ------------------------- tiny utils -------------------------
def _count_params(obj) -> int:
    if isinstance(obj, dict):
        tot = 0
        for v in obj.values():
            if hasattr(v, "parameters"):
                tot += sum(p.numel() for p in v.parameters() if p.requires_grad)
        return tot
    return sum(p.numel() for p in obj.parameters() if p.requires_grad)

def _synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def _peak_mem_mb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return float("nan")

def _reset_peak_mem(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _now():
    return time.perf_counter()

# ------------------------- benchmarking kernels -------------------------
@torch.no_grad()
def _bench_infer_step(model_kind: str, model, batch, dt, device) -> None:
    x0, _x1, _ = batch
    x0 = x0.to(device)
    if model_kind == "meshft":
        _ = model(x0, dt)
    elif model_kind == "hnn":
        # HNN forward builds grads internally; keep grad disabled for pure inference
        with torch.enable_grad():
            out = model(x0, dt)
        out = out.detach()
    else:  # mgn / mgnhp dict
        net = model["net"]
        coords = model["coords"]
        src = model.get("src"); dst = model.get("dst")
        eattr = model.get("eattr"); node_extras = model.get("node_extras")
        v = net(x0, coords, src, dst, dt, eattr=eattr, node_extras=node_extras)
        alpha = float(model.get("gate_alpha", 1.0))
        _ = x0 + dt * alpha * v

def _masked_weighted_loss(ab, x_pred, x1, mask_2c, sigma_q, sigma_p):
    if mask_2c is None:
        return F.mse_loss(x_pred, x1)
    maskB = mask_2c.unsqueeze(0).expand_as(x_pred)
    return ab.masked_weighted_mse(x_pred, x1, maskB, sigma_q, sigma_p)

def _bench_train_step(ab, model_kind: str, model, opt, batch, dt, device,
                      mask_2c, sigma_q, sigma_p, lam_ham: float = 0.0) -> None:
    x0, x1, _ = batch
    x0 = x0.to(device); x1 = x1.to(device)

    if model_kind == "meshft":
        pred = model(x0, dt)
        loss = _masked_weighted_loss(ab, pred, x1, mask_2c, sigma_q, sigma_p)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return

    if model_kind == "hnn":
        with torch.enable_grad():
            pred = model(x0, dt)
            loss = _masked_weighted_loss(ab, pred, x1, mask_2c, sigma_q, sigma_p)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return

    # MGN family (dict)
    net = model["net"]
    coords = model["coords"]; src = model["src"]; dst = model["dst"]
    eattr = model.get("eattr"); node_extras = model.get("node_extras")
    alpha = float(model.get("gate_alpha", 1.0))
    v_pred = net(x0, coords, src, dst, dt, eattr=eattr, node_extras=node_extras)
    x_pred = x0 + dt * alpha * v_pred
    loss = _masked_weighted_loss(ab, x_pred, x1, mask_2c, sigma_q, sigma_p)

    # HP penalty (if mgnhp)
    if lam_ham > 0.0 and ("energy_net" in model) and (model["energy_net"] is not None):
        enet = model["energy_net"]
        xreq = x0.detach().requires_grad_(True)
        H = enet(xreq, coords, src, dst, eattr=eattr, node_extras=node_extras)
        (g,) = torch.autograd.grad(H.sum(), xreq, create_graph=True)
        v_ham = ab.apply_J_to_grad(g, state_mode=model.get("state_mode", "canonical"))
        loss = loss + lam_ham * F.mse_loss(v_pred, v_ham)

    params = list(net.parameters()) + (list(model.get("energy_net").parameters()) if model.get("energy_net", None) is not None else [])
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params if len(params)>0 else [], 1.0)
    opt.step()

def _time_loop(fn, warmup, iters, device) -> Tuple[float,float,float]:
    times = []
    # warmup
    for _ in range(warmup):
        fn()
    _synchronize(device)
    # measure
    for _ in range(iters):
        t0 = _now()
        fn()
        _synchronize(device)
        t1 = _now()
        times.append((t1 - t0) * 1000.0)  # ms
    mu = statistics.mean(times)
    sd = statistics.pstdev(times) if len(times)>1 else 0.0
    md = statistics.median(times)
    return mu, sd, md

# ------------------------- assemble the scenario -------------------------
def build_world(ab, args):
    ab.set_seed(args.seed)
    device = args.device

    # mesh
    nx, ny = args.grid
    if args.mesh == "grid":
        coords, src, dst, V0, elen = ab.build_periodic_grid(nx, ny, args.Lx, args.Ly)
        V1inv = torch.ones_like(elen)
    else:
        npts = args.npoints if (args.npoints and args.npoints>0) else int(nx*ny)
        coords, src, dst, V0, elen, simplices = ab.build_delaunay_mesh(npts, args.Lx, args.Ly, seed=args.seed)
        V1inv = ab.cotangent_W_from_tris(coords, src, dst, simplices)
    coords = coords.to(device); src = src.to(device); dst = dst.to(device)
    V0 = V0.to(device); V1inv = V1inv.to(device)

    # dataset / loader (canonical by default)
    c_wave = args.c_wave if args.c_wave is not None else args.c_speed
    ds = ab.PlaneWaveDataset(nx, ny, dt=args.dt, size=args.val_size, coords=coords,
                             M_data=V0.clone().to(device), Lx=args.Lx, Ly=args.Ly,
                             c_wave=c_wave, kmax=args.kmax, device=device, state_mode_data=args.data_state_mode)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # fixed mask for compute fairness
    mask_seed = args.mask_seed if args.mask_seed is not None else args.seed + 123
    mask_2c = ab.build_fixed_obs_mask(coords, args.miss_ratio, args.miss_mode,
                                      args.grid_stride, nx, ny, args.Lx, args.Ly, mask_seed, device)
    # channel std for weighted loss (same as main codeの流儀に合わせる)
    sigma_q, sigma_p = (1.0, 1.0)
    if args.use_weighted_loss:
        sigma_q, sigma_p = ab.estimate_channel_std(loader, device)

    # common edge/node feats
    eattr = ab.build_edge_attr(coords, src, dst).to(device)
    node_extras = V0.unsqueeze(-1).to(device)

    # common eval hodge
    eval_hodge = ab.HodgeBlockTheory(V0.to(device), V1inv.to(device), c_speed=args.c_speed, use_speed_scalar=False).to(device)

    world = dict(nx=nx, ny=ny, coords=coords, src=src, dst=dst, V0=V0, V1inv=V1inv,
                 loader=loader, mask_2c=mask_2c, sigma_q=sigma_q, sigma_p=sigma_p,
                 eattr=eattr, node_extras=node_extras, eval_hodge=eval_hodge, c_wave=c_wave)
    return world

def build_models(ab, args, world):
    device = args.device
    coords, src, dst, V0, V1inv = world["coords"], world["src"], world["dst"], world["V0"], world["V1inv"]
    eattr, node_extras = world["eattr"], world["node_extras"]

    # MeshFT-Net
    meshft_hodge, _ = ab.make_hodge(args.meshft_hodge_mode, V0, V1inv,
                                 use_speed_scalar=bool(args.meshft_use_speed_scalar),
                                 w_structure=args.meshft_w_structure,
                                 offdiag_init=args.offdiag_init,
                                 normalize=bool(args.normalize_hodge),
                                 c_speed=args.c_speed,
                                 coords=coords, src=src, dst=dst,
                                 geom_hidden=args.meshft_geom_hidden, geom_layers=args.meshft_geom_layers,
                                 geom_use_sn=bool(args.use_spectral_norm))
    meshft = ab.MeshFTNet(src, dst, meshft_hodge.to(device), state_mode=args.state_mode).to(device)
    ab._prepare_hodge_for_dt(meshft.hodge, src, dst, args.dt, target_c2=(world["c_wave"]**2), guard=bool(args.meshft_use_speed_scalar))
    try:
        omega = ab._estimate_omega_max(src, dst, meshft.hodge, iters=20)
        meshft._nsub = int(math.ceil(max(1.0, (omega * args.dt))))
    except Exception:
        meshft._nsub = 1
    meshft_opt = torch.optim.AdamW([p for p in meshft.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-6)

    # MGN
    mgn_net = ab.MeshGraphNetVF(in_dim=5, hidden=args.mgn_hidden, layers=args.mgn_layers, out_dim=2,
                                edge_in_dim=eattr.shape[1], use_sn=bool(args.use_spectral_norm)).to(device)
    mgn = {"net": mgn_net, "coords": coords, "eattr": eattr, "node_extras": node_extras,
           "src": src, "dst": dst, "state_mode": args.state_mode}
    # CFL gate from theory (optional)
    alpha_gate = 1.0
    if args.cfl_gate:
        try:
            omega_theory = ab._estimate_omega_max(src, dst, world["eval_hodge"], iters=15)
            alpha_gate = min(1.0, float(args.cfl_gate_safety) / max(1e-12, omega_theory * args.dt))
        except Exception:
            pass
    mgn["gate_alpha"] = alpha_gate
    mgn_opt = torch.optim.AdamW([p for p in mgn_net.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-6)

    # MGN-HP
    mgnhp_net  = ab.MeshGraphNetVF(in_dim=5, hidden=args.mgn_hidden, layers=args.mgn_layers, out_dim=2,
                                   edge_in_dim=eattr.shape[1], use_sn=bool(args.use_spectral_norm)).to(device)
    mgnhp_enet = ab.EnergyNet(node_in_dim=5, edge_in_dim=eattr.shape[1],
                              hidden=args.mgn_hidden, layers=args.mgn_layers,
                              use_sn=bool(args.use_spectral_norm)).to(device)
    mgnhp = {"net": mgnhp_net, "energy_net": mgnhp_enet, "coords": coords, "eattr": eattr,
             "node_extras": node_extras, "src": src, "dst": dst, "state_mode": args.state_mode,
             "gate_alpha": alpha_gate}
    mgnhp_opt = torch.optim.AdamW(
        [p for p in list(mgnhp_net.parameters()) + list(mgnhp_enet.parameters()) if p.requires_grad],
        lr=1e-3, weight_decay=1e-6
    )

    # HNN (only canonical makes sense here)
    hnn = None; hnn_opt = None
    if args.hnn_enable and args.state_mode == "canonical":
        U = ab._SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                    edge_in_dim=eattr.shape[1], use_sn=bool(args.use_spectral_norm)).to(device)
        T = ab._SeparableNodeEnergy(hidden=args.hnn_hidden, layers=args.hnn_layers,
                                    edge_in_dim=eattr.shape[1], use_sn=bool(args.use_spectral_norm)).to(device)
        hnn = ab.HNNSeparableSymplectic(U, T, coords, src, dst, eattr=eattr, node_extras=node_extras).to(device)
        hnn_opt = torch.optim.AdamW(
            [p for p in list(U.parameters()) + list(T.parameters()) if p.requires_grad],
            lr=1e-3, weight_decay=1e-6
        )

    return dict(meshft=(meshft, meshft_opt),
                mgn=(mgn, mgn_opt),
                mgnhp=(mgnhp, mgnhp_opt),
                hnn=(hnn, hnn_opt))

def run_one_bench(ab, args):
    world = build_world(ab, args)
    models = build_models(ab, args, world)

    # prepare a small iterator over data
    loader = world["loader"]; it = iter(loader)
    batches = []
    for _ in range(max(args.warmup + args.iters, 1)):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader); batches.append(next(it))

    def _pick(i): return batches[i % len(batches)]

    results = []
    for kind in ["meshft", "mgn", "mgnhp", "hnn"]:
        model, opt = models[kind]
        if model is None:
            continue
        # param count
        n_params = _count_params(model)

        # ---------- Inference ----------
        _reset_peak_mem(args.device)
        def infer_once(i=[0]):
            b = _pick(i[0]); i[0] += 1
            _bench_infer_step("hnn" if kind=="hnn" else ("meshft" if kind=="meshft" else "mgn"), model, b, args.dt, args.device)
        mu_i, sd_i, md_i = _time_loop(lambda: infer_once(), args.warmup, args.iters, args.device)
        peak_inf = _peak_mem_mb(args.device)

        # throughput estimates
        B = batches[0][0].shape[0]; N = world["coords"].shape[0]
        samples_per_s = (B) / (mu_i / 1000.0)
        nodes_per_s = (B * N) / (mu_i / 1000.0)

        # ---------- Training ----------
        _reset_peak_mem(args.device)
        lam = float(args.lam_ham) if kind == "mgnhp" else 0.0
        def train_once(i=[0]):
            b = _pick(i[0]); i[0] += 1
            _bench_train_step(ab, "hnn" if kind=="hnn" else ("meshft" if kind=="meshft" else "mgn"),
                              model, opt, b, args.dt, args.device, world["mask_2c"],
                              world["sigma_q"], world["sigma_p"], lam_ham=lam)
        mu_t, sd_t, md_t = _time_loop(lambda: train_once(), args.warmup, args.iters, args.device)
        peak_train = _peak_mem_mb(args.device)

        results.append(dict(
            model=("MeshFT-Net" if kind=="meshft" else "MGN-HP" if kind=="mgnhp" else "MGN" if kind=="mgn" else "HNN"),
            params=n_params,
            infer_ms=mu_i, infer_ms_sd=sd_i, infer_ms_med=md_i, infer_peakMB=peak_inf,
            train_ms=mu_t, train_ms_sd=sd_t, train_ms_med=md_t, train_peakMB=peak_train,
            samples_per_s=samples_per_s, nodes_per_s=nodes_per_s,
            gate_alpha=float(model.get("gate_alpha", 1.0)) if isinstance(model, dict) else float("nan"),
        ))

    # print table
    hdr = ["model","params","infer_ms","train_ms","samples/s","nodes/s","inf_peakMB","train_peakMB","gate_alpha"]
    print("\n=== Compute-cost benchmark ===")
    print(f"(mesh={args.mesh} grid={args.grid[0]}x{args.grid[1]} N={world['coords'].shape[0]} "
          f"B={args.batch_size} dt={args.dt} device={args.device})")
    print(",".join(hdr))
    for r in results:
        print(",".join([
            r["model"],
            str(r["params"]),
            f"{r['infer_ms']:.3f}±{r['infer_ms_sd']:.3f}",
            f"{r['train_ms']:.3f}±{r['train_ms_sd']:.3f}",
            f"{r['samples_per_s']:.1f}",
            f"{r['nodes_per_s']:.1f}",
            f"{r['infer_peakMB']:.1f}",
            f"{r['train_peakMB']:.1f}",
            f"{r['gate_alpha']:.3f}" if not math.isnan(r["gate_alpha"]) else "-"
        ]))
    # optional CSV
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr)
            for r in results:
                w.writerow([
                    r["model"], r["params"],
                    f"{r['infer_ms']:.6f}", f"{r['train_ms']:.6f}",
                    f"{r['samples_per_s']:.6f}", f"{r['nodes_per_s']:.6f}",
                    f"{r['infer_peakMB']:.3f}", f"{r['train_peakMB']:.3f}",
                    f"{r['gate_alpha']:.6f}" if not math.isnan(r["gate_alpha"]) else ""
                ])

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-module", type=str, default="analytic_bench.py",
                    help="Path or module name of the original script")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mesh", type=str, default="grid", choices=["grid","delaunay"])
    ap.add_argument("--grid", type=int, nargs=2, default=[32,32])
    ap.add_argument("--npoints", type=int, default=None)
    ap.add_argument("--Lx", type=float, default=1.0); ap.add_argument("--Ly", type=float, default=1.0)

    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_size", type=int, default=256)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--c_speed", type=float, default=1.0)
    ap.add_argument("--c_wave", type=float, default=None)
    ap.add_argument("--state_mode", type=str, default="canonical", choices=["canonical","velocity"])
    ap.add_argument("--data_state_mode", type=str, default="canonical", choices=["canonical","velocity"])
    ap.add_argument("--miss_ratio", type=float, default=0.0)
    ap.add_argument("--miss_mode", type=str, default="random", choices=["random","grid"])
    ap.add_argument("--grid_stride", type=int, default=2)
    ap.add_argument("--mask_seed", type=int, default=None)
    ap.add_argument("--use_weighted_loss", type=int, default=1)

    ap.add_argument("--mgn_hidden", type=int, default=64)
    ap.add_argument("--mgn_layers", type=int, default=4)
    ap.add_argument("--lam_ham", type=float, default=0.1)  # only for MGN-HP in cost test

    ap.add_argument("--hnn_enable", type=int, default=1)
    ap.add_argument("--hnn_hidden", type=int, default=64)
    ap.add_argument("--hnn_layers", type=int, default=4)

    ap.add_argument("--meshft_hodge_mode", type=str, default="learn_geom", choices=["learn","learn_geom","theory"])
    ap.add_argument("--meshft_geom_hidden", type=int, default=64)
    ap.add_argument("--meshft_geom_layers", type=int, default=2)
    ap.add_argument("--meshft_use_speed_scalar", type=int, default=0)
    ap.add_argument("--meshft_w_structure", type=str, default="diag", choices=["diag","offdiag"])
    ap.add_argument("--offdiag_init", type=float, default=-6.0)
    ap.add_argument("--normalize_hodge", type=int, default=0)
    ap.add_argument("--use_spectral_norm", type=int, default=0)
    ap.add_argument("--cfl_gate", type=int, default=1)
    ap.add_argument("--cfl_gate_safety", type=float, default=1.0)

    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="runs/analytic_bench/cost_bench.csv")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ab = _import_ab(args.ab-module if hasattr(args, "ab-module") else args.ab_module)  # hyphen guard
    torch.set_default_dtype(torch.float32)
    run_one_bench(ab, args)
    