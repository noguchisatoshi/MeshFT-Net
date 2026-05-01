## MeshFT-Net Benchmarks

This repo contains compact runners to evaluate structure-preserving dynamics models—centered around **MeshFT-Net**—across five tasks:

- **Analytic Wave Benchmark** — closed-form 2D plane-wave data; one-step MSE and long-horizon energy drift.
- **OOD Benchmark** — frequency / resolution / parameter / long-horizon extrapolation on the analytic setup.
- **Physical Consistency Benchmark** — dispersion accuracy, canonical consistency, PDE residual, equipartition, momentum conservation, plus short-horizon learning diagnostics.
- **Dissipative Wave Benchmark** — MeshFT-Net under controlled damping; drift-vs-damping behavior.
- **The Well (Acoustics) Benchmark** — *acoustic_scattering_discontinuous* from **The Well** dataset; MeshFT-Net only.
- **Ablation Study (Topology / Orientation / Metric)** — stress-tests structure by (i) dropping orientation, (ii) scrambling topology, (iii) allowing indefinite metric, and (iv) learning \(J\) with/without PSD constraints.

---

## Create a clean env

```bash
conda create -n meshft-net python=3.10 -y
conda activate meshft-net
pip install -r requirements.txt
```

## How to run

Each benchmark has a one-line shell driver. All scripts are self-documenting via env vars and have sensible defaults.  
Make them executable first:

### Analytic wave benchmark (plane waves; accuracy/energy)
```bash
bash scripts/run_analytic_wave_bench.sh
```

### OOD extrapolation benchmark (freq/resolution/param/long)
```bash
bash scripts/run_ood_bench.sh
```

### Physical-consistency benchmark (dispersion, PDE residual, etc.)
```bash
bash scripts/run_phys_bench.sh
```

### Dissipative benchmark (near-Hamiltonian + controlled damping)
```bash
bash scripts/run_dissipative_bench.sh
```

### The Well (acoustics) benchmark with MeshFT-Net(+Diss)
```bash
bash scripts/run_well_bench.sh
```

### Ablation study (topology / orientation / metric structure)
```bash
bash scripts/run_ablation.sh
```
