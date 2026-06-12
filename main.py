"""GPU pollutant diffusion & hotspot detector.

A 2-D grid is seeded with pollutant *sources* and *absorbers*. On every time
step the substance diffuses (5-point stencil + decay), sources/absorbers are
re-applied, and each cell is checked against a danger threshold. After the
simulation we report:

  * how many time steps each cell spent in a dangerous state,
  * how many cells were ever dangerous,
  * per-region statistics (mean, standard deviation),
  * the hotspots: cells that stand out within their own region.

The whole pipeline runs on the GPU through a handful of CUDA kernels
(see the ``kernels/`` directory). Two analysis stages are demonstrated:

  basic  -- a single global danger threshold and hotspot sensitivity (alpha),
  bonus  -- per-region *profiles*, so each region reacts with its own
            threshold and sensitivity (stored in CUDA constant memory).

Run ``python main.py --help`` to see all options.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pycuda.autoinit  # noqa: F401  (creates the CUDA context on import)
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

# Number of distinct region profiles (must match MAX_TYPES in region_profile.cuh).
MAX_TYPES = 4

# Per-type (alpha, danger_threshold) used by the bonus stage.
REGION_PROFILES = np.array(
    [
        (0.8, 0.60),
        (1.2, 0.10),
        (1.7, 0.35),
        (1.1, 0.15),
    ],
    dtype=np.float32,
)

# Elementwise (per-cell) kernels are launched with 32x32 thread blocks.
CELL_BLOCK = (32, 32, 1)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    width: int
    height: int
    steps: int
    decay: float
    source_value: float
    absorb_value: float
    danger_threshold: float
    alpha: float
    region_rows: int
    region_cols: int
    num_sources: int
    num_absorbers: int
    seed: int
    plot: bool
    assets_dir: str

    @property
    def region_grid(self) -> tuple[int, int]:
        """(region_rows_count, region_cols_count)."""
        return self.height // self.region_rows, self.width // self.region_cols

    @property
    def total_regions(self) -> int:
        rows, cols = self.region_grid
        return rows * cols


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        description="GPU pollutant diffusion & hotspot detector (PyCUDA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--width", type=int, default=128, help="Grid width (cells).")
    p.add_argument("--height", type=int, default=128, help="Grid height (cells).")
    p.add_argument("--steps", type=int, default=100, help="Number of time steps.")
    p.add_argument("--decay", type=float, default=0.05, help="Dissipation per step.")
    p.add_argument("--source-value", type=float, default=1.0, help="Source concentration.")
    p.add_argument("--absorb-value", type=float, default=0.5, help="Amount absorbers remove.")
    p.add_argument("--danger-threshold", type=float, default=0.30,
                   help="Global danger threshold (basic stage).")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Global hotspot sensitivity (basic stage).")
    p.add_argument("--region-rows", type=int, default=16, help="Rows per region.")
    p.add_argument("--region-cols", type=int, default=16, help="Columns per region.")
    p.add_argument("--num-sources", type=int, default=40, help="Number of source cells.")
    p.add_argument("--num-absorbers", type=int, default=24, help="Number of absorber cells.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for cell placement.")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="Skip writing the visualization PNG.")
    p.add_argument("--assets-dir", default="assets", help="Where to write the PNG.")

    a = p.parse_args(argv)
    cfg = Config(
        width=a.width, height=a.height, steps=a.steps, decay=a.decay,
        source_value=a.source_value, absorb_value=a.absorb_value,
        danger_threshold=a.danger_threshold, alpha=a.alpha,
        region_rows=a.region_rows, region_cols=a.region_cols,
        num_sources=a.num_sources, num_absorbers=a.num_absorbers,
        seed=a.seed, plot=a.plot, assets_dir=a.assets_dir,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.width % cfg.region_cols or cfg.height % cfg.region_rows:
        raise SystemExit("Grid size must be divisible by the region size.")
    block_size = cfg.region_rows * cfg.region_cols
    if block_size & (block_size - 1):
        raise SystemExit("region_rows * region_cols must be a power of two "
                         "(required by the shared-memory reduction).")
    if cfg.region_rows > 32 or cfg.region_cols > 32 or block_size > 1024:
        raise SystemExit("A region must fit in one block (<= 32x32, <= 1024 threads).")
    if cfg.num_sources + cfg.num_absorbers > cfg.width * cfg.height:
        raise SystemExit("More sources + absorbers than cells in the grid.")


# --------------------------------------------------------------------------- #
# Kernel loading / launch helpers                                             #
# --------------------------------------------------------------------------- #
def load_kernel(name: str) -> str:
    with open(os.path.join(KERNELS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def cell_grid(cfg: Config) -> tuple[int, int, int]:
    """Launch grid (in blocks) for the per-cell kernels."""
    return (math.ceil(cfg.width / 32), math.ceil(cfg.height / 32), 1)


def build_initial_state(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Place sources/absorbers deterministically and return (conc0, cell_type)."""
    rng = np.random.default_rng(cfg.seed)
    conc = np.zeros((cfg.height, cfg.width), dtype=np.float32)
    cell_type = np.zeros((cfg.height, cfg.width), dtype=np.int8)

    n = cfg.num_sources + cfg.num_absorbers
    picked = rng.choice(cfg.width * cfg.height, size=n, replace=False)
    src_cells, abs_cells = picked[:cfg.num_sources], picked[cfg.num_sources:]

    flat_type, flat_conc = cell_type.reshape(-1), conc.reshape(-1)
    flat_type[src_cells] = 1          # source
    flat_conc[src_cells] = cfg.source_value
    flat_type[abs_cells] = 2          # absorber
    return conc, cell_type


# --------------------------------------------------------------------------- #
# Simulation                                                                  #
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> dict:
    conc0, cell_type = build_initial_state(cfg)
    rows, cols = cfg.region_grid

    # --- compile kernels ---------------------------------------------------- #
    diffuse = SourceModule(load_kernel("diffusion.cu")).get_function("diffusion_step_kernel")
    sources = SourceModule(load_kernel("sources.cu")).get_function("apply_sources_kernel")
    count_danger = SourceModule(load_kernel("count.cu")).get_function("count_danger_kernel")
    ever = SourceModule(load_kernel("ever_dangerous.cu")).get_function("count_ever_dangerous_kernel")
    hotspot = SourceModule(load_kernel("hotspot.cu")).get_function("region_hotspot_stats_kernel")

    bonus_mod = SourceModule(
        load_kernel("region_profile.cuh")
        + load_kernel("region_profile.cu")
        + load_kernel("count_bonus.cu")
        + load_kernel("hotspot_bonus.cu")
    )
    count_danger_bonus = bonus_mod.get_function("count_danger_kernel_bonus")
    hotspot_bonus = bonus_mod.get_function("region_hotspot_stats_kernel_bonus")

    # --- device buffers ----------------------------------------------------- #
    conc_gpu = cuda.mem_alloc(conc0.nbytes)
    conc_out_gpu = cuda.mem_alloc(conc0.nbytes)
    cell_type_gpu = cuda.mem_alloc(cell_type.nbytes)
    cuda.memcpy_htod(cell_type_gpu, cell_type)

    danger_count = np.zeros((cfg.height, cfg.width), dtype=np.int32)
    danger_count_gpu = cuda.mem_alloc(danger_count.nbytes)

    region_means = np.zeros(cfg.total_regions, dtype=np.float32)
    region_dev = np.zeros(cfg.total_regions, dtype=np.float32)
    is_hotspot = np.zeros((cfg.height, cfg.width), dtype=np.int8)
    region_means_gpu = cuda.mem_alloc(region_means.nbytes)
    region_dev_gpu = cuda.mem_alloc(region_dev.nbytes)
    is_hotspot_gpu = cuda.mem_alloc(is_hotspot.nbytes)

    block = CELL_BLOCK
    grid = cell_grid(cfg)
    hot_block = (cfg.region_cols, cfg.region_rows, 1)
    hot_grid = (cols, rows, 1)
    hot_shared = 2 * cfg.region_rows * cfg.region_cols * np.dtype(np.int32).itemsize

    def diffuse_loop(count_step):
        """Reset the field to conc0 and run `steps` diffusion/source/count steps.

        Returns the device buffer holding the final concentration field.
        Buffers are ping-ponged, so no per-step device-to-device copy is needed.
        """
        cuda.memcpy_htod(conc_gpu, conc0)
        src, dst = conc_gpu, conc_out_gpu
        for _ in range(cfg.steps):
            diffuse(src, dst, np.float32(cfg.decay),
                    np.int32(cfg.height), np.int32(cfg.width), block=block, grid=grid)
            src, dst = dst, src
            sources(cell_type_gpu, src,
                    np.float32(cfg.source_value), np.float32(cfg.absorb_value),
                    np.int32(cfg.height), np.int32(cfg.width), block=block, grid=grid)
            count_step(src)
        cuda.Context.synchronize()
        return src

    start = time.perf_counter()

    # --- basic stage: uniform threshold & alpha ----------------------------- #
    cuda.memset_d32(danger_count_gpu, 0, danger_count.size)
    final_basic = diffuse_loop(lambda field: count_danger(
        field, danger_count_gpu, np.float32(cfg.danger_threshold),
        np.int32(cfg.height), np.int32(cfg.width), block=block, grid=grid))

    ever_gpu = cuda.mem_alloc(np.int32(0).nbytes)
    cuda.memset_d32(ever_gpu, 0, 1)
    ever(danger_count_gpu, np.int32(cfg.height), np.int32(cfg.width), ever_gpu,
         block=block, grid=grid)

    hotspot(danger_count_gpu, region_means_gpu, region_dev_gpu, is_hotspot_gpu,
            np.int32(cfg.width), np.int32(cfg.region_rows), np.int32(cfg.region_cols),
            np.float32(cfg.alpha), block=hot_block, grid=hot_grid, shared=hot_shared)
    cuda.Context.synchronize()

    conc_final = np.empty_like(conc0)
    cuda.memcpy_dtoh(conc_final, final_basic)
    cuda.memcpy_dtoh(danger_count, danger_count_gpu)
    cuda.memcpy_dtoh(region_means, region_means_gpu)
    cuda.memcpy_dtoh(region_dev, region_dev_gpu)
    cuda.memcpy_dtoh(is_hotspot, is_hotspot_gpu)
    ever_count = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(ever_count, ever_gpu)

    # --- bonus stage: per-region profiles ----------------------------------- #
    cuda.memcpy_htod(bonus_mod.get_global("d_region_profiles")[0], REGION_PROFILES)
    region_type = (np.arange(cfg.total_regions) % MAX_TYPES).astype(np.int32)
    region_type_gpu = cuda.mem_alloc(region_type.nbytes)
    cuda.memcpy_htod(region_type_gpu, region_type)

    danger_bonus = np.zeros_like(danger_count)
    danger_bonus_gpu = cuda.mem_alloc(danger_bonus.nbytes)
    cuda.memset_d32(danger_bonus_gpu, 0, danger_bonus.size)

    diffuse_loop(lambda field: count_danger_bonus(
        field, danger_bonus_gpu, region_type_gpu,
        np.int32(cfg.height), np.int32(cfg.width),
        np.int32(cfg.region_rows), np.int32(cfg.region_cols), np.int32(cols),
        block=block, grid=grid))

    means_bonus = np.zeros(cfg.total_regions, dtype=np.float32)
    dev_bonus = np.zeros(cfg.total_regions, dtype=np.float32)
    hotspot_bonus_map = np.zeros((cfg.height, cfg.width), dtype=np.int8)
    means_bonus_gpu = cuda.mem_alloc(means_bonus.nbytes)
    dev_bonus_gpu = cuda.mem_alloc(dev_bonus.nbytes)
    hotspot_bonus_gpu = cuda.mem_alloc(hotspot_bonus_map.nbytes)

    hotspot_bonus(danger_bonus_gpu, means_bonus_gpu, dev_bonus_gpu, hotspot_bonus_gpu,
                  region_type_gpu, np.int32(cfg.width),
                  np.int32(cfg.region_rows), np.int32(cfg.region_cols),
                  block=hot_block, grid=hot_grid, shared=hot_shared)
    cuda.Context.synchronize()

    cuda.memcpy_dtoh(danger_bonus, danger_bonus_gpu)
    cuda.memcpy_dtoh(means_bonus, means_bonus_gpu)
    cuda.memcpy_dtoh(dev_bonus, dev_bonus_gpu)
    cuda.memcpy_dtoh(hotspot_bonus_map, hotspot_bonus_gpu)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return {
        "cell_type": cell_type,
        "conc_final": conc_final,
        "danger_count": danger_count,
        "is_hotspot": is_hotspot,
        "region_means": region_means.reshape(rows, cols),
        "region_dev": region_dev.reshape(rows, cols),
        "ever_count": int(ever_count[0]),
        "danger_bonus": danger_bonus,
        "means_bonus": means_bonus.reshape(rows, cols),
        "hotspot_bonus": hotspot_bonus_map,
        "elapsed_ms": elapsed_ms,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def print_summary(cfg: Config, res: dict) -> None:
    rows, cols = cfg.region_grid
    print(f"Grid                : {cfg.width} x {cfg.height} cells, {cfg.steps} steps")
    print(f"Regions             : {rows} x {cols} ({cfg.region_rows}x{cfg.region_cols} each)")
    print(f"Cells ever dangerous: {res['ever_count']} / {cfg.width * cfg.height}")
    print(f"Hotspots (basic)    : {int(res['is_hotspot'].sum())}")
    print(f"Hotspots (bonus)    : {int(res['hotspot_bonus'].sum())}")
    print(f"GPU pipeline time   : {res['elapsed_ms']:.2f} ms")


def save_visualization(cfg: Config, res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (axes[0, 0], res["cell_type"], "Cell layout (1=source, 2=absorber)", "tab20c"),
        (axes[0, 1], res["conc_final"], "Final concentration", "inferno"),
        (axes[0, 2], res["danger_count"], "Danger count (basic)", "viridis"),
        (axes[1, 0], res["is_hotspot"], "Hotspots (basic)", "Reds"),
        (axes[1, 1], res["means_bonus"], "Region mean danger (bonus)", "magma"),
        (axes[1, 2], res["hotspot_bonus"], "Hotspots (bonus)", "Reds"),
    ]
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Pollutant diffusion -- {cfg.width}x{cfg.height} grid, {cfg.steps} steps",
        fontsize=14,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved : {path}")


def main(argv=None) -> None:
    cfg = parse_args(argv)
    res = run(cfg)
    print_summary(cfg, res)
    if cfg.plot:
        try:
            save_visualization(cfg, res, os.path.join(cfg.assets_dir, "simulation.png"))
        except ImportError:
            print("matplotlib not installed -- skipping visualization "
                  "(install it or pass --no-plot).")


if __name__ == "__main__":
    main()
