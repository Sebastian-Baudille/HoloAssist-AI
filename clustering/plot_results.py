"""plot_results.py — Generate validation graphs from accuracy_report.json.

Reads ~/holoassist_dataset/accuracy_report.json (produced by verify_detection.py)
and writes PNGs to clustering/plots/ and optionally to the site screenshots folder.

Usage:
    python3 clustering/plot_results.py
    python3 clustering/plot_results.py --dataset ~/holoassist_dataset
    python3 clustering/plot_results.py --site /path/to/HoloAssist-AI-site/screenshots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1a1a1a",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#cccccc",
    "axes.titlecolor":   "#ffffff",
    "text.color":        "#cccccc",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "grid.color":        "#2a2a2a",
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "legend.framealpha": 0.15,
    "legend.edgecolor":  "#444444",
    "savefig.facecolor": "#1a1a1a",
    "savefig.dpi":       160,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})

ORANGE  = "#FF6600"
BLUE    = "#33BBFF"
GREEN   = "#44CC88"
YELLOW  = "#FFCC44"
GREY    = "#555555"


def load_report(dataset_dir: Path) -> dict:
    path = dataset_dir / "accuracy_report.json"
    if not path.exists():
        print(f"ERROR: {path} not found — run verify_detection.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ── Plot 1: per-scene error histogram ─────────────────────────────────────────

def plot_per_scene_histogram(report: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    all_errors_cm: list[float] = []
    split_data: list[tuple[str, list[float]]] = []

    for split in ("train", "val"):
        if split not in report:
            continue
        errors = [r["mean_error_m"] * 100 for r in report[split]["scenes"]]
        all_errors_cm.extend(errors)
        split_data.append((split, errors))

    if not all_errors_cm:
        print("No scene data found in report.")
        return out_dir / "empty.png"

    # Shared bins across both splits
    lo, hi = 0, max(all_errors_cm) * 1.15
    bins = np.linspace(lo, hi, 22)

    colors  = [ORANGE, BLUE]
    labels  = [f"Train ({len(d)} scenes)", f"Val ({len(d)} scenes)"]
    for (split_name, errors), color, label in zip(split_data, colors, labels):
        ax.hist(errors, bins=bins, color=color, alpha=0.78, label=label,
                edgecolor="#0a0a0a", linewidth=0.5)

    overall_mean = np.mean(all_errors_cm)
    ax.axvline(overall_mean, color="#ffffff", linewidth=1.4, linestyle="--",
               label=f"Mean = {overall_mean:.2f} cm")
    ax.axvline(3.0, color=YELLOW, linewidth=1.0, linestyle=":", alpha=0.7,
               label="Target < 3 cm")

    ax.set_xlabel("Mean centroid error per scene (cm)")
    ax.set_ylabel("Scene count")
    ax.set_title("DBSCAN Clustering — Per-scene centroid error distribution")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--")

    path = out_dir / "clustering_error_histogram.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 2: error by cube count ───────────────────────────────────────────────

def plot_error_by_k(report: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    by_k: dict[int, list[float]] = {}
    for split in ("train", "val"):
        if split not in report:
            continue
        for r in report[split]["scenes"]:
            k = r["cube_count"]
            by_k.setdefault(k, []).append(r["mean_error_m"] * 100)

    if not by_k:
        plt.close(fig)
        return out_dir / "empty.png"

    ks      = sorted(by_k)
    means   = [np.mean(by_k[k]) for k in ks]
    stds    = [np.std(by_k[k])  for k in ks]
    counts  = [len(by_k[k])     for k in ks]

    x = np.arange(len(ks))
    bars = ax.bar(x, means, yerr=stds, width=0.55,
                  color=ORANGE, alpha=0.85, capsize=6,
                  error_kw={"ecolor": "#aaaaaa", "linewidth": 1.4},
                  edgecolor="#0a0a0a", linewidth=0.5)

    ax.axhline(3.0, color=YELLOW, linewidth=1.0, linestyle=":", alpha=0.7,
               label="Target < 3 cm")
    ax.axhline(np.mean([np.mean(v) for v in by_k.values()]),
               color="#ffffff", linewidth=1.2, linestyle="--",
               label=f"Overall mean = {np.mean(means):.2f} cm")

    for bar, mean, std, count in zip(bars, means, stds, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.05,
                f"{mean:.2f} cm\n(n={count})",
                ha="center", va="bottom", fontsize=9, color="#cccccc")

    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k} cube{'s' if k > 1 else ''}" for k in ks])
    ax.set_ylabel("Mean centroid error (cm)")
    ax.set_title("DBSCAN Clustering — Error by number of cubes in scene")
    ax.legend()
    ax.set_ylim(0, max(means) * 1.55 + 0.5)
    ax.grid(axis="y", linestyle="--")

    path = out_dir / "clustering_error_by_k.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 3: K-Means vs DBSCAN comparison ──────────────────────────────────────

def plot_method_comparison(out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Cube detection method comparison", fontsize=14, fontweight="bold",
                 color="#ffffff", y=1.01)

    # Error comparison
    ax = axes[0]
    methods = ["AprilTag\n(base project)", "K-Means\n(baseline)", "DBSCAN 1.5×\n(early)", "DBSCAN 2.0×\n(final)"]
    errors  = [0.75, 2.65, 1.63, 1.63]     # cm; AprilTag ~5-10 mm → 0.75 approx midpoint
    colors  = [GREY, GREY, "#884400", ORANGE]

    bars = ax.bar(methods, errors, color=colors, alpha=0.85,
                  edgecolor="#0a0a0a", linewidth=0.5, width=0.6)
    ax.axhline(3.0, color=YELLOW, linewidth=1.0, linestyle=":", label="Target < 3 cm")

    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.06,
                f"{val:.2f} cm", ha="center", va="bottom", fontsize=9, color="#cccccc")

    ax.set_ylabel("Mean centroid error (cm)")
    ax.set_title("Centroid accuracy")
    ax.legend()
    ax.set_ylim(0, 3.6)
    ax.grid(axis="y", linestyle="--")

    # Exact-count comparison
    ax = axes[1]
    methods2 = ["K-Means", "DBSCAN 1.5×", "DBSCAN 2.0×"]
    exact    = [90, 82, 100]
    colors2  = [GREY, "#884400", ORANGE]

    bars2 = ax.bar(methods2, exact, color=colors2, alpha=0.85,
                   edgecolor="#0a0a0a", linewidth=0.5, width=0.5)
    for bar, val in zip(bars2, exact):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                f"{val}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#ffffff")

    ax.set_ylabel("Exact cube count rate (%)")
    ax.set_title("Exact count accuracy")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--")

    plt.tight_layout()
    path = out_dir / "clustering_method_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 4: per-scene scatter (detected vs GT count) ─────────────────────────

def plot_detection_scatter(report: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))

    gt_counts:  list[int] = []
    det_counts: list[int] = []
    errors_cm:  list[float] = []
    splits:     list[str] = []

    for split in ("train", "val"):
        if split not in report:
            continue
        for r in report[split]["scenes"]:
            gt_counts.append(r["cube_count"])
            det_counts.append(r["detected_count"])
            errors_cm.append(r["mean_error_m"] * 100)
            splits.append(split)

    if not gt_counts:
        plt.close(fig)
        return out_dir / "empty.png"

    gt_arr  = np.array(gt_counts)
    det_arr = np.array(det_counts)
    err_arr = np.array(errors_cm)

    # Colour by error magnitude, shape by split
    scatter = ax.scatter(gt_arr, det_arr, c=err_arr,
                         cmap="RdYlGn_r", vmin=0, vmax=3.0,
                         s=55, alpha=0.85, edgecolors="#333333", linewidths=0.5,
                         zorder=3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean centroid error (cm)", color="#cccccc")
    cbar.ax.yaxis.set_tick_params(color="#888888")

    # Perfect detection line
    lo, hi = min(gt_counts) - 0.5, max(gt_counts) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="#ffffff", linewidth=1.0,
            alpha=0.5, label="Perfect detection", zorder=2)

    ax.set_xlabel("Ground truth cube count")
    ax.set_ylabel("Detected cube count")
    ax.set_title("DBSCAN — Detected vs ground truth count\n(coloured by centroid error)")
    ax.set_xticks(sorted(set(gt_counts)))
    ax.set_yticks(sorted(set(det_counts)))
    ax.legend(loc="upper left")
    ax.grid(linestyle="--")

    path = out_dir / "clustering_detection_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate validation plots from accuracy_report.json"
    )
    parser.add_argument(
        "--dataset", default=str(Path.home() / "holoassist_dataset"),
        help="Dataset directory containing accuracy_report.json"
    )
    parser.add_argument(
        "--out", default=str(Path(__file__).parent / "plots"),
        help="Output directory for PNGs (default: clustering/plots/)"
    )
    parser.add_argument(
        "--site", default=None,
        help="Also copy PNGs here (e.g. path/to/HoloAssist-AI-site/screenshots/)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = load_report(Path(args.dataset))

    print(f"\nGenerating plots → {out_dir}/")

    generated: list[Path] = []
    generated.append(plot_per_scene_histogram(report, out_dir))
    generated.append(plot_error_by_k(report, out_dir))
    generated.append(plot_method_comparison(out_dir))
    generated.append(plot_detection_scatter(report, out_dir))

    if args.site:
        import shutil
        site_dir = Path(args.site)
        site_dir.mkdir(parents=True, exist_ok=True)
        for p in generated:
            if p.exists():
                dest = site_dir / p.name
                shutil.copy2(p, dest)
                print(f"  Copied → {dest}")

    # Print summary
    print("\n── Summary ──────────────────────────────────────────────")
    for split in ("train", "val"):
        if split not in report:
            continue
        r = report[split]
        print(f"  {split.upper()} ({r['scene_count']} scenes): "
              f"mean={r['mean_error_m']*100:.2f} cm  "
              f"std={r['std_error_m']*100:.2f} cm  "
              f"exact={r['exact_count_rate']*100:.0f}%  "
              f"recall={r['cube_recall']*100:.0f}%")

    print(f"\nDone — {len(generated)} plots saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
