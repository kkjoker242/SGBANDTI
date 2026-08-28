"""Generate 300 dpi manuscript figures for the SGBANDTI paper.

Run with the project environment:
    conda activate sentiment
    python code/generate_paper_figures.py --paper-dir "D:/毕设/Research_template"

In PowerShell sessions where ``conda activate`` is not initialized, the same
script can be called through the environment's Python executable.

Figure design:
    1. BioSNAP split-level box-style summaries for random, unseen-drug, and
       unseen-target experiments.
    2. Corrected two-by-two ablation summary.

The split-level boxplots follow the visual convention of the local reference
``plot_ablation_boxplot.py``: box statistics are reconstructed from five-seed
mean and sample standard deviation. This is appropriate for a compact
manuscript summary, but the figure caption must state that the boxes are
mean/SD-derived summaries rather than empirical quartiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# Editable text in vector exports; PNG is required for the current manuscript.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"


METHODS = [
    "RF",
    "GNN-CPI",
    "TransformerCPI",
    "MolTrans",
    "MGNDTI",
    "INGNN",
    "DrugBAN",
    "SGBANDTI",
]

SHORT_LABELS = {
    "RF": "RF",
    "GNN-CPI": "GNN-CPI",
    "TransformerCPI": "TransCPI",
    "MolTrans": "MolTrans",
    "MGNDTI": "MGNDTI",
    "INGNN": "INGNN",
    "DrugBAN": "DrugBAN",
    "SGBANDTI": "SGBANDTI",
}

# Color-blind friendly, close to the reference script but with SGBANDTI as the
# manuscript highlight.
COLORS = {
    "RF": "#648FFF",
    "GNN-CPI": "#785EF0",
    "TransformerCPI": "#FFB000",
    "MolTrans": "#DC267F",
    "MGNDTI": "#FE6100",
    "INGNN": "#00B0BE",
    "DrugBAN": "#9A4D8E",
    "SGBANDTI": "#009E73",
}

GRID_COLOR = "#D0D0D0"
TEXT_COLOR = "#222222"


# Values are mean and sample standard deviation over seeds 42, 52, 62, 72, 82.
# Source: results/00_实验结果汇总.md.
SPLIT_SUMMARY = {
    ("Random", "AUROC"): {
        "RF": (0.8402, 0.0008),
        "GNN-CPI": (0.7094, 0.0029),
        "TransformerCPI": (0.8399, 0.0068),
        "MolTrans": (0.8867, 0.0045),
        "MGNDTI": (0.8947, 0.0019),
        "INGNN": (0.8722, 0.0006),
        "DrugBAN": (0.9100, 0.0023),
        "SGBANDTI": (0.9062, 0.0019),
    },
    ("Random", "AUPRC"): {
        "RF": (0.8678, 0.0007),
        "GNN-CPI": (0.7247, 0.0017),
        "TransformerCPI": (0.8553, 0.0048),
        "MolTrans": (0.8927, 0.0048),
        "MGNDTI": (0.8983, 0.0042),
        "INGNN": (0.8776, 0.0013),
        "DrugBAN": (0.9172, 0.0031),
        "SGBANDTI": (0.9132, 0.0043),
    },
    ("Unseen drug", "AUROC"): {
        "RF": (0.8493, 0.0006),
        "GNN-CPI": (0.6797, 0.0897),
        "TransformerCPI": (0.8460, 0.0113),
        "MolTrans": (0.8407, 0.0082),
        "MGNDTI": (0.8589, 0.0057),
        "INGNN": (0.8417, 0.0057),
        "DrugBAN": (0.8750, 0.0030),
        "SGBANDTI": (0.8794, 0.0019),
    },
    ("Unseen drug", "AUPRC"): {
        "RF": (0.8724, 0.0005),
        "GNN-CPI": (0.6795, 0.0933),
        "TransformerCPI": (0.8551, 0.0061),
        "MolTrans": (0.8473, 0.0057),
        "MGNDTI": (0.8675, 0.0038),
        "INGNN": (0.8510, 0.0023),
        "DrugBAN": (0.8807, 0.0048),
        "SGBANDTI": (0.8821, 0.0026),
    },
    ("Unseen target", "AUROC"): {
        "RF": (0.6979, 0.0110),
        "GNN-CPI": (0.6501, 0.0030),
        "TransformerCPI": (0.6160, 0.0319),
        "MolTrans": (0.6820, 0.0166),
        "MGNDTI": (0.6910, 0.0105),
        "INGNN": (0.6526, 0.0090),
        "DrugBAN": (0.6560, 0.0152),
        "SGBANDTI": (0.6345, 0.0182),
    },
    ("Unseen target", "AUPRC"): {
        "RF": (0.6867, 0.0068),
        "GNN-CPI": (0.6526, 0.0017),
        "TransformerCPI": (0.6032, 0.0216),
        "MolTrans": (0.6788, 0.0193),
        "MGNDTI": (0.6836, 0.0168),
        "INGNN": (0.6421, 0.0075),
        "DrugBAN": (0.6333, 0.0134),
        "SGBANDTI": (0.6124, 0.0212),
    },
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
        }
    )


def build_boxplot_stats(mean: float, std: float, y_min: float, y_max: float) -> dict:
    """Construct bxp statistics from a five-seed mean and sample SD."""
    q1 = mean - 0.6745 * std
    q3 = mean + 0.6745 * std
    lo = mean - 1.5 * std
    hi = mean + 1.5 * std
    return {
        "med": mean,
        "q1": max(y_min, q1),
        "q3": min(y_max, q3),
        "whislo": max(y_min, lo),
        "whishi": min(y_max, hi),
        "fliers": [],
        "label": "",
    }


def auto_y_range(rows: dict[str, tuple[float, float]]) -> tuple[float, float]:
    vals = []
    for mean, std in rows.values():
        vals.extend([mean - 1.5 * std, mean + 1.5 * std])
    lo = max(0.50, min(vals) - 0.015)
    hi = min(0.95, max(vals) + 0.015)
    return np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=TEXT_COLOR,
    )


def save_png(fig: plt.Figure, out_file: Path, dpi: int = 300) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_biosnap_split_boxplots(out_dir: Path) -> None:
    scenarios = ["Random", "Unseen drug", "Unseen target"]
    metrics = ["AUROC", "AUPRC"]
    panel_labels = iter(["a", "b", "c", "d", "e", "f"])

    fig = plt.figure(figsize=(8.4, 5.9))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.07,
        right=0.985,
        top=0.94,
        bottom=0.17,
        wspace=0.22,
        hspace=0.30,
    )

    for row, metric in enumerate(metrics):
        for col, scenario in enumerate(scenarios):
            ax = fig.add_subplot(gs[row, col])
            rows = SPLIT_SUMMARY[(scenario, metric)]
            y_min, y_max = auto_y_range(rows)
            stats = [
                build_boxplot_stats(rows[method][0], rows[method][1], y_min, y_max)
                for method in METHODS
            ]
            positions = np.arange(len(METHODS))
            bp = ax.bxp(
                stats,
                positions=positions,
                widths=0.68,
                showfliers=False,
                showcaps=True,
                capwidths=0.34,
                patch_artist=True,
                manage_ticks=False,
                zorder=3,
            )

            for box, method in zip(bp["boxes"], METHODS):
                box.set_facecolor(COLORS[method])
                box.set_edgecolor("black")
                box.set_linewidth(0.7)
                box.set_alpha(0.76)
                box.set_zorder(4)

            for whisk in bp["whiskers"]:
                whisk.set_color("black")
                whisk.set_linewidth(0.75)
                whisk.set_zorder(2)
            for cap in bp["caps"]:
                cap.set_color("black")
                cap.set_linewidth(0.75)
                cap.set_zorder(2)
            for med in bp["medians"]:
                med.set_color("black")
                med.set_linewidth(1.0)
                med.set_solid_capstyle("butt")
                med.set_zorder(5)

            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
            ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.45, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xticks(positions)
            ax.set_xticklabels([SHORT_LABELS[m] for m in METHODS], rotation=34, ha="right")
            if col == 0:
                ax.set_ylabel(metric, fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(scenario, fontsize=9, fontweight="bold", pad=6)
            add_panel_label(ax, next(panel_labels))

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[m], edgecolor="black", linewidth=0.5, alpha=0.76)
        for m in METHODS
    ]
    fig.legend(
        legend_handles,
        [SHORT_LABELS[m] for m in METHODS],
        loc="lower center",
        ncol=8,
        bbox_to_anchor=(0.5, 0.035),
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=8,
        handlelength=1.1,
        handleheight=0.7,
        columnspacing=0.8,
        borderpad=0.35,
    )

    save_png(fig, out_dir / "figure_biosnap_split_boxplots.png")


def plot_ablation(out_dir: Path) -> None:
    configs = ["Full\nSGBANDTI", "No\nsubgraph", "No\nBAN", "No\nboth"]
    auroc = np.array([0.9062, 0.8757, 0.8777, 0.8598])
    auroc_sd = np.array([0.0019, 0.0012, 0.0023, 0.0025])
    auprc = np.array([0.9132, 0.8782, 0.8759, 0.8551])
    auprc_sd = np.array([0.0043, 0.0030, 0.0032, 0.0033])
    drops = auroc[0] - auroc

    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    x = np.arange(len(configs))
    width = 0.36
    ax0.bar(
        x - width / 2,
        auroc,
        width,
        yerr=auroc_sd,
        label="AUROC",
        color="#0F4D92",
        edgecolor="white",
        linewidth=0.6,
    )
    ax0.bar(
        x + width / 2,
        auprc,
        width,
        yerr=auprc_sd,
        label="AUPRC",
        color="#7884B4",
        edgecolor="white",
        linewidth=0.6,
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels(configs)
    ax0.set_ylim(0.84, 0.925)
    ax0.set_ylabel("Score")
    ax0.set_title("Full model and ablated variants", fontsize=8.5, pad=6)
    ax0.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    ax0.set_axisbelow(True)
    ax0.legend(loc="lower left", frameon=False)
    add_panel_label(ax0, "a")

    drop_labels = ["No subgraph", "No BAN", "No both"]
    drop_values = drops[1:]
    y = np.arange(len(drop_labels))[::-1]
    ax1.barh(y, drop_values, color=["#B64342", "#B64342", "#7F2D2D"], edgecolor="white", linewidth=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(drop_labels)
    ax1.set_xlabel("AUROC drop from full model")
    ax1.set_xlim(0, 0.052)
    ax1.set_title("Effect size of component removal", fontsize=8.5, pad=6)
    ax1.grid(axis="x", color="#E7E7E7", linewidth=0.6)
    ax1.set_axisbelow(True)
    for yi, value in zip(y, drop_values):
        ax1.text(value + 0.001, yi, f"{value:.4f}", va="center", fontsize=7)
    add_panel_label(ax1, "b")

    fig.tight_layout()
    save_png(fig, out_dir / "figure_ablation_2x2.png")


def parse_hypertune_report(report_file: Path) -> list[dict[str, float | int | str]]:
    order = [
        "existing_final_baseline",
        "tune_dropout02_wd1e4",
        "tune_heads4",
        "tune_lr1e4",
        "tune_hidden256",
        "tune_bs32",
        "tune_cosine",
        "tune_maxepoch250",
    ]
    short_labels = {
        "existing_final_baseline": "Baseline\n150 ep",
        "tune_dropout02_wd1e4": "Dropout+WD",
        "tune_heads4": "Heads=4",
        "tune_lr1e4": "LR=1e-4",
        "tune_hidden256": "Hidden=256",
        "tune_bs32": "Batch=32",
        "tune_cosine": "LR+Cosine",
        "tune_maxepoch250": "250 ep",
    }
    entries: dict[str, dict[str, float | int | str]] = {}
    pattern = re.compile(
        r"^\|\s*(?P<name>.+?)\s+\(`(?P<tag>[^`]+)`\)\s*\|\s*"
        r"(?P<auroc>\d+\.\d+)\s*\|\s*(?P<auprc>\d+\.\d+)\s*\|\s*(?P<epoch>\d+)\s*\|"
    )
    for line in report_file.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        tag = match.group("tag")
        if tag not in short_labels:
            continue
        entries[tag] = {
            "tag": tag,
            "label": short_labels[tag],
            "display_name": match.group("name"),
            "auroc": float(match.group("auroc")),
            "auprc": float(match.group("auprc")),
            "epoch": int(match.group("epoch")),
        }
    missing = [tag for tag in order if tag not in entries]
    if missing:
        raise ValueError(f"Missing hypertune entries in {report_file}: {missing}")
    return [entries[tag] for tag in order]


def plot_hyperparameter_sensitivity(out_dir: Path, report_file: Path) -> None:
    rows = parse_hypertune_report(report_file)
    baseline = rows[0]["auroc"]
    deltas = [row["auroc"] - baseline for row in rows]
    labels = [str(row["label"]) for row in rows]
    aurocs = np.array([float(row["auroc"]) for row in rows])
    epochs = [int(row["epoch"]) for row in rows]

    better_color = "#1B9E77"
    tie_color = "#8A8A8A"
    worse_color = "#C44E52"
    baseline_color = "#0F4D92"
    bar_colors = []
    for idx, delta in enumerate(deltas):
        if idx == 0:
            bar_colors.append(baseline_color)
        elif delta > 0.001:
            bar_colors.append(better_color)
        elif delta < -0.001:
            bar_colors.append(worse_color)
        else:
            bar_colors.append(tie_color)

    fig = plt.figure(figsize=(7.4, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    x = np.arange(len(rows))
    bars = ax0.bar(
        x,
        aurocs,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax0.axhline(float(baseline), color=baseline_color, linestyle="--", linewidth=0.9)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=28, ha="right")
    ax0.set_ylim(0.885, 0.9165)
    ax0.set_ylabel("Best validation AUROC")
    ax0.set_title("Single-seed validation checks", fontsize=8.5, pad=6)
    ax0.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    ax0.set_axisbelow(True)
    for bar, value, epoch in zip(bars, aurocs, epochs):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00055,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            rotation=90,
        )
        if epoch == 231:
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.0018,
                "best epoch 231",
                ha="center",
                va="top",
                fontsize=6.4,
                rotation=90,
                color=TEXT_COLOR,
            )
    add_panel_label(ax0, "a")

    y_labels = labels[1:][::-1]
    y = np.arange(len(y_labels))
    delta_values = np.array(deltas[1:])[::-1]
    delta_colors = np.array(bar_colors[1:])[::-1]
    ax1.barh(y, delta_values, color=delta_colors, edgecolor="white", linewidth=0.6)
    ax1.axvline(0.0, color=baseline_color, linestyle="--", linewidth=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(y_labels)
    ax1.set_xlabel(r"$\Delta$AUROC vs baseline")
    ax1.set_xlim(-0.018, 0.0085)
    ax1.set_title("Effect relative to 150-epoch baseline", fontsize=8.5, pad=6)
    ax1.grid(axis="x", color="#E7E7E7", linewidth=0.6)
    ax1.set_axisbelow(True)
    for yi, value in zip(y, delta_values):
        x_text = value + (0.00035 if value >= 0 else -0.00035)
        ha = "left" if value >= 0 else "right"
        ax1.text(x_text, yi, f"{value:+.4f}", va="center", ha=ha, fontsize=6.8)
    add_panel_label(ax1, "b")

    fig.tight_layout()
    save_png(fig, out_dir / "figure_hyperparameter_sensitivity.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Research_template",
        help="Path to the LaTeX manuscript directory containing the image folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for generated figure files.",
    )
    args = parser.parse_args()

    apply_style()
    out_dir = args.out_dir if args.out_dir is not None else args.paper_dir / "image"
    plot_biosnap_split_boxplots(out_dir)
    plot_ablation(out_dir)
    plot_hyperparameter_sensitivity(
        out_dir,
        Path(__file__).resolve().parents[1] / "results" / "per_seed" / "hypertune" / "tune_report.md",
    )
    print(f"Generated 300 dpi PNG figures in: {out_dir}")


if __name__ == "__main__":
    main()
