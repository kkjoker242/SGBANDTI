from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "F1"]
COLORS = {
    "AUROC": "#1f77b4",
    "AUPRC": "#ff7f0e",
    "Accuracy": "#2ca02c",
    "Sensitivity": "#d62728",
    "Specificity": "#9467bd",
    "F1": "#8c564b",
}


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.major.size"] = 4.0
    plt.rcParams["ytick.major.size"] = 4.0


def parse_ablation_file(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8").splitlines()
    sections: list[dict[str, object]] = []
    i = 0
    while i < len(text):
        line = text[i].strip()
        if not line:
            i += 1
            continue
        if "\t" in line:
            header = line.split("\t")
            name = header[0]
            rows = []
            i += 1
            while i < len(text) and text[i].strip():
                parts = text[i].strip().split("\t")
                if len(parts) >= 7:
                    rows.append(
                        {
                            "x": parts[0],
                            "AUROC": float(parts[1]),
                            "AUPRC": float(parts[2]),
                            "F1": float(parts[3]),
                            "Sensitivity": float(parts[4]),
                            "Specificity": float(parts[5]),
                            "Accuracy": float(parts[6]),
                        }
                    )
                i += 1
            sections.append({"name": name, "rows": rows})
            continue
        if line.startswith("|"):
            header = [part.strip() for part in line.strip("|").split("|")]
            name = header[0]
            rows = []
            i += 2
            while i < len(text) and text[i].strip().startswith("|"):
                parts = [part.strip() for part in text[i].strip().strip("|").split("|")]
                if len(parts) >= 7:
                    rows.append(
                        {
                            "x": parts[0],
                            "AUROC": float(parts[1]),
                            "AUPRC": float(parts[2]),
                            "F1": float(parts[3]),
                            "Sensitivity": float(parts[4]),
                            "Specificity": float(parts[5]),
                            "Accuracy": float(parts[6]),
                        }
                    )
                i += 1
            sections.append({"name": name, "rows": rows})
            continue
        i += 1
    return sections


def normalize_sections(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    meta = {
        "Hop": {"title": "(a) k-hop graph", "ticks": ["0e+00", "1", "2", "3"]},
        "CNN_Embedding": {"title": "(b) CNN embedding sizes", "ticks": ["32", "64", "128", "256"]},
        "LR": {"title": "(c) Learning rate", "ticks": ["1e-04", "5e-05", "1e-05"]},
        "GCN_Layer": {"title": "(d) GCN layer number", "ticks": ["1", "2", "3", "4"]},
    }
    ordered = []
    for section in sections:
        name = str(section["name"])
        if name not in meta:
            continue
        rows = list(section["rows"])
        if name == "LR":
            rows = sorted(rows, key=lambda row: float(str(row["x"]).replace("e", "E")), reverse=True)
        else:
            rows = sorted(rows, key=lambda row: float(str(row["x"])))
        ordered.append({"name": name, "title": meta[name]["title"], "ticks": meta[name]["ticks"], "rows": rows})
    order = {"Hop": 0, "CNN_Embedding": 1, "LR": 2, "GCN_Layer": 3}
    ordered.sort(key=lambda item: order[str(item["name"])])
    return ordered


def save_all(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    base = out_png.with_suffix("")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_panel(ax: plt.Axes, section: dict[str, object], legend_loc: str | None) -> None:
    rows = section["rows"]
    x = np.arange(len(rows))
    for metric in METRICS:
        y = np.array([float(row[metric]) for row in rows])
        ax.plot(
            x,
            y,
            color=COLORS[metric],
            marker="o",
            linewidth=2.1,
            markersize=6.0,
            label=metric,
        )

    ax.set_title(str(section["title"]), fontsize=16, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(section["ticks"])
    ax.set_ylim(0.74, 0.93)
    ax.set_yticks([0.74, 0.78, 0.82, 0.86, 0.90, 0.93])
    ax.grid(True, which="major", axis="both", color="#CFCFCF", linestyle=(0, (4, 6)), linewidth=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    if legend_loc is not None:
        ax.legend(
            loc=legend_loc,
            fontsize=8.5,
            framealpha=0.85,
            facecolor="white",
            edgecolor="#BFBFBF",
            borderpad=0.6,
            handlelength=1.5,
            labelspacing=0.28,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-file",
        type=Path,
        default=Path(r"D:\毕设\Research_template\repo_audit\SGBANDTI-main\result\ablation.txt"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    apply_publication_style()
    sections = normalize_sections(parse_ablation_file(args.ablation_file))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axes = axes.ravel()
    legend_locs = ["lower right", "lower right", "lower left", "lower right"]
    for ax, section, legend_loc in zip(axes, sections, legend_locs):
        plot_panel(ax, section, legend_loc)
    fig.tight_layout(pad=1.5, w_pad=1.6, h_pad=2.0)
    save_all(fig, args.out)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
