"""Report plots generated from saved experiment results (never hand-entered)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_metric(df: pd.DataFrame, metric: str, path: Path, title: str, err: str | None = None,
               lower_is_better: bool = False):
    d = df.dropna(subset=[metric]).sort_values(metric, ascending=lower_is_better)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.2))
    xerr = d[err] if err and err in d else None
    ax.barh(d["model"], d[metric], xerr=xerr, color=SERIES[0], height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    ax.invert_yaxis()
    for y, v in enumerate(d[metric]):
        ax.text(v, y, f"  {v:.4f}", va="center", color=INK, fontsize=8)
    _style(ax, title, metric)
    ax.margins(x=0.15)
    _save(fig, path)


def grouped_bars(df: pd.DataFrame, metrics: list[str], labels: list[str], path: Path, title: str):
    d = df.dropna(subset=metrics, how="all").sort_values(metrics[0], ascending=False)
    if d.empty:
        return
    y = np.arange(len(d))
    h = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(d) + 1.4))
    for k, (m, lab) in enumerate(zip(metrics, labels)):
        ax.barh(y + k * h, d[m], height=h * 0.9, color=SERIES[k], label=lab)
        for yi, v in zip(y + k * h, d[m]):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.3f}", va="center", color=INK, fontsize=7)
    ax.set_yticks(y + h * (len(metrics) - 1) / 2, d["model"])
    ax.invert_yaxis()
    _style(ax, title)
    ax.margins(x=0.12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=len(metrics),
              fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def confusion(cm, path: Path, title: str):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["not ref", "ref"])
    ax.set_yticks([0, 1], ["not ref", "ref"])
    ax.set_xlabel("predicted", color=INK_2)
    ax.set_ylabel("actual", color=INK_2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.set_facecolor(SURFACE)
    _save(fig, path)


def feature_importance(names, values, path: Path, title: str, top: int = 19):
    s = pd.Series(values, index=names).sort_values(key=np.abs, ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=SERIES[0], height=0.6)
    ax.invert_yaxis()
    _style(ax, title)
    _save(fig, path)


def loss_curves(curves: dict[str, list[float]], path: Path, title: str, ylabel: str):
    """Only called for models that actually record epoch/iteration histories."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, (lab, vals) in enumerate(curves.items()):
        if vals is not None and len(vals):
            ax.plot(np.arange(1, len(vals) + 1), vals, color=SERIES[k], linewidth=2, label=lab)
    _style(ax, title, "iteration / epoch", ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _save(fig, path)


def cv_boxplot(folds: pd.DataFrame, metric: str, path: Path, title: str):
    order = folds.groupby("model")[metric].mean().sort_values(ascending=False).index
    data = [folds.loc[folds["model"] == m, metric].to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(order) + 1.2))
    ax.boxplot(data, vert=False, tick_labels=list(order), widths=0.5,
               medianprops={"color": SERIES[1], "linewidth": 2},
               boxprops={"color": INK_2}, whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    ax.invert_yaxis()
    _style(ax, title, metric)
    _save(fig, path)
