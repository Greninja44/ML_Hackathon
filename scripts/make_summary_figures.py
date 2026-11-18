"""Summary figures for the team write-up, built only from saved results.

    python scripts/make_summary_figures.py --out <folder>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.visualization.plots import GRID, INK, INK_2, SERIES, _save, _style, grouped_bars  # noqa: E402

R = ROOT / "results"


def hmm_gain_per_trip(out: Path):
    w = pd.read_csv(R / "decoders/hmm_gain_per_trip.csv", dtype={"trip_id": str}).sort_values("hmm_gain")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [SERIES[1] if g < 0 else SERIES[0] for g in w["hmm_gain"]]
    ax.barh(w["trip_id"], 100 * w["hmm_gain"], color=colors, height=0.6)
    for y, g in enumerate(w["hmm_gain"]):
        ax.text(100 * g, y, f" {100 * g:+.1f}" if g >= 0 else f"{100 * g:+.1f} ", va="center",
                ha="left" if g >= 0 else "right", color=INK, fontsize=8)
    ax.axvline(0, color=INK_2, linewidth=1)
    _style(ax, "HMM gain in Top-1 per student-test trip (points)", "percentage points")
    ax.margins(x=0.2)
    _save(fig, out / "hmm_gain_per_trip.png")


def beta_curve(out: Path):
    g = pd.read_csv(R / "decoders/hmm_tuning_grid.csv").groupby("beta_m")["hmm_top1"].max()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(g.index, g.values, color=SERIES[0], linewidth=2, marker="o", markersize=7)
    ax.set_xscale("log")
    ax.set_xticks(g.index, [f"{b:g}" for b in g.index])
    _style(ax, "HMM tuning: best Top-1 by transition scale β (out-of-fold, training trips)", "β (metres)", "Top-1")
    ax.grid(axis="y", color=GRID)
    _save(fig, out / "hmm_tuning_beta.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "summary"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(R / "model_comparison.csv")
    grouped_bars(raw, ["cv_top1_mean", "student_top1", "student_hmm_top1"],
                 ["CV Top-1 (80 train trips)", "Student-test Top-1", "Student-test + HMM (untuned)"],
                 out / "model_comparison.png", "All eight models: accuracy")
    grouped_bars(raw, ["train_top1", "cv_top1_mean"], ["Training subset Top-1", "CV Top-1"],
                 out / "overfitting_gap.png", "Training vs cross-validation Top-1 (gap = overfitting)")
    tv = pd.read_csv(R / "tuned_vs_untuned.csv")
    grouped_bars(tv, ["cv_top1_mean", "student_top1", "student_macro_f1"], ["CV Top-1", "Student Top-1", "Student Macro F1"],
                 out / "tuned_vs_untuned.png", "HistGradientBoosting before and after Optuna")
    dec = pd.read_csv(R / "decoders/decoder_comparison.csv").rename(columns={"method": "model"})
    grouped_bars(dec, ["top1", "continuity", "seq_f1"], ["Top-1", "Route continuity", "Route F1"],
                 out / "decoder_methods.png", "Student-test: decoding methods")
    ab = pd.DataFrame({"model": ["A: Person 1 CSVs", "B: raw pipeline (final)"],
                       "clf": [pd.read_csv(R / "csv_pipeline/decoders/decoder_comparison.csv").set_index("method").loc["classifier_only", "top1"],
                               dec.set_index("model").loc["classifier_only", "top1"]],
                       "hmm": [pd.read_csv(R / "csv_pipeline/decoders/decoder_comparison.csv").set_index("method").loc["classifier_hmm", "top1"],
                               dec.set_index("model").loc["classifier_hmm", "top1"]]})
    grouped_bars(ab, ["clf", "hmm"], ["Classifier only", "Classifier + HMM"], out / "workflow_a_vs_b.png",
                 "Student-test Top-1: Person 1 CSV workflow vs raw pipeline")
    hmm_gain_per_trip(out)
    beta_curve(out)
    for src in ["eda/obs_per_trip.png", "eda/sampling_interval.png", "eda/gps_to_route_distance.png",
                "optuna/hist_gradient_boosting/optimization_history.png", "plots/cv_top1_folds.png",
                "models/hist_gradient_boosting_tuned/confusion_matrix.png", "models/hist_gradient_boosting_tuned/loss_curve.png"]:
        p = R / src
        if p.exists():
            shutil.copy(p, out / src.replace("/", "__"))
    print("figures written to", out)


if __name__ == "__main__":
    main()
