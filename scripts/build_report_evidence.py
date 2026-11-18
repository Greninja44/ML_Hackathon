"""Build the figures, maps and checks the technical report needs.

    python scripts/build_report_evidence.py [--out report/evidence]

Uses only saved experiment results and the training / student-testing data.
Evaluator data is never read. Nothing is retrained.
"""
from __future__ import annotations

import argparse
import json
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

from src.visualization.plots import GRID, INK, INK_2, SERIES, SURFACE, _save, _style  # noqa: E402

R = ROOT / "results"
NICE = {"hist_gradient_boosting_tuned": "HistGradientBoosting (tuned, final)",
        "hist_gradient_boosting": "HistGradientBoosting", "random_forest": "Random Forest", "mlp": "MLP",
        "extra_trees": "Extra Trees", "decision_tree": "Decision Tree", "knn": "KNN",
        "logistic_regression": "Logistic Regression", "gaussian_nb": "Gaussian NB"}


# ------------------------------------------------------------------ helpers
def load_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, folds = [], []
    for d in sorted((R / "models").iterdir()):
        if not (d / "metrics.json").exists():
            continue
        s = json.loads((d / "metrics.json").read_text())["summary"]
        rows.append(s)
        f = pd.read_csv(d / "cv_folds.csv")
        f["model"] = d.name
        folds.append(f)
    df = pd.DataFrame(rows)
    df["label"] = df["model"].map(NICE)
    return df.sort_values("cv_top1_mean", ascending=False).reset_index(drop=True), pd.concat(folds)


def hbar(df, col, path, title, xlabel, err=None, fmt="{:.3f}", log=False, highlight=None, note=None):
    d = df.sort_values(col, ascending=not log if False else True if log else False)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.6))
    colors = [SERIES[1] if highlight and m == highlight else SERIES[0] for m in d["model"]]
    ax.barh(d["label"], d[col], color=colors, height=0.6, xerr=d[err] if err else None,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    ax.invert_yaxis()
    for y, v in enumerate(d[col]):
        ax.text(v, y, "  " + fmt.format(v), va="center", color=INK, fontsize=8)
    if log:
        ax.set_xscale("log")
    _style(ax, title, xlabel)
    ax.margins(x=0.18)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    _save(fig, path)


def grouped(df, cols, labels, path, title, xlabel, fmt="{:.3f}"):
    d = df.sort_values(cols[0], ascending=False)
    y = np.arange(len(d))
    h = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(d) + 1.8))
    for k, (c, lab) in enumerate(zip(cols, labels)):
        ax.barh(y + k * h, d[c], height=h * 0.9, color=SERIES[k], label=lab)
        for yi, v in zip(y + k * h, d[c]):
            ax.text(v, yi, " " + fmt.format(v), va="center", color=INK, fontsize=7)
    ax.set_yticks(y + h * (len(cols) - 1) / 2, d["label"])
    ax.invert_yaxis()
    _style(ax, title, xlabel)
    ax.margins(x=0.14)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(cols), fontsize=8,
              labelcolor=INK_2)
    _save(fig, path)


def hist(values, path, title, xlabel, ylabel="count", bins=50, log=False, vline=None, note=None, color=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(values, bins=bins, color=color or SERIES[0], edgecolor="white", linewidth=0.6, log=log)
    if vline is not None:
        ax.axvline(vline[0], color=SERIES[1], linewidth=1.5, linestyle="--")
        ax.text(vline[0], ax.get_ylim()[1] * 0.9, f" {vline[1]}", color=SERIES[1], fontsize=8, va="top")
    _style(ax, title, xlabel, ylabel)
    ax.grid(axis="y", color=GRID)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    _save(fig, path)


# ------------------------------------------------------------------ 1. model figures
def model_figures(out: Path, models: pd.DataFrame, folds: pd.DataFrame):
    o = out / "models"
    final = "hist_gradient_boosting_tuned"

    # A. CV Macro F1 with fold points, mean and sd
    d = models.sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(d) + 1.8))
    for y, m in enumerate(d["model"]):
        f = folds.loc[folds["model"] == m, "macro_f1"].to_numpy()
        mean, sd = f.mean(), f.std(ddof=1)
        ax.barh(y, mean, color=SERIES[1] if m == final else SERIES[0], height=0.6, alpha=0.9)
        ax.errorbar(mean, y, xerr=sd, color=INK, capsize=4, elinewidth=1.2)
        ax.scatter(f, np.full_like(f, y) + np.linspace(-0.18, 0.18, len(f)), s=14, color=INK, zorder=3)
        ax.text(1.002, y, f"{mean:.3f} ± {sd:.3f}", va="center", fontsize=8, color=INK,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(d)), d["label"])
    ax.invert_yaxis()
    ax.set_xlim(0.7, 1.0)
    _style(ax, "Cross-validation Macro F1 by model", "Macro F1 (5 trip-grouped folds, 80 training trips)")
    fig.text(0.01, 0.01, "Bar = mean, whisker = ±1 standard deviation, dots = individual folds. Orange = final model.",
             fontsize=7.5, color=INK_2)
    fig.subplots_adjust(right=0.8)
    _save(fig, o / "cv_macro_f1_comparison.png")

    # CV Top-1 (primary selection metric) the same way
    d = models.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(d) + 1.8))
    for y, m in enumerate(d["model"]):
        f = folds.loc[folds["model"] == m, "top1_accuracy"].to_numpy()
        ax.barh(y, f.mean(), color=SERIES[1] if m == final else SERIES[0], height=0.6)
        ax.errorbar(f.mean(), y, xerr=f.std(ddof=1), color=INK, capsize=4, elinewidth=1.2)
        ax.scatter(f, np.full_like(f, y) + np.linspace(-0.18, 0.18, len(f)), s=14, color=INK, zorder=3)
        ax.text(1.002, y, f"{f.mean():.3f} ± {f.std(ddof=1):.3f}", va="center", fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(d)), d["label"])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    _style(ax, "Cross-validation Top-1 accuracy by model", "Top-1 accuracy (primary selection metric)")
    fig.text(0.01, 0.01, "The low fold (~0.53) contains trips 00000054 and 00000033, whose GPS does not follow "
             "their reference route.", fontsize=7.5, color=INK_2)
    fig.subplots_adjust(right=0.8)
    _save(fig, o / "cv_top1_comparison.png")

    # B. Student-test Top-1 and Macro F1
    hbar(models, "student_top1", o / "student_test_top1_accuracy.png",
         "Student-test Top-1 accuracy", "Top-1 accuracy (classifier only, 10 trips, 19,160 points)",
         highlight=final)
    hbar(models, "student_macro_f1", o / "student_test_macro_f1.png",
         "Student-test Macro F1", "Macro F1 (candidate pairs, threshold 0.5)", highlight=final)
    grouped(models, ["student_top1", "student_hmm_top1"], ["Classifier only", "Classifier + HMM (default HMM)"],
            o / "student_test_top1_classifier_vs_hmm.png", "Student-test Top-1 with and without the HMM",
            "Top-1 accuracy")

    # C. Log loss
    grouped(models, ["cv_log_loss_mean", "student_log_loss"], ["Cross-validation", "Student test"],
            o / "log_loss_comparison.png", "Log loss by model", "log loss on candidate-pair probabilities (lower is better)")

    # F. extra comparisons
    grouped(models, ["student_precision", "student_recall", "student_f1"], ["Precision", "Recall", "F1"],
            o / "student_test_precision_recall.png", "Student-test precision and recall",
            "score for the 'correct road' class")
    grouped(models, ["train_top1", "cv_top1_mean"], ["Training trips (20-trip subset)", "Cross-validation"],
            o / "overfitting_train_vs_cv.png", "Training vs cross-validation Top-1",
            "Top-1 accuracy (a large gap means overfitting)")
    hbar(models, "train_time_s", o / "training_time.png", "Training time on all 80 training trips", "seconds (log scale)",
         fmt="{:.2f} s", log=True, highlight=final,
         note="KNN only stores a 150,000-row sample, so its cost shows up at prediction time instead.")
    hbar(models, "inference_ms_per_1k_pairs", o / "inference_time.png",
         "Inference time", "milliseconds per 1,000 candidate pairs (log scale)", fmt="{:.2f} ms",
         log=True, highlight=final)
    models = models.assign(model_size_mb=models["model_size_kb"] / 1024)
    hbar(models, "model_size_mb", o / "model_size.png", "Saved model size", "megabytes (log scale)", fmt="{:.2f} MB",
         log=True, highlight=final)

    # D. confusion matrices (existing, generated from student-test predictions during training)
    cm = o / "confusion_matrices"
    cm.mkdir(parents=True, exist_ok=True)
    for d in (R / "models").iterdir():
        if (d / "confusion_matrix.png").exists():
            shutil.copy(d / "confusion_matrix.png", cm / f"{d.name}.png")

    # E. genuine training histories only
    th = o / "training_histories"
    th.mkdir(parents=True, exist_ok=True)
    for name, files in {"mlp": ["loss_curve.png", "validation_curve.png"],
                        "hist_gradient_boosting": ["loss_curve.png"],
                        "hist_gradient_boosting_tuned": ["loss_curve.png"]}.items():
        for f in files:
            if (R / "models" / name / f).exists():
                shutil.copy(R / "models" / name / f, th / f"{name}__{f}")

    # feature importance: impurity-based for tree models (existing) + permutation for final model
    fi = o / "feature_importance"
    fi.mkdir(parents=True, exist_ok=True)
    for d in (R / "models").iterdir():
        if (d / "feature_importance.png").exists():
            shutil.copy(d / "feature_importance.png", fi / f"{d.name}.png")
    permutation_importance_final(fi / "hist_gradient_boosting_tuned_permutation.png")

    # Bayesian optimisation + tuned vs untuned + decoder comparison
    bo = o / "bayesian_optimisation"
    bo.mkdir(parents=True, exist_ok=True)
    for f in ("optimization_history.png", "param_importances.png"):
        p = R / "optuna/hist_gradient_boosting" / f
        if p.exists():
            shutil.copy(p, bo / f)
    tv = pd.read_csv(R / "tuned_vs_untuned.csv")
    tv["label"] = tv["model"].map(NICE)
    grouped(tv, ["cv_macro_f1_mean", "cv_top1_mean", "student_top1", "student_macro_f1"],
            ["CV Macro F1", "CV Top-1", "Student Top-1", "Student Macro F1"], bo / "tuned_vs_untuned.png",
            "HistGradientBoosting: untuned vs tuned", "score (same data and evaluation code)")
    hm = o / "hmm"
    hm.mkdir(parents=True, exist_ok=True)
    dec = pd.read_csv(R / "decoders/decoder_comparison.csv")
    dec["label"] = dec["method"].map({"nearest_arc": "Nearest road (no ML, no HMM)",
                                      "hmm_gaussian": "HMM, distance only (no ML)",
                                      "classifier_only": "Classifier only", "classifier_hmm": "Classifier + HMM (final)"})
    dec["model"] = dec["method"]
    grouped(dec, ["top1", "continuity", "seq_f1"], ["Top-1", "Route continuity", "Route (sequence) F1"],
            hm / "decoder_comparison.png", "Decoding methods on the student test", "score (final model)")
    w = pd.read_csv(R / "decoders/hmm_gain_per_trip.csv", dtype={"trip_id": str}).sort_values("hmm_gain")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(w["trip_id"], 100 * w["hmm_gain"], color=[SERIES[1] if g < 0 else SERIES[0] for g in w["hmm_gain"]],
            height=0.6)
    for y, g in enumerate(w["hmm_gain"]):
        ax.text(100 * g, y, f" {100 * g:+.1f} " if g >= 0 else f"{100 * g:+.1f} ", va="center",
                ha="left" if g >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.margins(x=0.25)
    _style(ax, "HMM gain per student-test trip", "change in Top-1 from adding the HMM (percentage points)")
    _save(fig, hm / "hmm_gain_per_trip.png")
    g = pd.read_csv(R / "decoders/hmm_tuning_grid.csv").groupby("beta_m")["hmm_top1"].max()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(g.index, g.values, color=SERIES[0], linewidth=2, marker="o", markersize=7)
    ax.set_xscale("log")
    ax.set_xticks(g.index, [f"{b:g}" for b in g.index])
    _style(ax, "HMM tuning: best Top-1 by β", "β (metres, log scale)",
           "Top-1 (out-of-fold, training trips)")
    ax.grid(axis="y", color=GRID)
    _save(fig, hm / "hmm_tuning_beta.png")


def permutation_importance_final(path: Path):
    from sklearn.inspection import permutation_importance

    from src.data.features import FEATURE_COLUMNS
    from src.models.inference import CandidateScorer
    scorer = CandidateScorer.load(ROOT / "models/final")
    pairs = pd.read_pickle(next((ROOT / "data/processed").glob("student_test_*_pairs.pkl")))
    sample = pairs.sample(n=min(40000, len(pairs)), random_state=42)
    r = permutation_importance(scorer.pipeline, sample[FEATURE_COLUMNS].to_numpy(float), sample["label"].to_numpy(),
                               scoring="neg_log_loss", n_repeats=5, random_state=42, n_jobs=-1)
    s = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean": r.importances_mean, "sd": r.importances_std}) \
        .sort_values("mean", ascending=False)
    s.to_csv(path.with_suffix(".csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(s) + 1.6))
    ax.barh(s["feature"], s["mean"], xerr=s["sd"], color=SERIES[0], height=0.6,
            error_kw={"ecolor": INK_2, "capsize": 2})
    ax.invert_yaxis()
    _style(ax, "Final model: permutation feature importance",
           "increase in log loss when the feature is shuffled")
    fig.text(0.01, 0.01, "40,000 student-test candidate pairs, 5 repeats. Analysis only; not used for any selection.",
             fontsize=7.5, color=INK_2)
    _save(fig, path)


# ------------------------------------------------------------------ 2. EDA
def eda_figures(out: Path):
    from src.config import load_config
    from src.data.io import iter_partition
    o = out / "eda"
    o.mkdir(parents=True, exist_ok=True)
    note = "Partition: training (80 trips)."
    v = pd.read_csv(R / "eda/validation.csv", dtype={"trip_id": str})
    ts = pd.read_csv(R / "eda/trip_summary.csv", dtype={"trip_id": str})
    tr_v, tr_s = v[v.partition == "train"], ts[ts.partition == "train"]
    hist(tr_s["length_km"], o / "trip_length_km.png", "Trip length", "kilometres (along GPS points)", bins=30, note=note)
    hist(tr_v["n_obs"], o / "gps_points_per_trip.png", "GPS points per trip", "GPS points", bins=30, note=note)

    pairs = pd.read_pickle(next((ROOT / "data/processed").glob("train_*_pairs.pkl")))
    obs = pd.read_pickle(next((ROOT / "data/processed").glob("train_*_obs.pkl")))
    vc = obs["n_candidates"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(vc.index.astype(str), vc.values, color=SERIES[0], width=0.6)
    for x, y in zip(range(len(vc)), vc.values):
        ax.text(x, y, f"{100 * y / vc.sum():.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    _style(ax, "Candidate road segments per GPS point", "candidates (≤ 8 within 50 m, 150 m fallback)", "GPS points")
    ax.grid(axis="y", color=GRID)
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    _save(fig, o / "candidates_per_point.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.linspace(0, 60, 61)
    ax.hist(pairs.loc[pairs.label == 0, "dist_m"].clip(upper=60), bins=bins, color=SERIES[0], alpha=0.85,
            label="other candidates", density=True)
    ax.hist(pairs.loc[pairs.label == 1, "dist_m"].clip(upper=60), bins=bins, color=SERIES[1], alpha=0.85,
            label="correct road", density=True, histtype="stepfilled")
    _style(ax, "GPS point to candidate road distance", "metres (clipped at 60)", "density")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    _save(fig, o / "candidate_distance_distribution.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.linspace(0, 180, 37)
    ax.hist(pairs.loc[pairs.label == 0, "heading_diff"], bins=bins, color=SERIES[0], alpha=0.85,
            label="other candidates", density=True)
    ax.hist(pairs.loc[pairs.label == 1, "heading_diff"], bins=bins, color=SERIES[1], alpha=0.85,
            label="correct road", density=True)
    _style(ax, "Vehicle heading vs road direction", "angle in degrees (0 = same direction)", "density")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.text(0.01, 0.01, note + " Heading from a ±2-point window; 90° is used when the vehicle is stationary.",
             fontsize=7.5, color=INK_2)
    _save(fig, o / "heading_difference_distribution.png")

    pts = pairs.drop_duplicates(["trip_id", "obs"])
    sp = pts["speed_mps"].to_numpy()
    hist(np.clip(sp * 3.6, 0, 250), o / "speed_distribution.png", "Vehicle speed", "km/h between consecutive GPS points (clipped at 250)", bins=50, log=True, vline=(180, "50 m/s outlier threshold"),
         note=f"{note} {int((sp > 50).sum()):,} points ({100 * (sp > 50).mean():.2f}%) exceed 50 m/s and are flagged, not removed.")

    cfg = load_config()
    dts, coverage = [], []
    for trip in iter_partition(cfg, "train"):
        dts.append(np.diff(trip.track["t"].to_numpy(float)))
    dt = np.concatenate(dts)
    hist(np.clip(dt, 0, 30), o / "timestamp_intervals.png", "Sampling interval", "seconds between consecutive GPS points (clipped at 30)",
         bins=30, log=True, note=f"{note} {100 * (dt == 1).mean():.1f}% of steps are exactly 1 s; longest gap {dt.max():,.0f} s.")

    checks = {"missing / non-finite values": int(tr_v["n_missing_values"].sum()),
              "out-of-range coordinates": int(tr_v["n_out_of_range"].sum()),
              "duplicate timestamps": int(tr_v["n_duplicate_timestamps"].sum()),
              "backward timestamps": int(tr_v["n_backward_timestamps"].sum()),
              "gaps over 10 s": int(tr_v["n_gaps_over_10s"].sum()),
              "arcs with bad node ids": int(tr_v["n_arcs_bad_node_ref"].sum()),
              "route arc ids out of range": int(tr_v["n_route_bad_ids"].sum())}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(list(checks), list(checks.values()), color=SERIES[0], height=0.6)
    for y, val in enumerate(checks.values()):
        ax.text(val, y, f"  {val:,}", va="center", fontsize=8, color=INK)
    ax.invert_yaxis()
    _style(ax, "Missing-value and data-quality checks", "count across the 80 training trips")
    ax.margins(x=0.15)
    _save(fig, o / "missing_values_and_quality.png")

    miss = 1 - obs.groupby("trip_id")["label_in_candidates"].mean()
    miss = miss.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(range(len(miss)), 100 * miss.values, color=[SERIES[1] if v > 0.5 else SERIES[0] for v in miss.values],
           width=0.8)
    flagged = [t for t, val in miss.items() if val > 0.5]
    if flagged:
        ax.text(len(flagged) + 0.5, 100 * miss.iloc[0] - 2, "  trips " + " and ".join(flagged), va="top",
                fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_xlim(-1, len(miss))
    _style(ax, "Candidate miss rate per training trip", "training trips, sorted by miss rate",
           "% of points missed")
    ax.grid(axis="y", color=GRID)
    fig.text(0.01, 0.01, f"Overall training miss rate {100 * (1 - obs['label_in_candidates'].mean()):.2f}%. "
             "Orange: two trips whose GPS lies ~120–140 m from the reference route.", fontsize=7.5, color=INK_2)
    _save(fig, o / "candidate_coverage_per_trip.png")

    feats = ["dist_m", "dist_minus_min", "heading_diff", "ctx_mean_dist_m", "prev_link", "speed_mps"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, f in zip(axes.ravel(), feats):
        lo, hi = np.nanpercentile(pairs[f], [0.5, 99.5])
        b = np.linspace(lo, hi, 40) if pairs[f].nunique() > 10 else np.arange(pairs[f].min() - 0.5, pairs[f].max() + 1)
        ax.hist(pairs.loc[pairs.label == 0, f].clip(lo, hi), bins=b, color=SERIES[0], alpha=0.8, density=True,
                label="other candidates")
        ax.hist(pairs.loc[pairs.label == 1, f].clip(lo, hi), bins=b, color=SERIES[1], alpha=0.8, density=True,
                label="correct road")
        _style(ax, f, "", "density")
        ax.title.set_fontsize(10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Key features: correct road vs other candidates (training pairs)", x=0.01, ha="left",
                 color=INK, fontsize=12)
    _save(fig, o / "feature_distributions.png")

    from src.data.features import FEATURE_COLUMNS
    c = pairs[FEATURE_COLUMNS].sample(n=min(200000, len(pairs)), random_state=42).corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(c)), c.columns, fontsize=8)
    for (i, j), val in np.ndenumerate(c.values):
        if abs(val) >= 0.5 and i != j:
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color="white" if abs(val) > 0.7 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson correlation")
    ax.set_title("Feature correlation (200,000 training pairs)", loc="left", color=INK)
    fig.set_facecolor(SURFACE)
    _save(fig, o / "feature_correlation.png")

    for f in ("sampling_interval.png", "network_vs_track.png", "gps_to_route_distance.png"):
        if (R / "eda" / f).exists():
            shutil.copy(R / "eda" / f, o / f"eda_script__{f}")


# ------------------------------------------------------------------ 3. maps
def maps(out: Path) -> dict:
    from src.config import load_config
    from src.data.io import iter_partition
    from src.data.labels import align_route
    from src.map_matching.reconstruction import RouteReconstructor
    from src.models.inference import CandidateScorer
    from src.visualization.folium_maps import matching_detail_map, network_map, save_map, trip_map
    o = out / "maps"
    o.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["hmm"] = json.loads((ROOT / "models/final/hmm_config.json").read_text())
    rec = RouteReconstructor(cfg, CandidateScorer.load(ROOT / "models/final"))

    # representative trip: closest to the student-test medians of points and HMM Top-1
    per = pd.read_csv(R / "decoders/decoder_comparison_per_trip.csv", dtype={"trip_id": str})
    per = per[per["method"] == "classifier_hmm"].copy()
    z = lambda s: (s - s.median()).abs() / s.std()  # noqa: E731
    per["score"] = z(per["n_obs"]) + z(per["top1"])
    rep = per.sort_values("score").iloc[0]
    tid = rep["trip_id"]
    trips = {t.trip_id: t for t in iter_partition(cfg, "student_test")}
    trip = trips[tid]
    r = rec.reconstruct(trip.inputs(), "hmm")
    m = trip_map(r.prepared.net, trip.track[["lon", "lat"]].to_numpy(), trip.route,
                 {"Matched route (classifier + HMM)": r.route_arcs},
                 title=f"Student-test trip {tid}: GPS, reference route and matched route")
    save_map(m, o / "representative_trip.html")

    # map-matching detail: window with the most classifier/HMM disagreements
    rc = rec.reconstruct(trip.inputs(), "classifier", prepared=r.prepared, proba=r.proba)
    diff = (rc.pred_arc != r.pred_arc).astype(int)
    prep, c = r.prepared, r.prepared.cands
    W = 40
    speed = np.nan_to_num(prep.track.speed_mps)
    n_cand = c.counts()
    win_diff = np.convolve(diff, np.ones(W), "valid")
    win_speed = np.convolve(speed, np.ones(W), "valid") / W
    win_cand = np.convolve(n_cand, np.ones(W), "valid") / W

    def detail(start, name, label):
        sl = slice(start, start + W)
        cand_arcs = [c.arc[c.for_obs(i)].tolist() for i in range(sl.start, sl.stop)]
        k = r.chosen[sl]
        a, b = prep.net.arc_a[c.arc[np.maximum(k, 0)]], prep.net.arc_b[c.arc[np.maximum(k, 0)]]
        snapped = prep.net.proj.to_lonlat(a + c.t_along[np.maximum(k, 0)][:, None] * (b - a))
        snapped[k < 0] = np.nan
        gps = np.column_stack([prep.track.lon[sl], prep.track.lat[sl]])
        m = matching_detail_map(prep.net, gps, snapped, cand_arcs, r.pred_arc[sl].tolist(),
                                rec.assemble_route(prep, r.pred_arc[sl]),
                                title=f"Trip {tid}, points {start}–{start + W - 1}: {label}")
        save_map(m, o / f"{name}.html")
        return {"window": [start, start + W - 1], "classifier_hmm_disagreements": int(diff[sl].sum()),
                "mean_speed_mps": round(float(speed[sl].mean()), 1), "mean_candidates": round(float(n_cand[sl].mean()), 1)}

    # main view: vehicle moving (> 8 m/s) with several candidates; most classifier/HMM disagreements among those
    ok = (win_speed > 8) & (win_cand >= 4)
    start = int(np.argmax(np.where(ok, win_diff + 0.01 * win_cand, -1)))
    moving = detail(start, "map_matching_detail", "candidate roads, selected roads and route (vehicle moving)")
    # extra view: the window with the most disagreements overall
    stationary = detail(int(np.argmax(win_diff)), "map_matching_most_disagreement",
                        "most classifier/HMM disagreements (GPS points clustered in one place)")

    # road network: the smallest student-test network, drawn completely
    small = min(trips.values(), key=lambda t: len(t.arcs))
    from src.data.prepare import build_network
    save_map(network_map(build_network(small.inputs(), cfg), title=f"Road network of student-test trip {small.trip_id}"),
             o / "road_network.html")

    p1 = o / "person1_trip_00000001"
    p1.mkdir(exist_ok=True)
    for f in (ROOT / "visualizations").iterdir():
        shutil.copy(f, p1 / f.name)
    return {"representative_trip": tid, "rep_points": int(rep["n_obs"]), "rep_top1_hmm": float(rep["top1"]),
            "student_median_points": float(per["n_obs"].median()), "student_median_top1": float(per["top1"].median()),
            "detail_moving": moving, "detail_most_disagreement": stationary,
            "network_trip": small.trip_id, "network_arcs": int(len(small.arcs)), "network_nodes": int(len(small.nodes))}


# ------------------------------------------------------------------ 4. verification
def verification(out: Path) -> dict:
    from sklearn.model_selection import GroupKFold

    from src.config import load_config
    from src.data.io import list_partition
    cfg = load_config()
    ids = {p: set(list_partition(cfg, p)) for p in ("train", "student_test", "eval_test")}
    pairs = pd.read_pickle(next((ROOT / "data/processed").glob("train_*_pairs.pkl")))
    groups = pairs["trip_id"].to_numpy()
    leaks = 0
    fold_trips = []
    for tr, va in GroupKFold(n_splits=cfg["training"]["cv_folds"]).split(pairs, groups=groups):
        leaks += len(set(groups[tr]) & set(groups[va]))
        fold_trips.append(sorted(set(groups[va])))
    models_ok = {}
    for d in (R / "models").iterdir():
        if (d / "metrics.json").exists():
            meta = ROOT / "models/experiments" / d.name / "metadata.json"
            models_ok[d.name] = {"metrics": True, "artifact": (ROOT / "models/experiments" / d.name / "model.joblib").exists(),
                                 "name_matches": json.loads(meta.read_text()).get("model_name") == d.name if meta.exists() else None,
                                 "student_trips_evaluated": int(len(pd.read_csv(d / "student_test_per_trip.csv")))}
    final_meta = json.loads((ROOT / "models/final/metadata.json").read_text())
    res = {
        "trip_ids_disjoint": {"train∩student_test": len(ids["train"] & ids["student_test"]),
                              "train∩eval_test": len(ids["train"] & ids["eval_test"]),
                              "student_test∩eval_test": len(ids["student_test"] & ids["eval_test"])},
        "training_rows_from": sorted(set(groups))[:3] + ["…"], "training_trips_in_pairs": int(len(set(groups))),
        "training_pairs_include_student_or_eval_trips": bool(set(groups) & (ids["student_test"] | ids["eval_test"])),
        "groupkfold_trips_in_both_train_and_val": int(leaks), "fold_sizes_trips": [len(f) for f in fold_trips],
        "models": models_ok,
        "final_model": {"name": final_meta.get("model_name"), "train_trips": final_meta.get("train_trips"),
                        "train_rows": final_meta.get("train_rows")},
        "label_definition": "per-point reference arc from monotone alignment of the .route sequence; "
                            "pair label = candidate arc equals that arc (exactly one positive per covered point)",
        "max_positives_per_point": int(pairs.groupby(["trip_id", "obs"])["label"].sum().max()),
        "evaluator_data_used_for_training_or_tuning": False,
        "evaluator_data_note": "iter_partition() refuses eval_test for development code (tested); "
                               "eval trips only passed through the input-only check, labels never read",
    }
    (out / "verification.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "report/evidence"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    models, folds = load_models()
    models.drop(columns=["label"]).to_csv(out / "model_results_table.csv", index=False)
    model_figures(out, models, folds)
    print("model figures done")
    eda_figures(out)
    print("EDA figures done")
    info = maps(out)
    (out / "maps" / "maps_info.json").write_text(json.dumps(info, indent=2))
    print("maps done:", info)
    v = verification(out)
    print("verification:", json.dumps({k: v[k] for k in ("trip_ids_disjoint", "groupkfold_trips_in_both_train_and_val",
                                                          "training_pairs_include_student_or_eval_trips",
                                                          "max_positives_per_point")}))


if __name__ == "__main__":
    main()
