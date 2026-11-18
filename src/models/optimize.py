"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model.

    python -m src.models.optimize                    # model from results/model_selection.json
    python -m src.models.optimize --model random_forest --trials 60

* Sampler: Tree-structured Parzen Estimator (a sequential model-based /
  Bayesian optimisation method), seeded for reproducibility.
* Objective: maximise mean observation-level Top-1 accuracy over trip-grouped
  CV folds of the 80 training trips (same primary metric as model selection).
* Pruning: MedianPruner on the running fold mean.
* The tuned model is refitted on all training trips and evaluated on the
  student-testing trips with exactly the same code as the untuned model.
* Final model = tuned if its CV Top-1 ≥ untuned CV Top-1, otherwise untuned.
  The decision uses CV only; student-testing numbers are reported, and the
  evaluator partition is never read.

Outputs: results/optuna/<model>/{trials.csv, best_params.json, study_summary.json,
optimization_history.png, param_importances.png}, results/tuned_vs_untuned.csv,
models/final/.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.experiments.trip_eval import load_labelled_trips  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.models.dataset import build_partition  # noqa: E402
from src.models.registry import default_specs  # noqa: E402
from src.models.train import cross_validate, run_model  # noqa: E402

log = logging.getLogger("optimize")


def search_space(name: str, trial: optuna.Trial, base: dict) -> dict:
    """Justified ranges: centred on the defaults, wide enough to cover under- and over-fitting regimes."""
    p = dict(base)
    if name == "logistic_regression":
        p["C"] = trial.suggest_float("C", 1e-3, 1e2, log=True)
        p["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
    elif name == "gaussian_nb":
        p["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True)
    elif name == "knn":
        p["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 101, log=True)
        p["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        p["p"] = trial.suggest_categorical("p", [1, 2])
    elif name == "decision_tree":
        p["max_depth"] = trial.suggest_int("max_depth", 3, 30)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 200, log=True)
        p["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
    elif name in ("random_forest", "extra_trees"):
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 600, step=50)
        p["max_depth"] = trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24])
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 50, log=True)
        p["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8])
    elif name == "hist_gradient_boosting":
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        p["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 8, 128, log=True)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 200, log=True)
        p["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True)
        p["max_iter"] = trial.suggest_int("max_iter", 100, 800, step=50)
    elif name == "mlp":
        width = trial.suggest_categorical("width", [32, 64, 128, 256])
        depth = trial.suggest_int("depth", 1, 3)
        p["hidden_layer_sizes"] = tuple(max(width // (2 ** i), 8) for i in range(depth))
        p["alpha"] = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
        p["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    elif name == "xgboost":
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 800, step=50)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        p["min_child_weight"] = trial.suggest_float("min_child_weight", 1e-2, 20, log=True)
    elif name == "lightgbm":
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        p["num_leaves"] = trial.suggest_int("num_leaves", 8, 256, log=True)
        p["n_estimators"] = trial.suggest_int("n_estimators", 100, 800, step=50)
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 200, log=True)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    else:
        raise ValueError(f"No search space defined for {name}")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--timeout", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging("optimize", cfg["paths"]["results_dir"])
    results_dir = resolve(cfg["paths"]["results_dir"])
    name = args.model
    if not name:
        sel = results_dir / "model_selection.json"
        if not sel.exists():
            raise SystemExit("Run `python -m src.models.train` first or pass --model")
        name = json.loads(sel.read_text())["selected_model"]
    spec = default_specs(cfg["training"]["n_jobs"])[name]
    n_trials = args.trials or cfg["optuna"]["n_trials"]
    timeout = args.timeout or cfg["optuna"]["timeout_s"]

    train = build_partition(cfg, "train")
    out = results_dir / "optuna" / name
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        params = search_space(name, trial, spec.params)
        folds = cross_validate(spec, params, train.pairs, train.obs, cfg, trial=trial)
        trial.set_user_attr("cv_macro_f1", float(folds["macro_f1"].mean()))
        trial.set_user_attr("cv_log_loss", float(folds["log_loss"].mean()))
        trial.set_user_attr("cv_top1_std", float(folds["top1_accuracy"].std()))
        return float(folds["top1_accuracy"].mean())

    study = optuna.create_study(
        study_name=f"pathsnap_{name}", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=min(10, n_trials)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    hpo_s = time.perf_counter() - t0

    trials = study.trials_dataframe()
    trials.to_csv(out / "trials.csv", index=False)
    best_params = search_space(name, optuna.trial.FixedTrial(study.best_params), spec.params)
    (out / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    states = trials["state"].value_counts().to_dict() if len(trials) else {}
    summary = {"model": name, "sampler": "TPE", "pruner": "MedianPruner",
               "objective": "mean CV observation-level Top-1 accuracy (GroupKFold by trip)",
               "n_trials": len(study.trials), "trial_states": states, "best_value": study.best_value,
               "best_trial": study.best_trial.number, "best_params": best_params,
               "optimisation_wall_time_s": hpo_s}
    (out / "study_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        ax = plot_optimization_history(study)
        ax.figure.set_size_inches(8, 4.5)
        ax.figure.tight_layout()
        ax.figure.savefig(out / "optimization_history.png", dpi=150)
        if len(study.trials) >= 3 and len(study.best_params) > 1:
            ax = plot_param_importances(study)
            ax.figure.tight_layout()
            ax.figure.savefig(out / "param_importances.png", dpi=150)
    except Exception:
        log.exception("Could not draw Optuna plots")

    # ---- tuned vs untuned on identical data partitions and code paths
    student = load_labelled_trips(cfg, "student_test")
    untuned_metrics = results_dir / "models" / name / "metrics.json"
    if untuned_metrics.exists():
        untuned = json.loads(untuned_metrics.read_text())["summary"]
    else:
        untuned = run_model(spec, dict(spec.params), train, student, cfg, name)
    tuned = run_model(spec, best_params, train, student, cfg, f"{name}_tuned")
    cols = ["model", "cv_macro_f1_mean", "cv_top1_mean", "cv_top1_std", "student_top1", "student_hmm_top1",
            "student_macro_f1", "student_log_loss", "inference_ms_per_1k_pairs", "train_time_s"]
    comp = pd.DataFrame([untuned, tuned])[cols]
    comp.to_csv(results_dir / "tuned_vs_untuned.csv", index=False)
    log.info("\n%s", comp.to_string(index=False))

    use_tuned = tuned["cv_top1_mean"] >= untuned["cv_top1_mean"]
    chosen = f"{name}_tuned" if use_tuned else name
    final_dir = resolve(cfg["paths"]["final_model_dir"])
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(resolve(cfg["paths"]["models_dir"]) / "experiments" / chosen, final_dir)
    decision = {"final_model": chosen, "rule": "tuned if CV Top-1 >= untuned CV Top-1",
                "untuned_cv_top1": untuned["cv_top1_mean"], "tuned_cv_top1": tuned["cv_top1_mean"],
                "hmm_config": cfg["hmm"], "candidates_config": cfg["candidates"]}
    (final_dir / "final_model.json").write_text(json.dumps(decision, indent=2, default=float))
    log.info("Final model: %s -> %s", chosen, final_dir)


if __name__ == "__main__":
    main()
