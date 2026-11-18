# PathSnap model evaluation report

Every number here is read from files in `results/` that the scripts generated on
2026-11-18. Nothing was entered by hand. Re-run with the commands in the README.
Data: **80 training trips** (fitting, GroupKFold CV, Optuna, decoder tuning) and
**10 student-testing trips** (comparison only). No evaluator data was used.

The task is binary classification of (GPS point, candidate arc) pairs, with the
positive class = the aligned reference arc. Each observation's prediction is the
highest-probability candidate (classifier only) or the Viterbi path (classifier + HMM).

* **Primary metric (fixed before the experiments):** mean CV observation-level
  **Top-1 accuracy**. It directly measures whether each GPS point is placed on
  the right road. Pair-level accuracy is dominated by easy negatives and depends
  on a 0.5 threshold that is irrelevant once candidates are ranked.
* **Macro F1** = mean of the F1 for the "not reference" and "reference" classes on candidate pairs.
* **Route continuity** = share of consecutive distinct predicted arcs that are directly connected.
* **Sequence F1** = F1 between the set of predicted route arcs and the reference route arcs.

---

## 1. Workflow B: raw data (production pipeline used by the API)

Candidates: ≤ 8 arcs within 50 m (150 m fallback), on true geometry. Candidate
miss rate: train 9.21 %, student-test **0.42 %**. Features: 19 (`src/data/features.py`).

### 1.1 Model comparison (`results/model_comparison.csv`)

| Model | Family | CV Macro F1 | CV Top-1 (mean ± sd) | Train Top-1* | Student Top-1 | Student Top-1 + HMM† | Route continuity (clf → +HMM†) | Student Macro F1 | Student log loss | Inference ms / 1k pairs | Train s | Size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **HistGradientBoosting** | Boosting | 0.908 | **0.772 ± 0.204** | 0.895 | 0.877 | 0.877 | 0.768 → 0.910 | 0.925 | 0.112 | 2.08 | 4.8 | 216 KB |
| Random Forest | Ensemble (bagging) | **0.911** | 0.771 ± 0.208 | 0.944 | **0.884** | 0.877 | 0.792 → 0.904 | **0.929** | **0.101** | 4.90 | 95.2 | 193 MB |
| MLP | Neural network | 0.889 | 0.768 ± 0.208 | 0.900 | 0.882 | 0.875 | 0.788 → 0.904 | 0.927 | 0.107 | 0.13 | 46.6 | 83 KB |
| Extra Trees | Ensemble (random splits) | 0.906 | 0.767 ± 0.207 | 0.907 | 0.876 | 0.869 | 0.778 → 0.904 | 0.923 | 0.109 | 6.00 | 29.4 | 290 MB |
| Decision Tree | Tree | 0.890 | 0.765 ± 0.207 | 0.897 | 0.878 | 0.875 | 0.769 → 0.901 | 0.923 | 0.131 | 0.08 | 8.3 | 94 KB |
| KNN (k = 25, distance) | Instance-based | 0.894 | 0.757 ± 0.206 | 0.899 | 0.865 | 0.871 | 0.757 → 0.891 | 0.917 | 0.264 | 56.5 | 0.01‡ | 11 MB |
| Logistic Regression | Linear | 0.868 | 0.747 ± 0.200 | 0.854 | 0.855 | 0.855 | 0.740 → 0.906 | 0.899 | 0.165 | 0.08 | 0.5 | 2 KB |
| Gaussian NB | Probabilistic | 0.798 | 0.721 ± 0.192 | 0.827 | 0.809 | 0.859 | 0.767 → 0.906 | 0.805 | 0.841 | 0.11 | 0.1 | 3 KB |

\* Top-1 on a fixed random subset of 20 training trips, used as overfitting evidence.
† Default, untuned HMM (`hybrid`, w = 1.0, σ = 8 m, β = 5 m). The tuned decoder is in §1.3.
‡ KNN "training" only stores a 150k-row subsample; its cost is at prediction time.

Plots: `results/plots/` (CV Macro F1, CV Top-1 with error bars and per-fold box
plot, student Top-1 vs Macro F1, log loss, classifier vs HMM, continuity,
inference time). Per-model confusion matrices and feature importance are in
`results/models/<model>/`. Loss curves exist **only** for MLP and HistGradientBoosting.

**Observations**

* The models are close. The top five are within 0.7 points of CV Top-1, much
  less than the fold-to-fold spread (± 0.20).
* That spread is a data effect, not model instability. Fold 0 contains trips
  00000054 and 00000033, whose GPS does not follow the reference route: its
  candidate miss rate is 39 % and its Top-1 is 0.53, while the other folds score
  0.79–0.93 (`results/models/hist_gradient_boosting_tuned/cv_folds.csv`).
* **Random Forest overfits most.** Its training-subset Top-1 is 0.944 against
  0.771 in CV, and it is 193 MB. HistGradientBoosting has a smaller train/CV gap
  (0.895 vs 0.772), is 900× smaller and is 2.4× faster at inference.
* Gaussian NB is the weakest classifier (its feature-independence assumption is
  violated; for example, the distance features are strongly correlated), but it
  gains the most from the HMM (+5.0 points).
* **Selected:** HistGradientBoosting, by the pre-declared rule (highest CV
  Top-1; `results/model_selection.json`). Random Forest is a near-tie, but its
  size and overfitting make it the worse deployment choice.

### 1.2 Bayesian optimisation (`results/optuna/hist_gradient_boosting/`)

* Optuna TPE sampler (seed 42), MedianPruner. The objective was mean CV Top-1 (GroupKFold by trip, 80 trips).
* **30 trials** (13 complete, 17 pruned), 627 s.
* Search space: `learning_rate` 0.01–0.3 (log), `max_leaf_nodes` 8–128 (log),
  `min_samples_leaf` 5–200 (log), `l2_regularization` 1e-8–10 (log),
  `max_iter` 100–800.
* Best: `learning_rate` = 0.156, `max_leaf_nodes` = 18, `min_samples_leaf` = 7,
  `l2_regularization` = 0.0144, `max_iter` = 400 (early stopping on).

| (`results/tuned_vs_untuned.csv`) | CV Macro F1 | CV Top-1 | Student Top-1 | Student Top-1 + HMM† | Student Macro F1 | Student log loss | Inference ms / 1k pairs | Train s |
|---|---|---|---|---|---|---|---|---|
| Untuned | 0.9078 | 0.7716 ± 0.204 | 0.8770 | 0.8771 | 0.9245 | 0.1122 | 2.08 | 4.8 |
| **Tuned (final)** | 0.9077 | **0.7929 ± 0.157** | 0.8775 | 0.8755 | 0.9252 | 0.1079 | 1.15 | 3.4 |

Tuning raised CV Top-1 by 2.1 points and reduced the fold spread, mostly by doing
better on the hard fold. On student-test, Top-1 is essentially unchanged
(+0.04 points) and log loss improves slightly. The tuned model is final (rule:
tuned if CV Top-1 ≥ untuned): `models/final/`.

### 1.3 HMM / Viterbi vs classifier-only (`results/decoders/`)

Decoder parameters were tuned by grid search on **out-of-fold** probabilities for
20 training trips (`hmm_tuning_grid.csv`, 45 configurations). The best β
increases with the grid but flattens (β = 2/5/10/20/40 → best Top-1
0.813/0.829/0.836/0.840/0.840). The search therefore stopped at β = 40 m with a
hybrid emission (w = 0.5, σ = 12 m).

| Student-test (19,160 points) | Top-1 | Route continuity | Sequence F1 | Seq. precision | Seq. recall | Edit similarity | Decode ms / trip |
|---|---|---|---|---|---|---|---|
| Nearest arc (no ML, no HMM) | 0.544 | 0.278 | 0.616 | 0.456 | 0.986 | 0.223 | 2 |
| HMM, distance only (no ML) | 0.856 | 0.904 | 0.921 | 0.866 | 0.989 | 0.736 | 109 |
| Classifier only | 0.878 | 0.780 | 0.814 | 0.700 | 0.993 | 0.392 | 1 |
| **Classifier + HMM** | **0.885** | 0.895 | **0.930** | **0.879** | 0.991 | **0.759** | 103 |

* The HMM raises Top-1 by **0.7 points** and fixes route coherence: continuity
  +11.5 points, sequence precision +17.9 points, edit similarity +36.7 points.
  The classifier alone produces many one-point jumps to wrong arcs. They barely
  hurt Top-1 but they fragment the route.
* Per trip (`hmm_gain_per_trip.csv`), the HMM helps **9 of 10** trips (by up to
  +3.1 points). **Trip 00000007 is hurt by 5.0 points**. That trip has many
  segment breaks under the decoder (34 with the untuned settings), meaning
  consecutive candidates were not connected within the transition search bound.
  It is the main case where HMM decoding did not help and is worth inspecting on
  its map (`results/decoders/maps/00000007.html`).
* The distance-only HMM already beats nearest-arc by 31 points. That shows how
  much the sequence information adds before any ML.

---

## 2. Workflow A: Person 1's CSVs

Person 1's candidates (≤ 5 within 50 m), 16 features from the CSV columns
(`src/csv_pipeline/features.py`), labels by route alignment on CSV distances.
Candidate miss rate: train 13.73 %, student-test 3.21 %. Results: `results/csv_pipeline/`.

### 2.1 Model comparison (`results/csv_pipeline/model_comparison.csv`)

| Model | CV Macro F1 | CV Top-1 (mean ± sd) | CV Top-1, flagged trips excl.§ | Train Top-1* | Student Top-1 | + topology HMM† | Continuity (clf → +HMM†) | Student Macro F1 | Student log loss | ms / 1k pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| **MLP** | 0.856 | **0.714 ± 0.160** | 0.765 | 0.790 | **0.803** | 0.871 | 0.754 → 0.862 | 0.877 | 0.202 | 0.10 |
| HistGradientBoosting | 0.858 | 0.714 ± 0.162 | 0.762 | 0.794 | 0.803 | 0.862 | 0.743 → 0.867 | 0.879 | 0.205 | 0.76 |
| Random Forest | **0.860** | 0.713 ± 0.162 | 0.762 | 0.878 | 0.801 | 0.862 | 0.751 → 0.872 | **0.880** | **0.199** | 3.53 |
| Extra Trees | 0.853 | 0.712 ± 0.161 | 0.760 | 0.811 | 0.800 | 0.871 | 0.755 → 0.877 | 0.880 | 0.205 | 4.25 |
| Decision Tree | 0.853 | 0.710 ± 0.162 | 0.756 | 0.792 | 0.800 | **0.873** | 0.744 → 0.852 | 0.877 | 0.242 | 0.04 |
| KNN | 0.846 | 0.708 ± 0.160 | 0.757 | 0.837 | 0.798 | 0.865 | 0.764 → 0.845 | 0.874 | 0.353 | 31.97 |
| Logistic Regression | 0.835 | 0.706 ± 0.163 | 0.749 | 0.776 | 0.796 | 0.851 | 0.736 → 0.877 | 0.858 | 0.255 | 0.03 |
| Gaussian NB | 0.753 | 0.673 ± 0.155 | 0.710 | 0.743 | 0.756 | 0.843 | 0.759 → 0.788 | 0.763 | 1.018 | 0.06 |

§ Sensitivity only, never used for selection: the mean over the CV fold(s) that
contain the flagged trips (> 50 % alignment misses: 00000054, 00000033), computed
with those trips removed.
\* 20-trip training subset. † Untuned topology HMM (defaults in `src/csv_pipeline/hmm.py`).

Selected: **MLP** (highest CV Top-1, pre-declared rule). The top five models are
within 0.3 points, far inside the fold spread.

### 2.2 Bayesian optimisation (MLP, `results/csv_pipeline/optuna/mlp/`)

27 trials (14 complete, 13 pruned) before the 1-hour time limit (3,637 s). Best:
hidden layers (256, 128, 64), α = 1.0e-5, learning rate 1.07e-3.

| | CV Macro F1 | CV Top-1 | Student Top-1 | + HMM | Student Macro F1 | Student log loss | ms / 1k pairs | Train s |
|---|---|---|---|---|---|---|---|---|
| Untuned | 0.8560 | 0.7143 ± 0.160 | 0.8027 | 0.8713 | 0.8774 | 0.2018 | 0.10 | 12.2 |
| Tuned | 0.8555 | 0.7156 ± 0.161 | 0.8027 | 0.8713 | 0.8763 | 0.2022 | 0.49 | 52.9 |

**Tuning did not meaningfully improve the CSV model.** CV Top-1 rose by
0.13 points (within noise), student-test is identical, and the tuned network
is 4.8× slower at inference. The rule ("tuned if CV Top-1 ≥ untuned") still
selected it, which shows the limit of a ≥ rule. A minimum-improvement margin would
have kept the untuned model. Worth saying plainly in the report.

### 2.3 Topology-only HMM vs classifier-only (`results/csv_pipeline/decoders/`)

Tuned on out-of-fold probabilities for 30 training trips (36 configurations). Best: hybrid
emission (w = 0.7, σ = 8 m), c_next = 1.5, c_jump = 4, c_uturn = 10.

| Student-test | Top-1 | Route continuity | Sequence F1 | Edit similarity |
|---|---|---|---|---|
| Nearest arc | 0.527 | 0.614 | 0.618 | 0.289 |
| HMM, distance only | 0.862 | 0.849 | 0.856 | 0.758 |
| Classifier only | 0.803 | 0.770 | 0.792 | 0.470 |
| **Classifier + HMM** | **0.880** | 0.839 | **0.871** | **0.768** |

The HMM adds **7.7 points** of Top-1 here and helps 8 of 10 trips. It hurts trip
00000007 (−5.0 points, the same trip as in Workflow B) and 00000052 (−0.4 points).
With only Person 1's 15 columns, the classifier alone is *worse* than a
distance-only HMM (0.803 vs 0.862). Sequence information matters more than the
per-point model on this feature set.

### 2.4 Workflow A vs Workflow B (student-test, final configurations)

| | Candidate miss rate | Classifier-only Top-1 | Classifier + HMM Top-1 | Sequence F1 (+HMM) |
|---|---|---|---|---|
| A: Person 1 CSVs (MLP, topology HMM) | 3.21 % | 0.803 | 0.880 | 0.871 |
| B: raw pipeline (HGB, distance HMM) | 0.42 % | 0.878 | **0.885** | **0.930** |

Workflow B is used by the API. Its gains come mainly from better candidate recall
and richer context features (classifier-only +7.5 points), and from the
geometry-based HMM producing a more faithful route (sequence F1 +5.9 points).

---

## 3. Limitations

* Labels are derived by route alignment (stable under changed assumptions:
  ≥ 98.4 % agreement). The reference route itself is not guaranteed to be error-free (DE §6).
* Trips 00000054 and 00000033 are kept, so CV numbers include points whose
  reference arc is ~120–140 m from the GPS.
* Route continuity and sequence metrics are our own definitions.
* Student-test results come from 10 trips. Differences under ~1 point between models are within noise.
