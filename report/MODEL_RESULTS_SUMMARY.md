# Model results summary

All numbers come from `report/evidence/model_results_table.csv`, which the
training scripts wrote (`results/models/*/metrics.json`), and from
`results/decoders/` and `results/tuned_vs_untuned.csv`. Nothing here was typed in by hand.

**Data used:** 80 training trips for fitting, 5-fold cross-validation grouped by
trip, Optuna and HMM tuning. 10 student-testing trips (19,160 GPS points) for
comparison only. Evaluator trips were never used.

**What is predicted:** for every (GPS point, candidate road) pair, the probability
that the candidate is the point's reference road. Per-point reference roads come
from aligning the trip's `.route` sequence to its GPS points in order (≥ 98.4 %
of labels unchanged under different alignment assumptions). Each point has at
most one positive candidate (checked in `evidence/verification.json`).

**Metrics:**
* **Top-1 accuracy** (primary): share of GPS points whose highest-scoring candidate
  is the reference road. Points whose reference road is not a candidate count as wrong.
* **Macro F1, precision, recall, log loss, confusion matrix:** on candidate pairs,
  threshold 0.5. Precision and recall are for the "correct road" class.
* **Route continuity** and **route (sequence) F1:** how well the decoded route holds
  together and matches the reference arcs. These are our own definitions (see `docs/methodology.md`).

## 1. All models

| Model | Family | CV Macro F1 (mean ± sd) | CV Top-1 (mean ± sd) | CV log loss | Student Top-1 | Student Macro F1 | Student precision / recall | Student log loss | Train time | Inference (ms / 1k pairs) | Size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **HistGradientBoosting, tuned (final)** | Boosting | 0.908 ± 0.040 | **0.793 ± 0.157** | 0.142 | 0.878 | 0.925 | 0.883 / 0.868 | 0.108 | 3.4 s | 1.15 | 143 KB |
| HistGradientBoosting | Boosting | 0.908 ± 0.040 | 0.772 ± 0.204 | 0.142 | 0.877 | 0.925 | 0.880 / 0.869 | 0.112 | 4.8 s | 2.08 | 216 KB |
| Random Forest | Ensemble (bagging) | **0.911 ± 0.040** | 0.771 ± 0.208 | **0.123** | **0.884** | **0.929** | 0.894 / 0.871 | **0.101** | 95.2 s | 4.90 | 193 MB |
| MLP | Neural network | 0.889 ± 0.067 | 0.768 ± 0.208 | 0.168 | 0.882 | 0.927 | 0.884 / 0.872 | 0.107 | 46.6 s | 0.13 | 83 KB |
| Extra Trees | Ensemble (random splits) | 0.906 ± 0.040 | 0.767 ± 0.207 | 0.129 | 0.876 | 0.923 | 0.900 / 0.845 | 0.109 | 29.4 s | 6.00 | 290 MB |
| Decision Tree | Tree | 0.890 ± 0.066 | 0.765 ± 0.207 | 0.196 | 0.878 | 0.923 | 0.880 / 0.861 | 0.131 | 8.3 s | 0.08 | 94 KB |
| KNN (k = 25, distance-weighted) | Instance-based | 0.894 ± 0.043 | 0.757 ± 0.206 | 0.401 | 0.865 | 0.917 | 0.880 / 0.844 | 0.264 | 0.01 s* | 56.5 | 11 MB |
| Logistic Regression | Linear | 0.868 ± 0.050 | 0.747 ± 0.200 | 0.217 | 0.855 | 0.899 | 0.845 / 0.817 | 0.165 | 0.5 s | 0.08 | 2 KB |
| Gaussian NB | Probabilistic | 0.798 ± 0.032 | 0.721 ± 0.192 | 0.805 | 0.809 | 0.805 | 0.562 / 0.912 | 0.841 | 0.1 s | 0.11 | 3 KB |

\* KNN only stores a 150,000-row sample; its cost appears at prediction time.

All models produce probabilities, so log loss applies to all of them.

## 2. Best model and why

**HistGradientBoosting, tuned with Optuna.**

1. **The rule was fixed before the experiments:** choose the highest
   cross-validated Top-1. Untuned, HistGradientBoosting won (0.772), with
   Random Forest a near-tie (0.771).
2. **Random Forest was the alternative, and we rejected it.** It is slightly better on the
   student-test trips (0.884 vs 0.878) and on CV Macro F1, but:
   * it overfits more: Top-1 on training trips 0.944 vs 0.771 in CV, against 0.902 vs 0.793 for the final model;
   * it is about 1,400 times larger (193 MB vs 143 KB) and 4 times slower at inference.
   Choosing it because of the 10 student-test trips would be selecting on test data.
3. **Tuning helped where it should.** CV Top-1 rose from 0.772 to 0.793 and the fold
   spread narrowed (±0.204 to ±0.157). Student-test was essentially unchanged
   (0.877 to 0.878).
4. **Its probabilities are good.** Log loss is among the lowest, which matters because the HMM uses them.

## 3. Bayesian optimisation

* Optuna TPE sampler (seed 42), MedianPruner, objective = mean CV Top-1 (GroupKFold by trip).
* 30 trials (13 completed, 17 pruned), 627 s.
* Search space: learning rate 0.01–0.3 (log), max leaf nodes 8–128 (log), min
  samples per leaf 5–200 (log), L2 regularisation 1e-8–10 (log), max iterations 100–800.
* Best: learning rate 0.156, 18 leaf nodes, 7 samples per leaf, L2 0.0144, 400 iterations, early stopping.

## 4. Classifier + HMM (final configuration, student test)

HMM settings were tuned on out-of-fold probabilities for 20 training trips:
emission = 0.5 × classifier log-probability + 0.5 × Gaussian distance term (σ = 12 m),
transition scale β = 40 m.

| Method | Top-1 | Route continuity | Route F1 |
|---|---|---|---|
| Nearest road (no ML, no HMM) | 0.544 | 0.278 | 0.616 |
| HMM with distance only (no ML) | 0.856 | 0.904 | 0.921 |
| Classifier only (final model) | 0.878 | 0.780 | 0.814 |
| **Classifier + HMM (final system)** | **0.885** | **0.895** | **0.930** |

The HMM helps 9 of 10 trips. Trip 00000007 loses 5.0 points (classifier 0.951,
with HMM 0.900).

The "Student-test + HMM" column in `models/student_test_top1_classifier_vs_hmm.png`
uses the *default, untuned* HMM for every model. It is a model-to-model comparison,
not the final system's score.

## 5. Important observations

* **The models are close.** The top five are within 0.7 points of CV Top-1, much less
  than the fold spread.
* **The fold spread comes from the data.** One fold holds trips 00000054 and 00000033,
  whose GPS lies 120–140 m from the reference route. That fold scores about 0.53; the others score 0.79–0.93.
* **Heading matters most.** Permutation importance for the final model is led by
  heading difference (0.43 increase in log loss), then the distance margin to the
  nearest candidate (0.16), then the neighbouring-point distances (0.10 each).
  The heading feature is what separates the two directions of a two-way road.
* **Gaussian NB** has the highest recall (0.912) but very low precision (0.562): it says
  "yes" to too many candidates. Its independence assumption does not hold for our correlated distance features.
* **The HMM's main contribution is route coherence** (continuity +11.5 points),
  more than per-point accuracy (+0.7 points).

## 6. Limitations

* Labels are derived by alignment; the reference route is not guaranteed error-free.
* Two training trips are clearly misaligned and were kept, so CV figures include them.
* Student-test results come from 10 trips. Differences under about 1 point are within noise.
* Route continuity and route F1 are our own definitions.
* Roads are straight segments between nodes; curved roads with few nodes look further from the GPS.
