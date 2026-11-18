# Report handoff

Everything you need to write the PathSnap PDF. All paths are relative to
`ML_Hackathon/report/`. Every figure was generated from the real data and saved
results by `scripts/build_report_evidence.py`. Rerunning it reproduces them all.

Read these two first; they hold all the numbers:
* `MODEL_RESULTS_SUMMARY.md`: every model, the chosen model and why, tuning, HMM results.
* `MAP_AND_SYSTEM_SUMMARY.md`: maps, pipeline, HMM, API and UI.

The report must follow the section order in Deliverables.pdf (2.2 to 7).

---

## Section 2.2: Dataset understanding and EDA

All EDA figures use the **training partition (80 trips)**. Say so in the captions.

| Figure | Shows |
|---|---|
| `evidence/eda/gps_points_per_trip.png` | Trip sizes: median 1,529 points, from 361 to 19,313 |
| `evidence/eda/trip_length_km.png` | Distance driven per trip |
| `evidence/eda/timestamp_intervals.png` | ~1 Hz sampling (96.5 % of all steps are exactly 1 s, matching the guideline's "about 97 %"), with a long tail of gaps; the longest is 153,064 s (about 42 hours) |
| `evidence/eda/missing_values_and_quality.png` | **Missing-value analysis (required):** zero missing, out-of-range, duplicate or backward timestamps; gaps over 10 s counted |
| `evidence/eda/speed_distribution.png` | Speed with the 50 m/s outlier threshold; outliers are flagged, not removed |
| `evidence/eda/candidates_per_point.png` | How many candidate roads each GPS point gets |
| `evidence/eda/candidate_distance_distribution.png` | The correct road is usually within ~5 m; other candidates are further away |
| `evidence/eda/heading_difference_distribution.png` | The correct road points the same way the car moves |
| `evidence/eda/candidate_coverage_per_trip.png` | Two trips (00000054, 00000033) where most points have no correct candidate |
| `evidence/eda/feature_distributions.png` | Six key features, correct road vs others |
| `evidence/eda/feature_correlation.png` | Correlation between the 19 features (justifies dropping the independence assumption) |
| `evidence/maps/person1_trip_00000001/trip_00000001.png` and `gps_road_overlay.png` | **GPS trajectory plot** and GPS over roads (Person 1) |
| `evidence/maps/road_network.png` | **Road-network visualisation (required):** full network of a trip |
| `evidence/maps/representative_trip.png` | **Folium map (required)** |

Answers to the two questions:
* **Preprocessing:** UTM projection; drop invalid rows; sort by time; drop exact duplicates; flag speeds over 50 m/s;
  candidates within 50 m; per-point labels by aligning the route in order.
* **Features:** listed in `../docs/methodology.md`. The most useful, per
  `evidence/models/feature_importance/hist_gradient_boosting_tuned_permutation.png`, are heading difference,
  distance margin to the nearest candidate, and neighbouring points' distance to the same road.

## Section 3: Models

**3.1 table:** copy from `MODEL_RESULTS_SUMMARY.md` §1. Route continuity per model (classifier only, student test) is in
`evidence/model_results_table.csv`, column `student_route_continuity`.

**3.2, per model:** for each of the 8 models (plus the tuned final), use the matching row of the table, its confusion matrix
(`evidence/models/confusion_matrices/<model>.png`) and these overfitting numbers (training-subset Top-1 vs CV Top-1):

| Model | Train Top-1 | CV Top-1 | Reading |
|---|---|---|---|
| HistGradientBoosting (tuned) | 0.902 | 0.793 | moderate gap |
| HistGradientBoosting | 0.895 | 0.772 | moderate gap |
| Random Forest | 0.944 | 0.771 | **largest gap: overfits** |
| MLP | 0.900 | 0.768 | moderate |
| Extra Trees | 0.907 | 0.767 | moderate |
| Decision Tree | 0.897 | 0.765 | moderate (depth 12 limits it) |
| KNN | 0.899 | 0.757 | moderate |
| Logistic Regression | 0.854 | 0.747 | small gap: underfits (linear) |
| Gaussian NB | 0.827 | 0.721 | small gap: underfits |

The CV figure is pulled down for every model by the fold containing the two broken trips, so compare the gaps, not the levels.

Initial hyperparameters and loss for each model are in `../src/models/registry.py`
(e.g. Random Forest: 300 trees, min 5 samples per leaf, √features; loss = Gini impurity).

**3.3 plots (required):**

| Required | File |
|---|---|
| CV Macro F1 for all models | `evidence/models/cv_macro_f1_comparison.png` (folds, mean, sd) |
| Student-test Top-1 | `evidence/models/student_test_top1_accuracy.png` |
| Student-test Macro F1 | `evidence/models/student_test_macro_f1.png` |
| Log loss | `evidence/models/log_loss_comparison.png` |
| Confusion matrices (optional) | `evidence/models/confusion_matrices/*.png` (9) |
| Training/validation loss, **only** for models with histories | `evidence/models/training_histories/` : MLP loss and validation curves, HistGradientBoosting (untuned and tuned) train/validation curves. No other model records epoch histories, so there are none for them, by design. |
| Plots that justify the choice | `evidence/models/cv_top1_comparison.png` (primary metric), `overfitting_train_vs_cv.png`, `model_size.png`, `inference_time.png`, `training_time.png`, `student_test_precision_recall.png`, `student_test_top1_classifier_vs_hmm.png` |
| Feature importance | `evidence/models/feature_importance/` (impurity-based for trees, coefficients for logistic regression, permutation for the final model) |

## Section 4: Bayesian optimisation

`MODEL_RESULTS_SUMMARY.md` §3 answers all six questions.
Figures: `evidence/models/bayesian_optimisation/optimization_history.png`, `param_importances.png`, `tuned_vs_untuned.png`.

## Section 5: HMM and Viterbi

Answers: `MAP_AND_SYSTEM_SUMMARY.md` §2. Results: `MODEL_RESULTS_SUMMARY.md` §4.

| Figure | Use |
|---|---|
| `evidence/models/hmm/decoder_comparison.png` | classifier-only vs classifier + HMM (required comparison) |
| `evidence/models/hmm/hmm_gain_per_trip.png` | helped 9/10 trips; trip 00000007 is the "did not help" case |
| `evidence/models/hmm/hmm_tuning_beta.png` | how the transition scale was chosen |
| `evidence/maps/map_matching_detail.png` | candidates → selected roads → route at a junction |
| `evidence/maps/map_matching_most_disagreement.png` | where classifier and HMM disagree most (clustered points, two-way street) |

## Section 6: Folium maps

Main figure: `evidence/maps/representative_trip.png` (interactive: `.html`). It has GPS points, nearby roads, reference
route, predicted route, a legend and layer control, as the guidelines require. Also usable: `map_matching_detail.png`, `road_network.png`.
The map code is `../src/visualization/folium_maps.py` and `../scripts/build_report_evidence.py`; it must be in the repo (see the end).

## Section 7: API, UI and deployment

| Available now | File |
|---|---|
| UI connected with the trip list | `evidence/ui/01_connected_trip_list.png` |
| UI with a reconstructed route and results | `evidence/ui/02_route_reconstructed_trip_00000007.png`, `03_route_reconstructed_trip_00000009.png` |
| UI error handling | `evidence/ui/04_error_api_unreachable.png` |
| UI mobile layout | `evidence/ui/05_mobile_layout.png` |
| API Swagger page | `evidence/api/swagger_docs.png` |
| API requests, responses and server log (text) | `evidence/api/01_health.txt` … `04_error_responses.txt`, `server_terminal_output.txt` |

### Still to capture by hand (required)

The guidelines want "screenshots of successful API responses, on the CMD of the
receiving laptop". Those have to be real screenshots from the two-machine setup.

1. On the API machine (Mac mini):
   `.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.
   To rehearse without evaluator data, add `PATHSNAP_SERVE_PARTITION=student_test ` before the command.
2. On the UI laptop: `python scripts/serve_ui.py`, open `http://localhost:5173/?api=http://adarshs-Mac-mini.local:8000`,
   pick a trip and press "Match route".
3. Screenshot **the API machine's terminal** showing the lines
   `POST /reconstruct from <laptop-ip> trip=… → reconstruct trip=… top1=… → POST /reconstruct -> 200`.
   The `from <laptop-ip>` part proves the request came from the other machine.
4. Screenshot **the UI on the laptop** in the same session.

## Claims you must not make

* Don't say the HMM improves every trip (it hurts trip 00000007), or that it greatly raises per-point
  accuracy (+0.7 points). Its large effect is on route continuity.
* Don't quote Top-1 0.876 for the final system: that's the per-model figure with the *default* HMM. The final system is **0.885**.
* Don't claim Random Forest was rejected because it scored worse. It's a near-tie; it was rejected for overfitting, size and speed.
* Don't present any evaluator-trip accuracy. None was computed, and none may be.
* Don't call the per-point labels "ground truth". They're derived from the reference route by alignment.
* Don't describe a `/predict` endpoint or a file-upload screen; neither exists.
* Don't say the vehicle was stationary in `map_matching_most_disagreement.png`. The points are clustered, but we didn't verify the cause.
* Route continuity and sequence F1 are our own definitions. Say so.
* Disclose AI assistance, as the guidelines require.

## Before submission

`report/` is not committed yet. The guidelines require the map-generation code in GitHub, so at least
`scripts/build_report_evidence.py` and the updated `src/visualization/folium_maps.py` should be committed (ask Greninja44).
The evidence images can be committed too, about 15 MB in total. No evaluator data is in `report/`.
