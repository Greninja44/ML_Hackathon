# Part 1 Report – Data Processing and Feature Engineering

## Project: PathSnap – Distributed Map-Matching & Route Reconstruction

### 1. Overview

My responsibility in the PathSnap project was data processing, dataset exploration, candidate generation and feature engineering.

The objective of this phase was to prepare raw GPS trajectory data and road-network information for the map-matching process.

The work involved understanding the dataset, examining GPS trajectories, generating possible road-segment candidates for GPS observations and extracting useful features for subsequent processing.

### 2. Dataset Description

The dataset contains GPS trajectory information and corresponding road-network information.

Each trip contains:

- **`.track`** – GPS observations containing longitude, latitude and timestamp.
- **`.nodes`** – Road-network node coordinates.
- **`.arcs`** – Connections between road-network nodes.
- **`.route`** – Reference route represented as an ordered sequence of arc IDs.

Dataset distribution:

| Dataset | Number of trips | GPS observations |
|---|---:|---:|
| Training | 80 | 208,417 |
| Student test | 10 | 19,160 |
| Total | 90 | 227,577 |

### 3. Data Exploration and Analysis

Initial dataset exploration was performed to understand the structure and quality of the available data.

The following checks were performed:

- Number of training and student-test trips.
- Number of GPS observations.
- Number of road-network nodes and arcs.
- Missing GPS values.
- Duplicate GPS observations.
- Number of reference route entries.
- Timestamp interval analysis.

Results:

- Total GPS observations: 227,577.
- Missing GPS values: 0.
- Duplicate GPS observations: 0.
- Training reference route entries: 31,336.
- Student-test reference route entries: 3,477.

Timestamp intervals were also examined to identify unusually large gaps and negative time differences. No negative timestamp intervals were found. Some trips contained large gaps that were retained for further investigation rather than automatically removed.

### 4. Data Visualization

Three types of visualizations were generated.

**4.1 GPS Trajectory Visualization**

GPS coordinates were plotted to understand the movement pattern of individual trips.

**4.2 Road Network Visualization**

Road-network nodes and arcs were visualized to understand the structure and connectivity of the road network.

**4.3 GPS and Road Network Overlay**

GPS trajectories were plotted over the road network to visually inspect how closely GPS observations followed mapped roads.

The visualizations helped understand the relationship between GPS data and the underlying road network.

### 5. Candidate Generation

Candidate generation identifies possible road segments corresponding to each GPS observation.

Instead of comparing every GPS point with every road segment, a spatial search approach was implemented.

**Methodology:**

1. Road-network nodes were projected into a suitable UTM coordinate system to work with distances in metres.
2. Road segments were constructed using the start and end node IDs from the arc data.
3. Midpoints of road segments were calculated.
4. A KD-tree (`scipy.spatial.cKDTree`) was constructed using the road-segment midpoints.
5. The 20 nearest segment midpoints were shortlisted for each GPS observation.
6. Exact point-to-segment distances were calculated for the shortlisted segments.
7. A maximum of five candidates within 50 metres were retained.
8. If no candidate was found within 50 metres, the nearest candidate from the shortlist was retained.

This reduced the search space and generated a manageable set of possible road segments for each GPS observation.

### 6. Feature Engineering

After candidate generation, additional features were calculated for every GPS-point and candidate-road-segment combination.

The final feature dataset contains 15 columns, including identification fields and engineered features.

| Feature | Description |
|---|---|
| `partition` | Training or student-test dataset |
| `trip_id` | Unique trip identifier |
| `gps_index` | GPS observation index |
| `timestamp` | GPS observation timestamp |
| `arc_id` | Candidate road-segment identifier |
| `from_node_id` | Starting road-network node |
| `to_node_id` | Ending road-network node |
| `distance_m` | Distance between GPS point and candidate road segment |
| `gps_distance_m` | Distance travelled between consecutive GPS observations |
| `gps_bearing` | Direction of GPS movement |
| `time_gap_sec` | Time difference between consecutive observations |
| `gps_speed_mps` | Estimated GPS movement speed in metres per second |
| `road_length_m` | Length of the candidate road segment |
| `road_bearing` | Direction of the candidate road segment |
| `direction_difference` | Angular difference between GPS movement and road direction |

These features describe spatial proximity, movement patterns, road geometry and directional similarity.

### 7. Full Dataset Processing Results

The pipeline was executed across all 90 trips.

| Dataset | Feature records | Reference route entries |
|---|---:|---:|
| Training | 804,416 | 31,336 |
| Student test | 79,669 | 3,477 |
| Total | 884,085 | 34,813 |

The feature and reference-label datasets were saved separately.

Generated files:

- `data/processed/train_features.csv`
- `data/processed/train_labels.csv`
- `data/processed/student_test_features.csv`
- `data/processed/student_test_labels.csv`

The reference route labels were kept separate from the engineered candidate features to avoid accidental label leakage.

### 8. Data Quality Validation

Basic validation was performed on the training feature dataset.

| Validation check | Result |
|---|---:|
| Rows | 804,416 |
| Columns | 15 |
| Missing values | 0 |
| Duplicate rows | 0 |

Candidate-distance analysis:

- Candidate records with distance greater than 50 metres: 19,375.
- Candidate records with distance greater than 100 metres: 1,877.
- Candidate records with distance greater than 200 metres: 308.
- Candidate records within 50 metres: approximately 97.59%.

The candidate-generation implementation retains the nearest candidate when no candidate is available within the 50-metre threshold. Therefore, distances greater than 50 metres are possible.

### 9. Identified Issue – Trip 54

During validation, Trip 54 was identified as an outlier.

Its nearest-candidate distance statistics were:

| Metric | Result |
|---|---:|
| GPS observations | 19,313 |
| Nearest candidate within 50 metres | 2,788 |
| Nearest candidate over 50 metres | 16,525 |
| Nearest candidate over 100 metres | 1,103 |
| Mean nearest-candidate distance | 73.53 metres |
| Maximum nearest-candidate distance | 123.05 metres |

The GPS and road-network coordinate ranges overlap, but their coordinate distributions have noticeably different means.

This indicates a possible spatial-coverage or trajectory-matching issue that requires further investigation.

The trip has not been removed, and the original data has not been modified.

### 10. Tools and Technologies Used

- Python 3.13.7
- Pandas – data loading, manipulation and analysis.
- NumPy – numerical calculations.
- SciPy – KD-tree spatial candidate search.
- Scikit-learn – available for subsequent machine-learning workflows.
- PyProj – coordinate-system transformation.
- Matplotlib – visualization.
- Shapely – geometric operations.
- VS Code – development environment.
- Git and GitHub – version control.

### 11. Limitations and Future Improvements

The current implementation has some limitations:

1. The KD-tree searches using road-segment midpoints, so some nearby long road segments could be missed during the initial shortlist.
2. The 50-metre threshold may not be suitable for every GPS observation.
3. The fallback nearest-candidate strategy can retain road segments located far from GPS observations.
4. Large timestamp gaps require further investigation.
5. Candidate quality should be evaluated against reference routes once the label alignment and evaluation methodology are established.
6. Trip 54 requires additional spatial investigation.

Possible improvements include using a spatial index based on road-segment geometries, improving candidate recall and investigating anomalous trips.

### 12. Conclusion

The data-processing and feature-engineering phase of PathSnap was implemented across all 90 trips.

The pipeline successfully generated 884,085 candidate feature records and retained reference route information separately.

Dataset exploration, GPS and road-network visualization, candidate generation, feature extraction and initial data-quality validation were completed.

The resulting datasets provide the prepared inputs for the subsequent map-matching and route-reconstruction stages of the project.