"""Track cleaning and kinematic derivations.

Cleaning decisions (each is reported, never silent):
* rows with non-finite or out-of-range lon/lat/t are dropped;
* rows are stably sorted by timestamp if timestamps are not monotonic;
* rows with a duplicate timestamp *and* identical position are dropped;
* points implying an implausible speed (> ``max_speed_mps`` from both
  neighbours) are flagged as outliers — they are kept and still matched, but
  the flag is a model feature and is reported.

The original row number of every kept observation is preserved in
``obs_index`` so predictions map back to the raw ``.track`` file.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.geo import LocalProjection, bearing_deg

MAX_SPEED_MPS = 70.0  # ~250 km/h; anything above from both neighbours is an outlier


@dataclass
class CleanTrack:
    obs_index: np.ndarray  # original row numbers in the .track file
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    xy: np.ndarray
    dt: np.ndarray  # seconds since previous kept observation (nan for first)
    step_m: np.ndarray  # metres since previous kept observation (nan for first)
    speed_mps: np.ndarray  # centred speed estimate
    heading_deg: np.ndarray  # movement bearing over a window (nan if stationary)
    outlier: np.ndarray  # bool
    report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.obs_index)


def clean_track(track: pd.DataFrame, proj: LocalProjection, heading_window: int = 2,
                max_speed_mps: float = MAX_SPEED_MPS) -> CleanTrack:
    df = track.reset_index(drop=True).copy()
    df["obs_index"] = np.arange(len(df))
    n_raw = len(df)

    finite = np.isfinite(df[["lon", "lat", "t"]].to_numpy()).all(axis=1)
    in_range = df["lon"].between(-180, 180) & df["lat"].between(-90, 90)
    df = df[finite & in_range.to_numpy()]
    n_invalid = n_raw - len(df)

    was_sorted = bool(df["t"].is_monotonic_increasing)
    if not was_sorted:
        df = df.sort_values("t", kind="stable")
    dup = df.duplicated(subset=["lon", "lat", "t"], keep="first")
    n_dup = int(dup.sum())
    df = df[~dup]

    lon, lat, t = (df[c].to_numpy(float) for c in ("lon", "lat", "t"))
    xy = proj.to_xy(lon, lat) if len(df) else np.zeros((0, 2))
    n = len(df)

    dt = np.full(n, np.nan)
    step = np.full(n, np.nan)
    if n > 1:
        dt[1:] = np.diff(t)
        step[1:] = np.hypot(*np.diff(xy, axis=0).T)

    # speed: centred over immediate neighbours, falling back to one side
    speed = np.full(n, np.nan)
    if n > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = np.where(dt[1:] > 0, step[1:] / dt[1:], np.nan)
        back = np.concatenate([[np.nan], fwd])
        ahead = np.concatenate([fwd, [np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns when dt == 0
            speed = np.nanmean(np.vstack([back, ahead]), axis=0)
        outlier = (np.nan_to_num(back, nan=0) > max_speed_mps) & (np.nan_to_num(ahead, nan=0) > max_speed_mps)
        # end points: only one neighbour available
        outlier[0] = np.nan_to_num(ahead[0], nan=0) > max_speed_mps
        outlier[-1] = np.nan_to_num(back[-1], nan=0) > max_speed_mps
    else:
        outlier = np.zeros(n, bool)

    # heading: displacement between t-w and t+w (robust to 1 Hz jitter)
    heading = np.full(n, np.nan)
    if n > 1:
        w = max(int(heading_window), 1)
        lo = np.clip(np.arange(n) - w, 0, n - 1)
        hi = np.clip(np.arange(n) + w, 0, n - 1)
        d = xy[hi] - xy[lo]
        moving = np.hypot(d[:, 0], d[:, 1]) > 1.0
        heading[moving] = bearing_deg(d[moving, 0], d[moving, 1])

    report = {
        "n_raw": n_raw,
        "n_invalid_dropped": int(n_invalid),
        "n_duplicates_dropped": n_dup,
        "was_time_sorted": was_sorted,
        "n_outliers_flagged": int(outlier.sum()),
        "n_kept": n,
    }
    return CleanTrack(df["obs_index"].to_numpy(), lon, lat, t, xy, dt, step,
                      np.asarray(speed, float), heading, outlier, report)
