
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw")

results = []

for partition in ["train", "student_test"]:
    for trip in sorted((DATA_PATH / partition).iterdir()):
        if not trip.is_dir():
            continue

        track_file = trip / f"{trip.name}.track"

        df = pd.read_csv(
            track_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        gaps = df["timestamp"].diff().dropna()

        results.append({
            "partition": partition,
            "trip_id": trip.name,
            "observations": len(df),
            "min_interval": gaps.min(),
            "max_interval": gaps.max(),
            "mean_interval": gaps.mean(),
            "gaps_over_1_sec": (gaps > 1).sum(),
            "negative_intervals": (gaps < 0).sum()
        })

result = pd.DataFrame(results)

Path("outputs").mkdir(exist_ok=True)

result.to_csv(
    "outputs/timestamp_analysis.csv",
    index=False
)

print(result.to_string(index=False))

print("\nTrips with large timestamp gaps:")
print(
    result[result["max_interval"] > 5].to_string(index=False)
)
