"""Build a modelling-ready table from the raw Kaggle CSV.

raw CSV -> drop mostly-empty columns -> parse "metric (imperial)" strings into
numbers -> add Brand and Rating from the curated CSV -> data/processed/bikes.parquet
"""

from pathlib import Path

import pandas as pd

from two_wheel_data.data import CURATED_FILE, load_bikes

# Fraction of NaN (0.0-1.0) above which a column is dropped. 0.7 keeps
# sparse-but-interesting columns like Torque (57% NaN) and Top speed (67%).
DEFAULT_NAN_THRESHOLD = 0.7

PROCESSED_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "bikes.parquet"

# new column -> (source column, regex capturing the metric number)
# Every source value leads with metric and repeats itself in imperial units inside
# brackets, e.g. '174.5 kg (384.7 pounds)' -- we keep the metric number only.
MEASUREMENTS = {
    "Displacement (ccm)": ("Displacement", r"([\d.,]+)\s*ccm"),
    "Power (hp)": ("Power", r"([\d.,]+)\s*HP"),
    "Power rpm": ("Power", r"@\s*([\d.,]+)\s*RPM"),
    "Torque (Nm)": ("Torque", r"([\d.,]+)\s*Nm"),
    "Torque rpm": ("Torque", r"@\s*([\d.,]+)\s*RPM"),
    "Top speed (km/h)": ("Top speed", r"([\d.,]+)\s*km/h"),
    "Bore (mm)": ("Bore x stroke", r"([\d.,]+)\s*x"),
    "Stroke (mm)": ("Bore x stroke", r"x\s*([\d.,]+)\s*mm"),
    "Dry weight (kg)": ("Dry weight", r"([\d.,]+)\s*kg"),
    "Wet weight (kg)": ("Weight incl. oil, gas, etc", r"([\d.,]+)\s*kg"),
    "Fuel capacity (lts)": ("Fuel capacity", r"([\d.,]+)\s*litres"),
    "Compression ratio": ("Compression", r"([\d.,]+)\s*:\s*1"),
    "Power/weight (hp/kg)": ("Power/weight ratio", r"([\d.,]+)\s*HP/kg"),
    "Wheelbase (mm)": ("Wheelbase", r"([\d.,]+)\s*mm"),
    "Seat height (mm)": ("Seat height", r"([\d.,]+)\s*mm"),
    "Ground clearance (mm)": ("Ground clearance", r"([\d.,]+)\s*mm"),
    "Overall length (mm)": ("Overall length", r"([\d.,]+)\s*mm"),
    "Overall width (mm)": ("Overall width", r"([\d.,]+)\s*mm"),
    "Overall height (mm)": ("Overall height", r"([\d.,]+)\s*mm"),
    "Front wheel travel (mm)": ("Front wheel travel", r"([\d.,]+)\s*mm"),
    "Rear wheel travel (mm)": ("Rear wheel travel", r"([\d.,]+)\s*mm"),
    "Brake diameter (mm)": ("Diameter", r"([\d.,]+)\s*mm"),
}

# Free text that is only useful once mined into flags, plus the raw Rating column,
# which holds website copy rather than a score (the curated file has the number).
DROP_ALWAYS = ["Rating", "Comments", "Color options"]


def nan_report(df):
    """Per-column NaN count and fraction, emptiest columns first."""
    return pd.DataFrame(
        {
            "nan_count": df.isna().sum(),
            "nan_fraction": df.isna().mean(),
            "dtype": df.dtypes,
        }
    ).sort_values("nan_fraction", ascending=False)


def drop_sparse_columns(df, threshold=DEFAULT_NAN_THRESHOLD):
    """Return a copy of df without columns whose NaN fraction exceeds threshold."""
    return df.loc[:, df.isna().mean() <= threshold]


def parse_measurements(df):
    """Replace 'metric (imperial)' strings with numeric columns."""
    out = df.copy()
    for new_col, (source, pattern) in MEASUREMENTS.items():
        if source not in out.columns:
            continue
        extracted = out[source].astype("str").str.extract(pattern, expand=False)
        out[new_col] = pd.to_numeric(extracted.str.replace(",", "", regex=False), errors="coerce")
    return out.drop(columns={source for source, _ in MEASUREMENTS.values()} & set(out.columns))


def _join_key(model, year):
    """Normalised 'brand model|year' key -- raw spells the name differently to curated."""
    cleaned = model.astype("str").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    return cleaned + "|" + year.astype("str")


def add_curated_columns(df, curated=None):
    """Bring Brand and the numeric Rating over from the curated CSV (~99.6% match)."""
    if curated is None:
        curated = load_bikes(CURATED_FILE)

    lookup = curated[["Brand", "Rating"]].copy()
    lookup["_key"] = _join_key(
        curated["Brand"].astype("str") + " " + curated["Model"].astype("str"), curated["Year"]
    )
    lookup = lookup.drop_duplicates(subset="_key")

    out = df.copy()
    out["_key"] = _join_key(out["Model"], out["Year"])
    out = out.merge(lookup, on="_key", how="left").drop(columns="_key")

    cols = ["Brand"] + [c for c in out.columns if c != "Brand"]
    return out[cols]


def build_dataset(threshold=DEFAULT_NAN_THRESHOLD):
    """Full pipeline: raw CSV in, modelling-ready dataframe out."""
    df = load_bikes()
    df = drop_sparse_columns(df, threshold)
    df = parse_measurements(df)
    # Drop before the join: raw also has a Rating column, and pandas would suffix both.
    df = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns])
    return add_curated_columns(df)


def save_dataset(df, path=PROCESSED_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main():
    raw = load_bikes()
    with pd.option_context("display.max_rows", None):
        print(nan_report(raw))

    df = build_dataset()
    numeric = df.select_dtypes("number").columns
    print(f"\n{raw.shape[1]} raw columns -> {df.shape[1]} kept ({len(numeric)} numeric)")
    print(f"Brand filled: {df['Brand'].notna().mean():.1%}   Rating filled: {df['Rating'].notna().mean():.1%}")

    path = save_dataset(df)
    print(f"Saved {df.shape[0]} rows to {path}")


if __name__ == "__main__":
    main()
