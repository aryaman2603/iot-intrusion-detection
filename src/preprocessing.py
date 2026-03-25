"""
preprocessing.py
────────────────
Reads raw CICIoT2023 CSV, cleans it, applies label mapping,
and writes a stratified 5% and 20% sample for quick experiments.

Run:  python -m src.preprocessing
"""
import os
import polars as pl
from src.config import (
    RAW_CSV_PATH, PROCESSED_PATH, SAMPLE_PATH,
    LABEL_MAPPING, PROCESSED_DIR, SAMPLE_DIR,
)

KNOWN_CATEGORIES = set(LABEL_MAPPING.values())


def process_single_csv() -> None:
    print(f"Reading: {RAW_CSV_PATH}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    q = (
        pl.scan_csv(RAW_CSV_PATH, infer_schema_length=10_000)
        .rename({"Label": "label"})
        .with_columns(
            pl.col("label").replace(LABEL_MAPPING).alias("label_category")
        )
        .with_columns(
            pl.all().exclude("label", "label_category").cast(pl.Float32)
        )
        .filter(
            pl.all_horizontal(
                pl.all().exclude("label", "label_category").is_finite()
            )
        )
        .drop_nulls()
    )

    print("  Streaming to parquet…")
    q.sink_parquet(PROCESSED_PATH)
    print(f"  Saved → {PROCESSED_PATH}")

    # ── Validate mapping ───────────────────────────────────────────────────
    df_check = pl.read_parquet(PROCESSED_PATH).select("label", "label_category")
    unmapped = df_check.filter(
        ~pl.col("label_category").is_in(list(KNOWN_CATEGORIES))
    )
    if unmapped.height > 0:
        print(f"\n  ⚠  {unmapped.height} rows with unmapped labels:")
        print(unmapped["label"].value_counts().sort("count", descending=True))
    else:
        print("  ✓  All labels mapped correctly.")

    # ── Class distribution ─────────────────────────────────────────────────
    _print_distribution(PROCESSED_PATH, "Processed dataset")


def _stratified_sample(df: pl.DataFrame, fraction: float, seed: int = 42) -> pl.DataFrame:
    """Generic stratified sampler — samples `fraction` from each class."""
    frames = []
    for (category,), group in df.group_by("label_category"):
        n = max(10, int(round(group.height * fraction)))
        frames.append(group.sample(n=min(n, group.height), shuffle=True, seed=seed))
    return pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=seed)


def create_sample_file() -> None:
    """Stratified 5% sample — each class sampled independently."""
    print("\n  Creating stratified 5% sample…")
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    df = pl.read_parquet(PROCESSED_PATH)
    sample_df = _stratified_sample(df, fraction=0.05)
    sample_df.write_parquet(SAMPLE_PATH)
    print(f"  Saved → {SAMPLE_PATH}  ({sample_df.height:,} rows)")
    _print_distribution(sample_df, "5% sample")


def create_20_percent_sample() -> None:
    """Stratified 20% sample — better minority class coverage for M2 training."""
    from src.config import SAMPLE_20_PATH
    print("\n  Creating stratified 20% sample…")
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    df = pl.read_parquet(PROCESSED_PATH)
    sample_df = _stratified_sample(df, fraction=0.20)
    sample_df.write_parquet(SAMPLE_20_PATH)
    print(f"  Saved → {SAMPLE_20_PATH}  ({sample_df.height:,} rows)")
    _print_distribution(sample_df, "20% sample")


def _print_distribution(source, title: str) -> None:
    df = pl.read_parquet(source) if isinstance(source, str) else source
    dist = (
        df.select("label_category")
        .group_by("label_category")
        .len()
        .rename({"len": "count"})
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100)
            .round(2)
            .alias("pct%")
        )
        .sort("count", descending=True)
    )
    print(f"\n  {title} distribution:")
    print(dist)


if __name__ == "__main__":
    process_single_csv()
    create_sample_file()
    create_20_percent_sample()