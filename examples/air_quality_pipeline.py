"""Week 6 参考：空气质量合并、汇总与迷你图。

用法：
    python examples/air_quality_pipeline.py

输出：
    examples/output/clean_air.csv
    examples/output/air_summary.csv
    examples/output/region_summary.csv
    examples/output/dashboard.png
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["PM25", "PM10", "NO2", "SO2"]


def clean_air(df: pd.DataFrame) -> pd.DataFrame:
    """对空气质量数据做课程演示级清洗，不修改原文件。"""
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.drop_duplicates()
    cleaned[NUMERIC_COLS] = cleaned[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(subset=["city", "date"])
    return cleaned


def main() -> None:
    raw = pd.read_csv(DATA_DIR / "air_quality_dirty.csv")
    cleaned = clean_air(raw)
    cleaned.to_csv(OUT_DIR / "clean_air.csv", index=False)

    city_summary = (
        cleaned.groupby("city")[NUMERIC_COLS]
        .mean()
        .round(1)
        .sort_values("PM25", ascending=False)
        .reset_index()
    )

    city_info = pd.read_csv(DATA_DIR / "city_info.csv")
    merged = city_summary.merge(city_info, on="city", how="left")
    merged["pm25_per_million_pop"] = (
        merged["PM25"] / merged["population_million"]
    ).round(3)

    region_summary = (
        merged.groupby("region")
        .agg(
            city_count=("city", "count"),
            mean_pm25=("PM25", "mean"),
            total_population_million=("population_million", "sum"),
            max_pm25=("PM25", "max"),
        )
        .round(2)
        .reset_index()
    )

    merged.to_csv(OUT_DIR / "air_summary.csv", index=False)
    region_summary.to_csv(OUT_DIR / "region_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(merged["city"], merged["PM25"], color="#4C72B0")
    axes[0].set_title("Mean PM25 by City")
    axes[0].set_ylabel("PM25")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(
        region_summary["region"],
        region_summary["mean_pm25"],
        color="#55A868",
    )
    axes[1].set_title("Mean PM25 by Region")
    axes[1].set_ylabel("PM25")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dashboard.png", dpi=150)

    print("cleaned shape:", cleaned.shape)
    print("city summary:")
    print(city_summary.head())
    print("\nregion summary:")
    print(region_summary)


if __name__ == "__main__":
    main()
