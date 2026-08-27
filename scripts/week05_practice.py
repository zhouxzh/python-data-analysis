"""Week 5 实践：可视化表达。

运行：
    python scripts/week05_practice.py

输出：
    scripts/output/week05_dashboard.png
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("1. 读取小时级空气质量数据")
print("=" * 60)

df = pd.read_csv(REPO_ROOT / "data" / "synthetic_air_quality.csv", parse_dates=["datetime"])
print("shape:", df.shape)
print(df.head())
print()
print("城市:", df["city"].unique())
print("时间范围:", df["datetime"].min(), "->", df["datetime"].max())

print()
print("=" * 60)
print("2. CityA 每日平均 PM2.5")
print("=" * 60)

daily = (
    df[df["city"] == "CityA"]
    .set_index("datetime")["pm25"]
    .resample("D")
    .mean()
)
print(daily.head())

print()
print("=" * 60)
print("3. 生成 2x2 监测面板")
print("=" * 60)

daily_all = (
    df.groupby(["city", df["datetime"].dt.date])["pm25"]
    .mean()
    .reset_index()
)
daily_all.columns = ["city", "date", "pm25_daily"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for city in df["city"].unique():
    sub = daily_all[daily_all["city"] == city]
    axes[0, 0].plot(sub["date"], sub["pm25_daily"], label=city)
axes[0, 0].set_title("Daily mean PM2.5 by city")
axes[0, 0].set_ylabel("PM2.5")
axes[0, 0].legend()

sns.boxplot(data=df, x="city", y="pm25", ax=axes[0, 1])
axes[0, 1].set_title("PM2.5 distribution by city")

axes[1, 0].scatter(df["humidity"], df["pm25"], s=3, alpha=0.2)
axes[1, 0].set_title("Humidity vs PM2.5")
axes[1, 0].set_xlabel("Humidity")
axes[1, 0].set_ylabel("PM2.5")

axes[1, 1].scatter(df["precipitation"], df["pm25"], s=3, alpha=0.2)
axes[1, 1].set_title("Precipitation vs PM2.5")
axes[1, 1].set_xlabel("Precipitation")
axes[1, 1].set_ylabel("PM2.5")

fig.tight_layout()
output_path = OUT_DIR / "week05_dashboard.png"
fig.savefig(output_path, dpi=150)
print("saved:", output_path)
