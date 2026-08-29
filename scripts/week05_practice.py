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
print("1. 读取 Airbnb 价格数据")
print("=" * 60)

df = pd.read_csv(REPO_ROOT / "data" / "nyc_airbnb.csv")
plot_df = df[df["price"] <= 1000].copy()
print("shape:", df.shape)
print("price 范围:", df["price"].min(), "->", df["price"].max())
print("过滤 price > 1000 后:", plot_df.shape)

print()
print("=" * 60)
print("2. 生成 2x2 价格面板")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(data=plot_df, x="neighbourhood_group", y="price", ax=axes[0, 0])
axes[0, 0].set_title("Price distribution by borough")
axes[0, 0].set_xlabel("Borough")
axes[0, 0].set_ylabel("Price (USD / night)")

sns.boxplot(data=plot_df, x="room_type", y="price", ax=axes[0, 1])
axes[0, 1].set_title("Price distribution by room type")
axes[0, 1].set_xlabel("Room type")
axes[0, 1].set_ylabel("Price (USD / night)")

sc = axes[1, 0].scatter(
    df["longitude"], df["latitude"],
    s=1, c=df["price"], cmap="viridis", alpha=0.3
)
axes[1, 0].set_title("Location and price")
axes[1, 0].set_xlabel("Longitude")
axes[1, 0].set_ylabel("Latitude")
fig.colorbar(sc, ax=axes[1, 0], label="Price (USD)")

axes[1, 1].scatter(plot_df["minimum_nights"], plot_df["price"], s=3, alpha=0.2)
axes[1, 1].set_title("Minimum nights vs price")
axes[1, 1].set_xlabel("Minimum nights")
axes[1, 1].set_ylabel("Price (USD / night)")

fig.suptitle("NYC Airbnb price overview (n = 48895)")
fig.tight_layout()
output_path = OUT_DIR / "week05_dashboard.png"
fig.savefig(output_path, dpi=150)
print("saved:", output_path)
