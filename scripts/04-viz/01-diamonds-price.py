"""04-viz / 01-diamonds-price：钻石价格分布直方图。

运行：
    python scripts/04-viz/01-diamonds-price.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path("projects/demo/04-viz")
OUT.mkdir(parents=True, exist_ok=True)

diamonds = pd.read_csv("data/05-eda-viz/diamonds.csv")
print("price describe:")
print(diamonds["price"].describe())

plt.figure(figsize=(8, 4))
plt.hist(diamonds["price"], bins=50, color="#4c72b0", edgecolor="white")
plt.title("Diamond price distribution")
plt.xlabel("price")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(OUT / "diamonds_price.png", dpi=110)
plt.close()
print("图表已保存到:", OUT / "diamonds_price.png")
