"""Week 5 实践：EDA 与可视化。

运行：
    python scripts/week05_practice.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path("projects/demo/week05")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("1. 钻石价格分布：diamonds.csv")
print("=" * 60)

diamonds = pd.read_csv("data/05-eda-viz/diamonds.csv")
print(diamonds["price"].describe())
plt.figure(figsize=(8, 4))
plt.hist(diamonds["price"], bins=50, color="#4c72b0", edgecolor="white")
plt.title("Diamond price distribution")
plt.xlabel("price")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(OUT / "diamonds_price.png", dpi=110)
plt.close()

print()
print("=" * 60)
print("2. 人口统计：midwest.csv")
print("=" * 60)

midwest = pd.read_csv("data/05-eda-viz/midwest.csv")
state_pop = midwest.groupby("state")["poptotal"].sum().sort_values(ascending=False)
print("各州总人口前 5:")
print(state_pop.head())
plt.figure(figsize=(8, 4))
state_pop.plot.bar(color="#55a868")
plt.title("Total population by state")
plt.xlabel("state")
plt.ylabel("population")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT / "midwest_population.png", dpi=110)
plt.close()

print()
print("=" * 60)
print("3. 能源时间序列：energy_dataset.csv")
print("=" * 60)

energy = pd.read_csv("data/05-eda-viz/energy_dataset.csv")
energy["time"] = pd.to_datetime(energy["time"], errors="coerce", utc=True)
energy["total load actual"] = pd.to_numeric(
    energy["total load actual"], errors="coerce"
)
energy = energy.dropna(subset=["time", "total load actual"]).sort_values("time")
energy["date"] = energy["time"].dt.date
daily = energy.groupby("date")["total load actual"].mean()
print("日均负荷样本量:", len(daily))
print(daily.tail())
plt.figure(figsize=(8, 4))
daily.tail(120).plot.line(color="#c44e52")
plt.title("Average daily total load, last 120 days")
plt.xlabel("date")
plt.ylabel("total load actual")
plt.tight_layout()
plt.savefig(OUT / "energy_load.png", dpi=110)
plt.close()

print()
print("图表已保存到:", OUT)
