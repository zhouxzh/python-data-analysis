"""04-viz / 03-energy-load：能源负荷时间序列折线图。

运行：
    python scripts/04-viz/03-energy-load.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path("projects/demo/04-viz")
OUT.mkdir(parents=True, exist_ok=True)

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
print("图表已保存到:", OUT / "energy_load.png")
