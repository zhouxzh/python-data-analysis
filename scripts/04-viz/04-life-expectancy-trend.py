"""04-viz / 04-life-expectancy-trend：用 pandas 集成绘图展示预期寿命趋势。

运行：
    python scripts/04-viz/04-life-expectancy-trend.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path("projects/demo/04-viz")
OUT.mkdir(parents=True, exist_ok=True)

life = pd.read_csv("data/04-cleaning/Life_Expectancy_Data.csv")
life.columns = [col.strip() for col in life.columns]

year_mean = life.groupby("Year")["Life expectancy"].mean()
print("年份趋势样本量:", len(year_mean))
print(year_mean.tail().round(2))

status_year = (
    life.groupby(["Year", "Status"])["Life expectancy"]
    .mean()
    .unstack()
)
print("Status x Year 样本量:")
print(status_year.notna().sum())

plt.figure(figsize=(8, 4))
year_mean.plot.line(color="#c44e52", linewidth=2)
plt.title("Global mean life expectancy by year")
plt.xlabel("year")
plt.ylabel("life expectancy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "life_expectancy_year.png", dpi=110)
plt.close()

plt.figure(figsize=(8, 4))
status_year.plot.line()
plt.title("Life expectancy by development status")
plt.xlabel("year")
plt.ylabel("life expectancy")
plt.legend(title="Status")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "life_expectancy_status.png", dpi=110)
plt.close()

print("图表已保存到:", OUT)
