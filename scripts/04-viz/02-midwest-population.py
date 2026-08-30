"""04-viz / 02-midwest-population：各州人口柱状图。

运行：
    python scripts/04-viz/02-midwest-population.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path("projects/demo/04-viz")
OUT.mkdir(parents=True, exist_ok=True)

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
print("图表已保存到:", OUT / "midwest_population.png")
