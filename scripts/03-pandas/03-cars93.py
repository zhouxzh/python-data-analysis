"""03-pandas / 03-cars93：按汽车类型比较平均价格。

运行：
    python scripts/03-pandas/03-cars93.py
"""
import pandas as pd

cars = pd.read_csv("data/03-pandas/Cars93.csv")
by_type = (
    cars.groupby("Type")["Price"]
    .agg(["mean", "count"])
    .round(2)
    .sort_values("mean", ascending=False)
)
print(by_type)
