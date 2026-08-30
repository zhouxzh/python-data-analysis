"""01-agent / 03-room-type-price：回答第一个数据问题：哪种房型平均价格最高。

运行：
    python scripts/01-agent/03-room-type-price.py
"""
import pandas as pd

df = pd.read_csv("data/01-agent/nyc_airbnb.csv")
summary = (
    df.groupby("room_type")["price"]
    .agg(["mean", "count"])
    .round(2)
    .sort_values("mean", ascending=False)
)
print(summary)
print()
print("结论：Entire home/apt 平均价格最高；每个分组都要报告样本量 count。")
