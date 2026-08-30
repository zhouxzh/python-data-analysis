"""03-pandas / 02-college：比较公立和私立大学的学费与毕业率。

运行：
    python scripts/03-pandas/02-college.py
"""
import pandas as pd

college = pd.read_csv("data/03-pandas/College.csv")
private = college[college["Private"] == "Yes"]
public = college[college["Private"] == "No"]

print("公立学校数:", len(public), "私立学校数:", len(private))
print()
print("私立学校平均 Outstate:", round(private["Outstate"].mean(), 2))
print("公立学校平均 Outstate:", round(public["Outstate"].mean(), 2))
print("毕业率最高的 5 所学校:")
print(college.sort_values("Grad.Rate", ascending=False).head())
