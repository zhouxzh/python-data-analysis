"""Week 4 实践：EDA 与提出假设。

运行：
    python scripts/week04_practice.py
"""
import io
import zipfile

import pandas as pd

print("=" * 60)
print("1. 读取银行营销数据")
print("=" * 60)

with zipfile.ZipFile("data/bank_marketing.zip") as outer:
    inner_name = next(n for n in outer.namelist() if n.endswith("bank-additional.zip"))
    with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
        csv_name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
        with inner.open(csv_name) as f:
            df = pd.read_csv(f, sep=";")

print("shape:", df.shape)
print()
print(df.head())

print()
print("=" * 60)
print("2. 目标变量与 unknown")
print("=" * 60)

print("y 分布:")
print(df["y"].value_counts(normalize=True).round(4))

unknown_cols = ["job", "marital", "education", "default", "housing", "loan"]
print()
print("unknown 数量:")
print(df[unknown_cols].eq("unknown").sum().sort_values(ascending=False))

print()
print("=" * 60)
print("3. 带问题做 EDA")
print("=" * 60)


def success_rate_table(df, group_cols):
    return (
        df.groupby(group_cols)["y"]
        .agg(customers="size", success_rate=lambda s: (s == "yes").mean())
        .round(4)
        .sort_values("success_rate", ascending=False)
    )


print("contact:")
print(success_rate_table(df, "contact"))
print()
print("month 前 5:")
print(success_rate_table(df, "month").head(5))
print()
print("poutcome:")
print(success_rate_table(df, "poutcome"))

print()
print("=" * 60)
print("4. 目标泄漏检查：duration")
print("=" * 60)

df["contacted_before"] = df["pdays"].ne(999)
print(df.groupby("y")["duration"].mean().round(1))
