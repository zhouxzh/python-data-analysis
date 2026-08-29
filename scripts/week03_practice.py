"""Week 3 实践：数据清洗与审计。

运行：
    python scripts/week03_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 审计主数据集")
print("=" * 60)

df = pd.read_csv("data/nyc_airbnb.csv")
print("shape:", df.shape)
print("重复行数:", df.duplicated().sum())
print("id 重复数:", df["id"].duplicated().sum())
print()
print("缺失值:")
print(df.isna().sum())
print()
print("price describe:")
print(df["price"].describe().round(2))
print()
print("价格异常: price == 0 有", (df["price"] == 0).sum(), "行")
print("价格异常: price > 1000 有", (df["price"] > 1000).sum(), "行")

print()
print("=" * 60)
print("2. 清洗数据")
print("=" * 60)


def clean_airbnb(data):
    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned["last_review"] = pd.to_datetime(cleaned["last_review"], errors="coerce")
    cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
    cleaned["anomaly_price"] = (cleaned["price"] <= 0) | (cleaned["price"] > 1000)
    return cleaned


cleaned = clean_airbnb(df)
print("清洗前:", df.shape)
print("清洗后:", cleaned.shape)
print("重复行:", cleaned.duplicated().sum())
print("日期缺失:", cleaned["last_review"].isna().sum())
print("价格异常:", cleaned["anomaly_price"].sum())

print()
print("=" * 60)
print("3. 区分 0 与缺失")
print("=" * 60)

print("number_of_reviews == 0 的房源:", (df["number_of_reviews"] == 0).sum())
print("last_review 缺失的房源:", df["last_review"].isna().sum())
print("说明：0 是“没有评论”，NaN 是缺失；两者要分开报告。")
