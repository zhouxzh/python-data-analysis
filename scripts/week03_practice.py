"""Week 3 实践：数据清洗与审计。

运行：
    python scripts/week03_practice.py
"""
import io
import zipfile

import pandas as pd

numeric_cols = ["PM25", "PM10", "NO2", "SO2"]

print("=" * 60)
print("1. 审计脏数据")
print("=" * 60)

dirty = pd.read_csv("data/air_quality_dirty.csv")
print("shape:", dirty.shape)
print("重复行数:", dirty.duplicated().sum())
print()
print("缺失值:")
print(dirty.isna().sum())
print()
print("日期样例:")
print(dirty["date"].tail(8))

print()
print("=" * 60)
print("2. 清洗数据")
print("=" * 60)


def clean_air(df):
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned["date_parsed"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date_parsed"])
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned


cleaned = clean_air(dirty)
print("清洗前:", dirty.shape)
print("清洗后:", cleaned.shape)
print("重复行:", cleaned.duplicated().sum())
print("日期缺失:", cleaned["date_parsed"].isna().sum())
print()
print("清洗后数值缺失:")
print(cleaned[numeric_cols].isna().sum())

print()
print("=" * 60)
print("3. 认识伪缺失 unknown")
print("=" * 60)

with zipfile.ZipFile("data/bank_marketing.zip") as outer:
    inner_name = next(n for n in outer.namelist() if n.endswith("bank-additional.zip"))
    with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
        csv_name = next(n for n in inner.namelist() if n.endswith("bank-additional-full.csv"))
        with inner.open(csv_name) as f:
            bank = pd.read_csv(f, sep=";")

unknown_cols = ["job", "marital", "education", "default", "housing", "loan"]
print(bank[unknown_cols].eq("unknown").sum().sort_values(ascending=False))
