"""Week 2 实践：Python 编程基础，使用 csv 与 statistics。

运行：
    python scripts/week02_practice.py
"""
import csv
import statistics
from pathlib import Path

BASE = Path("data/02-python")

print("=" * 60)
print("1. 金融时间序列：stock_price.csv")
print("=" * 60)

with open(BASE / "stock_price.csv", encoding="utf-8", newline="") as f:
    stock_rows = list(csv.DictReader(f))

prices = [float(row["Price"]) for row in stock_rows]
up_days = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])

print("样本量:", len(prices))
print("平均价格:", round(statistics.mean(prices), 2))
print("最高价格:", max(prices))
print("最低价格:", min(prices))
print("上涨天数:", up_days)

print()
print("=" * 60)
print("2. 零售数据：supermarket_sales.csv")
print("=" * 60)

with open(BASE / "supermarket_sales.csv", encoding="utf-8", newline="") as f:
    sales_rows = list(csv.DictReader(f))

branch_totals = {}
for row in sales_rows:
    branch = row["Branch"]
    total = float(row["Total"])
    branch_totals[branch] = branch_totals.get(branch, 0.0) + total

for branch in sorted(branch_totals):
    print(f"Branch {branch}: {branch_totals[branch]:.2f}")

print()
print("=" * 60)
print("3. 医疗数据：breast_cancer.csv")
print("=" * 60)

with open(BASE / "breast_cancer.csv", encoding="utf-8", newline="") as f:
    cancer_rows = list(csv.DictReader(f))

bare_values = []
for row in cancer_rows:
    value = (row.get("Bare.nuclei") or "").strip()
    if value in {"", "?", "NA"}:
        continue
    try:
        bare_values.append(float(value))
    except ValueError:
        continue

print("总记录数:", len(cancer_rows))
print("可用 Bare.nuclei 值:", len(bare_values))
print("缺失/异常值数量:", len(cancer_rows) - len(bare_values))
if bare_values:
    print("Bare.nuclei 平均值:", round(statistics.mean(bare_values), 2))
    print("最小值:", min(bare_values), "最大值:", max(bare_values))
