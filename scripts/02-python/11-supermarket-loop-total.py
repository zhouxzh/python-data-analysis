"""02-python / 11-supermarket-loop-total：循环版汇总各 Branch 销售额。"""
import csv

path = "data/02-python/supermarket_sales.csv"
branch_totals = {}

with open(path, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        branch = row["Branch"]
        total = float(row["Total"])
        branch_totals[branch] = branch_totals.get(branch, 0.0) + total

for branch in sorted(branch_totals):
    print(f"Branch {branch}: {branch_totals[branch]:.2f}")
