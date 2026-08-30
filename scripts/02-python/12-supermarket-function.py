"""02-python / 12-supermarket-function：封装成函数。"""
import csv

def sales_by_branch(path):
    totals = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            branch = row["Branch"]
            total = float(row["Total"])
            totals[branch] = totals.get(branch, 0.0) + total
    return totals

result = sales_by_branch("data/02-python/supermarket_sales.csv")
for branch, total in sorted(result.items()):
    print(branch, round(total, 2))
