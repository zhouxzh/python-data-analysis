"""02-python / 14-breast-cancer-audit：审计行数、列数和表头。"""
import csv

path = "data/02-python/breast_cancer.csv"
with open(path, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

print("总记录数:", len(rows))
print("列数:", len(fieldnames))
print("列名:", fieldnames)
