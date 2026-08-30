"""02-python / 15-breast-cancer-missing：找出 Bare.nuclei 的 ? 和 NA。"""
import csv

path = "data/02-python/breast_cancer.csv"
missing = 0
with open(path, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["Bare.nuclei"].strip() in {"?", "NA", ""}:
            missing += 1

print("Bare.nuclei 缺失/异常标记:", missing)
