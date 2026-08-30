"""02-python / 19-class-counts：统计 Class 样本数。"""
import csv

path = "data/02-python/breast_cancer.csv"
classes = {}
with open(path, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        key = row["Class"]
        classes[key] = classes.get(key, 0) + 1

print("Class 样本数:", classes)
