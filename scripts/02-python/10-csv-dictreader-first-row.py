"""02-python / 10-csv-dictreader-first-row：逐行读取并打印第一行。"""
import csv

path = "data/02-python/supermarket_sales.csv"
with open(path, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["Branch"], row["Total"])
        break
