"""02-python / 06-stock-csv：读取 CSV 的 Price 列。"""
import csv
import statistics

path = "data/02-python/stock_price.csv"
prices = []

with open(path, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prices.append(float(row["Price"]))

print("样本量:", len(prices))
print("平均价格:", round(statistics.mean(prices), 2))
print("最高价格:", round(max(prices), 2))
print("最低价格:", round(min(prices), 2))
