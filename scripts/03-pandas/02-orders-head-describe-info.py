"""03-pandas / 02-orders-head-describe-info：查看订单表头尾、统计量和信息。"""
import pandas as pd

df = pd.read_csv('data/03-pandas/olist_orders_45d.csv')

print(df.head())
print()
print(df.tail(3))
print()
print(df.describe())
print()
df.info()
