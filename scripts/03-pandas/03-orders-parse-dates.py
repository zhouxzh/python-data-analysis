"""03-pandas / 03-orders-parse-dates：解析订单日期并检查缺失。"""
import pandas as pd

df = pd.read_csv('data/03-pandas/olist_orders_45d.csv')
df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

print(df.dtypes)
print()
print(df[['purchase_time', 'purchase_date', 'quantity']].head(3))
print()
print(df[['purchase_time', 'purchase_date']].isna().sum())
