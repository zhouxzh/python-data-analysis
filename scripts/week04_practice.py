"""Week 4 实践：数据清洗与审计。

运行：
    python scripts/week04_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 缺失汽车数据：Cars93_miss.csv")
print("=" * 60)

cars_miss = pd.read_csv("data/04-cleaning/Cars93_miss.csv")
print("shape:", cars_miss.shape)
print("每列缺失数:")
print(cars_miss.isna().sum()[cars_miss.isna().sum() > 0])
print("删除任意缺失值后:", cars_miss.dropna().shape)

print()
print("=" * 60)
print("2. 电信客户流失：telco_customer_churn.csv")
print("=" * 60)

telco = pd.read_csv("data/04-cleaning/telco_customer_churn.csv")
print("shape:", telco.shape)
print("重复 Customer ID 数:", telco["Customer ID"].duplicated().sum())
print("总缺失值:", int(telco.isna().sum().sum()))
telco["Total Charges"] = pd.to_numeric(
    telco["Total Charges"], errors="coerce"
)
print()
print("按 Contract 的流失率:")
churn_rate = (
    telco.groupby("Contract")["Churn"]
    .mean()
    .round(4)
)
print(churn_rate)

print()
print("=" * 60)
print("3. 公共健康数据：Life_Expectancy_Data.csv")
print("=" * 60)

life = pd.read_csv("data/04-cleaning/Life_Expectancy_Data.csv")
life.columns = [col.strip() for col in life.columns]
print("shape:", life.shape)
print("字段名已清理，Life expectancy 平均:")
print(round(life["Life expectancy"].mean(), 2))
print("按 Status 平均预期寿命:")
print(life.groupby("Status")["Life expectancy"].mean().round(2))
