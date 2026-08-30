"""05-cleaning-merge / 02-telco-churn：类型转换、缺失和按合同计算流失率。

运行：
    python scripts/05-cleaning-merge/02-telco-churn.py
"""
import pandas as pd

telco = pd.read_csv("data/04-cleaning/telco_customer_churn.csv")
print("shape:", telco.shape)
print("重复 Customer ID 数:", int(telco["Customer ID"].duplicated().sum()))
print("总缺失值:", int(telco.isna().sum().sum()))

telco["Total Charges"] = pd.to_numeric(
    telco["Total Charges"], errors="coerce"
)
print("Total Charges 转换后新增缺失:", int(telco["Total Charges"].isna().sum()))
print("Internet Type 缺失数:", int(telco["Internet Type"].isna().sum()))
print("Churn 分布:")
print(telco["Churn"].value_counts())
print("按 Contract 的流失率:")
print(telco.groupby("Contract")["Churn"].mean().round(4))
