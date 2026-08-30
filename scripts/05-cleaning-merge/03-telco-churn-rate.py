"""05-cleaning-merge / 03-telco-churn-rate：最小代码，按 Contract 计算流失率。"""
import pandas as pd

telco = pd.read_csv("data/04-cleaning/telco_customer_churn.csv")
telco["Total Charges"] = pd.to_numeric(telco["Total Charges"], errors="coerce")

print("shape:", telco.shape)
print("重复 Customer ID 数:", int(telco["Customer ID"].duplicated().sum()))
print("Internet Type 缺失数:", int(telco["Internet Type"].isna().sum()))
print(telco.groupby("Contract")["Churn"].mean().round(4))
