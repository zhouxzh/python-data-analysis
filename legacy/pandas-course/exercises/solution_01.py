"""
练习1答案：数据读取与查看
"""
import pandas as pd

# 1. 读取CSV文件
df = pd.read_csv('data/students.csv')

# 2. 显示前5行
print("前5行数据：")
print(df.head())

# 3. 显示基本信息
print("\n数据信息：")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print(df.info())

# 4. 计算成绩平均值
print(f"\n成绩平均值: {df['成绩'].mean()}")
