"""
Pandas基础示例 - 数据读取与查看
"""
import pandas as pd

# 1. 读取CSV文件
df = pd.read_csv('data/students.csv')

# 2. 查看数据
print("前3行数据：")
print(df.head(3))

print("\n数据信息：")
print(df.info())

print("\n数据统计：")
print(df.describe())

# 3. 查看特定列
print("\n姓名列：")
print(df['姓名'])

# 4. 查看多列
print("\n姓名和成绩：")
print(df[['姓名', '成绩']])
