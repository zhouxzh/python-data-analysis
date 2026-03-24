"""
Pandas数据操作示例
"""
import pandas as pd

df = pd.read_csv('data/students.csv')

# 1. 添加新列
df['等级'] = df['成绩'].apply(lambda x: '优秀' if x >= 90 else '良好' if x >= 80 else '及格')
print("添加等级列：")
print(df)

# 2. 修改数据
df.loc[df['姓名'] == '张三', '成绩'] = 90
print("\n修改后的数据：")
print(df)

# 3. 删除列
df_copy = df.copy()
df_copy = df_copy.drop('等级', axis=1)
print("\n删除等级列后：")
print(df_copy)

# 4. 排序
print("\n按成绩降序排列：")
print(df.sort_values('成绩', ascending=False))
