"""Week 1 实践：从 Excel 到第一个数据问题。

运行：
    python scripts/week01_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. Python 基础：列表与字典")
print("=" * 60)

students = ["张三", "李四", "王五", "赵六", "周七"]
scores = [78, 85, 90, 62, 73]

print("学生列表:", students)
print("第一个学生:", students[0])
print("最高分:", max(scores))
print("最低分:", min(scores))
print("总分:", sum(scores))
print("人数:", len(scores))

total = 0
for s in scores:
    total = total + s
print("循环计算总分:", total)
print("平均分:", total / len(scores))

print()
print("=" * 60)
print("2. 读取 Excel 成绩单")
print("=" * 60)

df = pd.read_excel("data/成绩单.xlsx")
print("shape:", df.shape)
print()
print(df.dtypes)
print()
print("缺失值:")
print(df.isna().sum())
print()
print(df)

print()
print("=" * 60)
print("3. 修复数学列并计算平均分")
print("=" * 60)

df["数学"] = df["数学"].astype(str).str.replace("..", ".", regex=False)
df["数学"] = pd.to_numeric(df["数学"], errors="coerce")
df["total"] = df[["语文", "数学", "英语"]].sum(axis=1)
df = df.sort_values("total", ascending=False).reset_index(drop=True)

print(df[["姓名", "语文", "数学", "英语", "total"]])
print()
print("平均分:")
print(df[["语文", "数学", "英语"]].mean().round(2))
