# Pandas 初学者教程

## 目录
1. [简介](#简介)
2. [安装](#安装)
3. [数据读取](#数据读取)
4. [数据查看](#数据查看)
5. [数据筛选](#数据筛选)
6. [数据操作](#数据操作)
7. [统计分析](#统计分析)
8. [练习题](#练习题)

## 简介

Pandas 是 Python 中最强大的数据分析库之一，提供了高效的数据结构和数据分析工具。

### 核心数据结构
- **Series**: 一维数组
- **DataFrame**: 二维表格数据

## 安装

```bash
pip install -r requirements.txt
```

## 数据读取

### 读取CSV文件

```python
import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/students.csv')
```

### 常用参数
- `encoding`: 指定编码（如 'utf-8'）
- `sep`: 分隔符（默认为逗号）
- `header`: 指定表头行

**示例代码**: [01_basic_read.py](examples/01_basic_read.py)

## 数据查看

### 基本查看方法

```python
# 查看前5行
df.head()

# 查看后5行
df.tail()

# 查看数据信息
df.info()

# 查看统计信息
df.describe()

# 查看形状
df.shape  # (行数, 列数)
```

## 数据筛选

### 单列选择

```python
# 选择单列
df['姓名']

# 选择多列
df[['姓名', '成绩']]
```

### 条件筛选

```python
# 单条件
df[df['成绩'] > 85]

# 多条件（与）
df[(df['年龄'] == 20) & (df['成绩'] > 80)]

# 多条件（或）
df[(df['城市'] == '北京') | (df['城市'] == '上海')]

# 使用 isin
df[df['城市'].isin(['北京', '上海'])]
```

### 位置选择

```python
# loc: 基于标签
df.loc[0:2, ['姓名', '成绩']]

# iloc: 基于位置
df.iloc[0:3, 0:2]
```

**示例代码**: [02_filter.py](examples/02_filter.py)

## 数据操作

### 添加列

```python
# 直接赋值
df['等级'] = '优秀'

# 基于计算
df['总分'] = df['成绩'] * 1.1

# 使用 apply
df['等级'] = df['成绩'].apply(lambda x: '优秀' if x >= 90 else '良好')
```

### 修改数据

```python
# 修改特定值
df.loc[df['姓名'] == '张三', '成绩'] = 90

# 修改整列
df['年龄'] = df['年龄'] + 1
```

### 删除数据

```python
# 删除列
df.drop('等级', axis=1, inplace=True)

# 删除行
df.drop([0, 1], axis=0, inplace=True)
```

### 排序

```python
# 按单列排序
df.sort_values('成绩', ascending=False)

# 按多列排序
df.sort_values(['年龄', '成绩'], ascending=[True, False])
```

**示例代码**: [03_operations.py](examples/03_operations.py)

## 统计分析

### 基本统计

```python
# 平均值
df['成绩'].mean()

# 中位数
df['成绩'].median()

# 最大值/最小值
df['成绩'].max()
df['成绩'].min()

# 标准差
df['成绩'].std()

# 计数
df['成绩'].count()
```

### 分组统计

```python
# 按单列分组
df.groupby('产品')['销量'].sum()

# 按多列分组
df.groupby(['产品', '日期'])['销量'].sum()

# 多种统计
df.groupby('产品')['销量'].agg(['sum', 'mean', 'count'])
```

**示例代码**: [04_statistics.py](examples/04_statistics.py)

## 练习题

### 练习1：数据读取与查看
- 文件：[exercise_01.py](exercises/exercise_01.py)
- 答案：[solution_01.py](exercises/solution_01.py)

### 练习2：数据筛选
- 文件：[exercise_02.py](exercises/exercise_02.py)
- 答案：[solution_02.py](exercises/solution_02.py)

## 运行示例

```bash
# 运行示例代码
cd /mnt/d/Github/python-data-analysis/pandas
python examples/01_basic_read.py

# 运行测试
pytest tests/

# 运行练习题答案
python exercises/solution_01.py
```

## 项目结构

```
pandas/
├── data/              # 测试数据
│   ├── students.csv
│   └── sales.csv
├── examples/          # 示例代码
│   ├── 01_basic_read.py
│   ├── 02_filter.py
│   ├── 03_operations.py
│   └── 04_statistics.py
├── exercises/         # 练习题
│   ├── exercise_01.py
│   ├── exercise_02.py
│   ├── solution_01.py
│   └── solution_02.py
├── tests/            # 测试代码
│   ├── test_basic.py
│   └── test_operations.py
├── requirements.txt
└── pandas.md         # 本教程
```

## 学习路径

1. 阅读教程文档
2. 运行示例代码
3. 完成练习题
4. 运行测试验证

