# Week 3：数据清洗与审计

> **本章导读**
> 时长：3 节课，每节 60 分钟
> 数据：`data/air_quality_dirty.csv`
> 你将学到：先审计再清洗，把清洗步骤写成函数，并识别 `unknown` 这种伪缺失
> 本周产出：`projects/<姓名>/cleaning_report.md`

## 1. 跟着老师做

### 1.1 先确定要回答的问题

```text
这份空气质量数据能不能直接做分析？
```

直接运行前，必须先回答三个问题：

- 有没有重复行？
- 有没有缺失值？
- 日期和数值列的类型是否统一？

### 1.2 发给 DSH 的第一版提示词

```text
请审计 data/air_quality_dirty.csv，不要修改原文件。
输出：
1. shape、dtypes；
2. 重复行数；
3. 每列缺失数；
4. 日期列样例；
5. 列出 3 个必须处理的数据质量问题。
```

### 1.3 数据质量审计

```python
import pandas as pd

dirty = pd.read_csv('data/air_quality_dirty.csv')

print('shape:', dirty.shape)
print('重复行数:', dirty.duplicated().sum())
print()
print('缺失值:')
print(dirty.isna().sum())
print()
print('日期样例:')
print(dirty['date'].tail(8))
```

预期输出：

```text
shape: (14, 7)
重复行数: 1

缺失值:
date    0
city    0
province    0
PM25    2
PM10    1
NO2     1
SO2     0
dtype: int64

日期样例:
6     2025-09-02
7     2025-09-02
8     2025-09-03
9     2025-09-03
10    2025-09-03
11    2025/09/03
12    2025-09-03
13    2025-09-03
```

可以看到三类问题：

1. 第 0 行和第 1 行完全重复。
2. `PM25`、`PM10`、`NO2` 存在缺失。
3. 日期列混合了 `2025-09-03` 和 `2025/09/03`。

### 1.4 清洗函数

```python
numeric_cols = ['PM25', 'PM10', 'NO2', 'SO2']


def clean_air(df):
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned['date_parsed'] = pd.to_datetime(cleaned['date'], errors='coerce')
    cleaned = cleaned.dropna(subset=['date_parsed'])
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    return cleaned


cleaned = clean_air(dirty)

print('清洗前:', dirty.shape)
print('清洗后:', cleaned.shape)
print('重复行:', cleaned.duplicated().sum())
print('日期缺失:', cleaned['date_parsed'].isna().sum())
```

预期输出：

```text
清洗前: (14, 7)
清洗后: (12, 8)
重复行: 0
日期缺失: 0
```

### 1.5 清洗报告模板

```markdown
# 清洗报告

- 数据：data/air_quality_dirty.csv
- 原始行数：14
- 清洗后行数：12
- 删除重复：1 行
- 删除日期无效：1 行
- 数值列缺失：PM25 2，PM10 1，NO2 1
- 保留的缺失策略：先观察，不在没依据时强行填充
```

## 2. 你自己动手做

1. 新建 `projects/<姓名>/cleaning_report.md`。
2. 让 DSH 把你的清洗步骤整理成函数 `clean_air(df)`。
3. 分别试 `dropna()` 删除所有缺失和 `fillna(0)` 填充 PM25，比较清洗后行数和均值变化。
4. 用中文解释：为什么清洗策略会影响后面的结论？

自己动手时建议用这个提示词：

```text
请审查我的清洗函数：
1. 是否误删了有价值的数据；
2. 缺失值是否应该填充；
3. unknown 是否应该转成 pd.NA；
4. 给出修改后的函数和清洗前后对比表。
```

## 3. 验证清单

- [ ] 原文件没有被修改
- [ ] 清洗前后 shape、重复行、缺失数都能对比
- [ ] 清洗步骤被封装成函数
- [ ] 报告写清了每个删除或填充的理由
- [ ] 脚本可用 `python scripts/week03_practice.py` 运行

## 4. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 重复行没删掉 | 只看了 head | 使用 `duplicated().sum()` |
| 日期无法解析 | 存在多种格式 | `to_datetime(errors='coerce')` 后检查 NaT |
| 删掉太多行 | 缺失去重策略太激进 | 先输出缺失分布再决定 |
| `unknown` 被当成普通类别 | 是伪缺失 | 判断业务语义后再处理 |
| 填充让均值失真 | 默认 `fillna` | 写清策略并保留原始缺失数 |

## 5. 作业

让 DSH 对同一份数据做“自动审计”，然后人工核对，列出 AI 遗漏的 2 个质量问题，并说明你是如何发现的。

## 评分要点

| 项目 | 要求 |
|---|---|
| 审计 | 能发现重复、缺失、日期格式三类问题 |
| 清洗 | 函数可运行，原文件未改 |
| 对比 | 输出清洗前后行数和缺失 |
| 伪缺失 | 能识别 unknown 与 NaN 的区别 |
