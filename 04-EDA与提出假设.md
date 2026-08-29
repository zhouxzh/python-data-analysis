# Week 4：EDA 与提出假设

> **本章导读**
> 时长：3 节课，每节 45 分钟
> 数据：`data/nyc_airbnb.csv`
> 你将学到：带着明确问题做 EDA，报告样本量，识别价格分层、缺失和异常值，并区分“发现”与“假设”
> 本周产出：`projects/<姓名>/output/eda_findings.md`

## 1. 跟着老师做

### 1.1 先确定要回答的问题

```text
哪些因素看起来和 Airbnb 价格有关？
```

先把它拆成可以计算的小问题：

- 不同行政区的价格差多少？
- 不同房型的价格差多少？
- 高价房源（price > 200）主要集中在哪里？
- 没有评论的房源占多少？

### 1.2 读取数据并做基础检查

```python
import pandas as pd

df = pd.read_csv('data/nyc_airbnb.csv')

print('shape:', df.shape)
print(df[['neighbourhood_group', 'room_type', 'price',
          'reviews_per_month']].isna().sum())
```

预期输出：

```text
shape: (48895, 16)
neighbourhood_group       0
room_type                 0
price                     0
reviews_per_month     10052
```

### 1.3 带着问题做 EDA

```python
print('按行政区:')
print(df.groupby('neighbourhood_group')['price']
      .agg(['mean', 'median', 'count'])
      .round(2)
      .sort_values('mean', ascending=False))
print()
print('按房型:')
print(df.groupby('room_type')['price']
      .agg(['mean', 'median', 'count'])
      .round(2)
      .sort_values('mean', ascending=False))
```

预期输出：

```text
按行政区:
                  mean  median  count
neighbourhood_group
Manhattan        196.88   150.0  21661
Brooklyn         124.38    90.0  20104
Staten Island    114.81    75.0    373
Queens            99.52    75.0   5666
Bronx             87.50    65.0   1091

按房型:
                  mean  median  count
room_type
Entire home/apt  211.79   160.0  25409
Private room      89.78    70.0  22326
Shared room       70.13    45.0   1160
```

### 1.4 高价比例与缺失

```python
df['high_price'] = df['price'] > 200

print('高价房源比例:')
print(df.groupby('room_type')['high_price']
      .agg(['mean', 'count'])
      .round(4))
print()
print('reviews_per_month 缺失数:')
print(df.groupby('room_type')['reviews_per_month']
      .apply(lambda s: s.isna().sum()))
```

预期输出：

```text
高价房源比例:
                  mean  count
room_type
Entire home/apt  0.3006  25409
Private room     0.0317  22326
Shared room      0.0345   1160

reviews_per_month 缺失数:
room_type
Entire home/apt    5077
Private room       4661
Shared room         314
Name: reviews_per_month, dtype: int64
```

### 1.5 解读

```text
发现 1：Manhattan 平均价格约 196.88，中位数 150，明显高于其他行政区。
发现 2：Entire home/apt 平均价格约 211.79，30.06% 属于高价房源。
发现 3：reviews_per_month 缺失 10052 行，说明约五分之一的房源没有评论记录。

假设 1：行政区价差可能来自房源位置和房型结构，而不只是“Manhattan 这个名字”。
假设 2：高价房源集中在 Manhattan + Entire home/apt，需要交叉表验证。
假设 3：没有评论的房源可能是新上架或较少被预订，不能直接当成“价格低”。
```

## 2. 你自己动手做

1. 新建 `projects/<姓名>/output/eda_findings.md`。
2. 写自己的 3 个发现 + 3 个假设，每条注明字段、计算方式、样本量。
3. 让 DSH 扮演反方，尝试推翻你的每个结论。
4. 检查价格与 `minimum_nights`、`number_of_reviews`、`availability_365` 的相关性：

```python
print(df[['price', 'minimum_nights', 'number_of_reviews', 'availability_365']]
      .corr()['price']
      .round(3))
```

预期输出：

```text
price               1.000
minimum_nights      0.043
number_of_reviews  -0.048
availability_365    0.082
Name: price, dtype: float64
```

自己动手时建议用这个提示词：

```text
请审查我的 EDA 报告：
1. 每个发现是否有数据依据；
2. 是否缺少样本量；
3. 是否把 0 和缺失混为一谈；
4. 是否把相关性写成因果；
5. 给出反方意见。
```

## 3. 验证清单

- [ ] 每个分组表包含 `count`
- [ ] 结论区分“发现”和“假设”
- [ ] 样本量被写进结论
- [ ] 提到缺失、价格异常和“相关不等于因果”
- [ ] 脚本可用 `python scripts/week04_practice.py` 运行

## 4. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 只看均值不看中位数和样本量 | 小样本或异常值被高估 | 分组表加 `median` 和 `count` |
| 把相关说成因果 | 没有对照组 | 写“发现/假设”而非“原因” |
| 用均价直接比较行政区 | 房型结构不同 | 交叉表或先控制房型 |
| 忽略缺失 | 无评论房源被排除 | 报告缺失数量 |
| `price == 0` 当成正常值 | 业务上不可能 | 标记异常并说明处理 |

## 5. 作业

把 3 个发现 + 3 个假设写成 `projects/<姓名>/output/eda_findings.md`，并让 DSH 做一次“反方审查”。

## 评分要点

| 项目 | 要求 |
|---|---|
| 问题 | 每个分析都对应一个明确问题 |
| 统计 | 分组表包含样本量和比例 |
| 风险 | 能识别缺失、异常值、相关不等于因果 |
| 表达 | 区分已验证发现与待验证假设 |
