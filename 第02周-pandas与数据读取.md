# Week 2：pandas 数据结构与读取

> **本章导读**
> 时长：3 节课，每节 60 分钟
> 数据：`data/nyc_airbnb.csv`
> 你将学到：让 DSH 读取 CSV、检查 shape/dtypes/缺失、做第一版分组统计，并发现“数据够不够回答你的问题”
> 本周产出：`projects/<姓名>/output/data_card.md`

## 1. 跟着老师做

### 1.1 先确定要回答的问题

```text
哪种房型的平均价格更高？
```

这个问题不能只看一个平均值：还要看每种房型有多少样本、有没有缺失、有没有异常价格。否则一条 10000 美元的房源就能把平均值拉高。

### 1.2 发给 DSH 的第一版提示词

```text
请读取 data/nyc_airbnb.csv。
输出：
1. shape、dtypes、缺失值；
2. 前 5 行；
3. 按 room_type 分组的 price 均值和样本量；
4. price 最高的 3 个房源；
5. 用一句话提醒我这份数据能不能回答“哪种房型平均价格更高”。
```

### 1.3 先审计，再排名

```python
import pandas as pd

df = pd.read_csv('data/nyc_airbnb.csv')

print('shape:', df.shape)
print(df.dtypes)
print()
print('缺失值:')
print(df.isna().sum())
```

预期输出：

```text
shape: (48895, 16)
id                                  int64
name                               object
host_id                             int64
host_name                          object
neighbourhood_group                object
neighbourhood                      object
latitude                          float64
longitude                         float64
room_type                          object
price                               int64
minimum_nights                      int64
number_of_reviews                   int64
last_review                        object
reviews_per_month                 float64
calculated_host_listings_count      int64
availability_365                    int64

缺失值:
id                                    0
name                                 16
host_id                               0
host_name                            21
neighbourhood_group                   0
neighbourhood                         0
latitude                              0
longitude                             0
room_type                             0
price                                 0
minimum_nights                        0
number_of_reviews                     0
last_review                       10052
reviews_per_month                 10052
calculated_host_listings_count        0
availability_365                      0
```

### 1.4 分组统计并报告样本量

```python
summary = (
    df.groupby('room_type')['price']
    .agg(['mean', 'count'])
    .sort_values('mean', ascending=False)
    .round(1)
)

print(summary)
```

预期输出：

```text
                 mean  count
room_type
Entire home/apt  211.8  25409
Private room      89.8  22326
Shared room       70.1   1160
```

再看价格最高的 3 个房源：

```python
print(df[['name', 'room_type', 'neighbourhood_group', 'price']]
      .sort_values('price', ascending=False)
      .head(3))
```

预期输出包含价格 `10000` 的房源。这些异常值必须先标记，不能直接当成“正常价格”。

### 1.5 解读

```text
按平均值看，Entire home/apt 的平均价格最高，约 211.8 美元/晚；
它有 25409 个样本，所以结论比较稳定。
但 price 的最大值是 10000，数据里存在明显异常值；
先报告“有异常”，再决定是否过滤。
```

## 2. 你自己动手做

1. 新建 `projects/<姓名>/output/data_card.md`。
2. 让 DSH 生成一份数据概览卡片，至少包含：
   - 行数、列数、每列类型；
   - 每列缺失数；
   - 街区数、行政区数、房型数；
   - 每个数值列的 min / max / mean；
   - 你发现的一个奇怪现象。
3. 把分组改成 `neighbourhood_group`，看哪个区平均价格最高。
4. 让 DSH 审查：这份数据真的能支持“哪种房型最贵”吗？

自己动手时建议用这个提示词：

```text
请审查我的 data_card.md 和 groupby 代码：
1. 是否输出样本量 count；
2. 是否处理 NaN；
3. 是否适合回答“哪种房型平均价格更高”；
4. 是否标记了价格为 0 或超过 1000 的异常；
5. 给出修改后的代码。
```

## 3. 验证清单

- [ ] `shape`、`dtypes`、缺失值都能输出
- [ ] 每个分组统计包含 `count`
- [ ] 排名结论注明样本量
- [ ] `last_review` 和 `reviews_per_month` 的缺失被明确报告
- [ ] 价格异常值被标记
- [ ] 脚本可用 `python scripts/week02_practice.py` 运行

## 4. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 只看均价不看样本量 | Shared room 只有 1160 行 | 分组表加 `count` |
| groupby 出现 NaN | 该组没有有效值或数据有缺失 | `dropna()` 或报告缺失 |
| 结论像“全纽约价格” | 房型/区域混合 | 写清分组口径 |
| 价格平均值被异常值拉高 | 存在 0 和 10000 | 先看 `price.describe()` 再决定 |
| AI 把填充值当成真实值 | 缺失处理默认不透明 | 明确填充策略并写进报告 |

## 5. 作业

让 DSH 生成一份“数据概览卡片”，保存为 `projects/<姓名>/output/data_card.md`，并在最后写一句：

```text
这份数据能回答什么，不能回答什么。
```

## 评分要点

| 项目 | 要求 |
|---|---|
| 读取 | 能解释 `shape`、`dtypes`、缺失 |
| 操作 | 能完成选列、过滤、新增列 |
| 聚合 | 能输出排名并报告样本量 |
| 验证 | 能发现房型样本量差异和价格异常 |
