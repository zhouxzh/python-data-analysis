# Week 2：pandas 数据结构与读取

> **本章导读**
> 时长：3 节课，每节 60 分钟
> 数据：`data/air_quality_simple.csv`
> 你将学到：让 DSH 读取 CSV、检查 shape/dtypes/缺失、做第一版分组统计，并发现“数据够不够回答你的问题”
> 本周产出：`projects/<姓名>/output/data_card.md`

## 1. 跟着老师做

### 1.1 先确定要回答的问题

```text
哪些城市 PM2.5 更高？
```

这个问题的坑是：每个城市只有 1 行数据，而且广州的 PM2.5 缺失。所以我们不能只输出排名，还要同时输出样本量。

### 1.2 发给 DSH 的第一版提示词

```text
请读取 data/air_quality_simple.csv。
输出：
1. shape、dtypes、缺失值；
2. 前 5 行；
3. 每个城市 PM25 的均值；
4. PM25 最高的 3 个城市；
5. 用一句话提醒我这份数据能不能回答“哪些城市 PM2.5 更高”。
```

### 1.3 先审计，再排名

```python
import pandas as pd

air = pd.read_csv('data/air_quality_simple.csv')

print('shape:', air.shape)
print(air.dtypes)
print()
print('缺失值:')
print(air.isna().sum())
```

预期输出：

```text
shape: (10, 6)
city         object
province     object
PM25        float64
PM10          int64
NO2           int64
SO2           int64
缺失值:
city        0
province    0
PM25        1
PM10        0
NO2         0
SO2         0
```

### 1.4 分组统计并报告样本量

```python
summary = (
    air.groupby('city')['PM25']
    .agg(['mean', 'count'])
    .sort_values('mean', ascending=False)
    .round(1)
)

print(summary)
```

预期输出：

```text
      mean  count
city
西安   63.0      1
北京   60.0      1
成都   58.0      1
重庆   55.0      1
佛山   50.0      1
武汉   48.0      1
上海   45.0      1
杭州   38.0      1
深圳   35.0      1
广州    NaN      0
```

### 1.5 解读

```text
按现有数据，PM2.5 最高的 3 个城市是西安、北京、成都。
但每个城市只有 1 行，mean 只是一行的值；
广州因缺失没有进入排名。
所以这个结论只能说“当前快照”，不能说“城市整体空气质量排名”。
```

## 2. 你自己动手做

1. 新建 `projects/<姓名>/output/data_card.md`。
2. 让 DSH 生成一份数据概览卡片，至少包含：
   - 行数、列数、每列类型；
   - 每列缺失数；
   - 城市数；
   - 每个数值列的 min / max / mean；
   - 你发现的一个奇怪现象。
3. 把排名改成 PM10，看前 3 是否变化。
4. 让 DSH 审查：这份数据真的能支持“城市污染排名”吗？

自己动手时建议用这个提示词：

```text
请审查我的 data_card.md 和 groupby 代码：
1. 是否输出样本量 count；
2. 是否处理 NaN；
3. 是否适合回答“PM2.5 最高的 3 个城市”；
4. 给出修改后的代码。
```

## 3. 验证清单

- [ ] `shape`、`dtypes`、缺失值都能输出
- [ ] 每个分组统计包含 `count`
- [ ] 排名结论注明“每城只有 1 行”
- [ ] 广州缺失被明确报告
- [ ] 脚本可用 `python scripts/week02_practice.py` 运行

## 4. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 筛选 PM25 > 50 少了城市 | NaN 不参与比较 | 先看缺失，再筛选 |
| groupby 排名出现 NaN | 该组没有有效值 | `dropna()` 或报告缺失 |
| 结论像“整体排名” | 样本太少 | 每次写结论都带样本量 |
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
| 验证 | 能发现每个城市只有 1 行这个局限 |
