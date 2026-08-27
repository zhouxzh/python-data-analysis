# Week 2：pandas 数据结构与读取

> 总时长：180 分钟（3 节 × 60 分钟）
> 数据：`data/air_quality_simple.csv`
> 关键概念：Series、DataFrame、`read_csv`、`head/info/dtypes`、列选择、行过滤、新增列、`groupby` 初体验

## 本节结束后学生能做到

- 区分 Series 和 DataFrame。
- 读取 CSV / Excel，并检查数据形状、类型和缺失。
- 用列名选择数据、用条件过滤行、新增计算列。
- 用 DSH 生成第一版 pandas 代码，并验证代码是否真的回答了问题。

## 课前准备

- `data/air_quality_simple.csv` 确认可读。
- 学生项目目录中已有 `scripts/` 和 `output/`。

---

## 第 1 节：DataFrame 是什么（60 分钟）

### 目标

建立“表格 = DataFrame”的心智模型，掌握最基本的读取与检查命令。

### 教师演示（10 分钟）

让 DSH 读取并检查：

```text
请读取 data/air_quality_simple.csv。
输出：
1. shape；
2. 前 5 行；
3. 每列 dtype；
4. 每列缺失数量。
只使用 pandas，不要修改原文件。
```

### 学生练习（35 分钟）

1. 在 DSH 中完成上面的检查，并把输出截图/复制到 notebook。
2. 自己回答三个问题：
   - 这份数据有多少个城市？
   - 哪些列是数值？
   - PM25 列有没有缺失？
3. 让 DSH 生成以下代码并运行：
   ```python
   import pandas as pd
   df = pd.read_csv("data/air_quality_simple.csv")
   print(df.shape)
   print(df.info())
   print(df.head())
   ```
4. 分别尝试把 `head()` 改成 `tail()`、把 `info()` 改成 `dtypes`，观察差异。

### 复盘（15 分钟）

- `shape` 的行列顺序是什么？（先行列后）
- 为什么检查缺失要在分析前做？

### 本节关键提示词

```text
请用 pandas 读取 data/air_quality_simple.csv，
然后输出 shape、dtypes、缺失数、前 5 行。
最后用一句中文解释：这份数据适合回答“哪些城市空气污染更严重”吗？
```

---

## 第 2 节：选列、过滤和新增列（60 分钟）

### 目标

掌握 DataFrame 的三种基础操作：选列、条件过滤、新增计算列。

### 教师演示（10 分钟）

让 DSH 完成一个“带业务语义”的任务：

```text
基于 data/air_quality_simple.csv：
1. 只保留 city、PM25、PM10；
2. 新增一列 pm10_ratio = PM10 / PM25；
3. 筛选出 PM25 大于 50 的行；
4. 输出结果表。
```

### 学生练习（35 分钟）

1. 运行上面的代码，检查 `pm10_ratio` 是否出现异常值。
2. 自己新增一列 `pollution_index = PM25 + PM10 + NO2 + SO2`。
3. 筛选出 `pollution_index > 150` 的城市，并说明这代表什么。
4. 让 DSH 解释为什么 `df["PM25"]` 和 `df[["PM25"]]` 的结果不一样。

### 复盘（15 分钟）

- “新增列”前必须确认数值类型；如果 PM25 是字符串，计算会怎样？
- 用 AI 生成代码时，如果它直接写 `df["pm25"]` 而原列是 `PM25`，会发生什么？这告诉我们：AI 可能猜错字段名，必须运行验证。

### 本节关键提示词

```text
基于 data/air_quality_simple.csv，写 pandas 代码：
1. 新增 pollution_index = PM25 + PM10 + NO2 + SO2；
2. 筛选 pollution_index > 150 的行；
3. 按 pollution_index 从高到低排序；
4. 输出 5 行结果和一句解释。
```

---

## 第 3 节：Mini case：找出 PM2.5 最高的 3 个城市（60 分钟）

### 目标

用 `groupby` 完成分组聚合，并学会检查“聚合结果是否被小样本误导”。

### 学生练习（35 分钟）

1. 用 DSH 运行：

```text
请基于 data/air_quality_simple.csv：
用 groupby("city")["PM25"].mean() 计算每个城市平均 PM25，
保留城市和平均值，按平均值降序，输出前 3 个城市。
同时输出每个城市的样本行数。
```

2. 检查输出：
   - 平均 PM25 是否和原表一致；
   - 每个城市有多少行；
   - 如果某个城市只有 1 行，能否代表整月？
3. 把结果保存为 `projects/<姓名>/output/city_pm25_rank.csv`。
4. 让 DSH 审查代码，是否满足“只用 pandas、输出可复现、有样本量”三个要求。

### 复盘（15 分钟）

- 一个城市的样本量只有 1 行时，均值有什么风险？
- 数据里已经有 `province`，为什么不能只用城市名做结论？

### 作业

让 DSH 生成一份“数据概览卡片”，保存为 `projects/<姓名>/output/data_card.md`，至少包含：

- 行数、列数、每列类型；
- 每列缺失数；
- 城市数；
- 每个数值列的 min / max / mean；
- 你发现的一个奇怪现象。

## 评分要点

| 项目 | 要求 |
|---|---|
| 读取 | 能解释 `shape`、`dtypes`、缺失 |
| 操作 | 能完成选列、过滤、新增列 |
| 聚合 | 能用 `groupby` 输出排名并报告样本量 |
| AI 协作 | 至少一次把真实输出反馈给 DSH 并修改代码 |
