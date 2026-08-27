# Week 6：合并、分组与迷你项目

> **本章导读**
> 时长：3 节课，每节 60 分钟
> 数据：`data/air_quality_simple.csv`、`data/city_info.csv`
> 你将学到：用 `merge` 连接两张表，用 `groupby` 做多指标汇总，并完成一个 120 字迷你报告
> 本周产出：`projects/<姓名>/mini_project/`

## 1. 跟着老师做

### 1.1 先确定要回答的问题

```text
人口更多的区域，PM2.5 是否更严重？
```

要回答这个问题，单张空气表不够，还要把城市人口表合并进来。

### 1.2 发给 DSH 的第一版提示词

```text
请读取 data/air_quality_simple.csv 和 data/city_info.csv。
按 city 合并，输出：
1. 合并后 shape；
2. 哪些城市只在空气表、哪些只在 city_info；
3. 合并后是否有空值；
4. 按 region 汇总：城市数、平均 PM25、人口加权平均 PM25、总人口。
不要修改原文件。
```

### 1.3 合并两张表

```python
import pandas as pd

air = pd.read_csv('data/air_quality_simple.csv')
info = pd.read_csv('data/city_info.csv')

merged = air.merge(info, on='city', how='left')

print('merged shape:', merged.shape)
print()
print('缺失值:')
print(merged.isna().sum())
```

预期输出：

```text
merged shape: (10, 10)

缺失值:
city                  0
province_x            0
PM25                  1
PM10                  0
NO2                   0
SO2                   0
province_y            0
region                0
population_million    0
area_km2              0
```

### 1.4 按 region 汇总

```python
merged['weighted_pm25'] = merged['PM25'] * merged['population_million']

region = merged.groupby('region').apply(
    lambda g: pd.Series({
        'cities': g['city'].nunique(),
        'mean_pm25': g['PM25'].mean(),
        'population_weighted_pm25': g['weighted_pm25'].sum() / g['population_million'].sum(),
        'total_population_million': g['population_million'].sum()
    }),
    include_groups=False
).round(2)

print(region)
```

预期输出：

```text
        cities  mean_pm25  population_weighted_pm25  total_population_million
region
华东       2.0       41.5                     42.67                      37.5
华中       1.0       48.0                     48.00                      13.6
华北       1.0       60.0                     60.00                      21.9
华南       3.0       42.5                     24.01                      46.3
西北       1.0       63.0                     63.00                      13.0
西南       2.0       56.5                     56.19                      53.1
```

### 1.5 解读

```text
简单平均 PM25 和人口加权 PM25 不同。
华南简单平均 42.5，人口加权只有 24.01，
说明华南的高污染样本集中在人口较少的城市。
结论必须写清楚：我用的是哪种平均。
```

## 2. 你自己动手做

1. 新建 `projects/<姓名>/mini_project/`。
2. 在 DSH 中使用 plan mode：

```text
/plan 我要回答：人口更多的区域，PM2.5 是否更严重？
请先调查数据字段，给出：指标定义、合并方式、统计方法、图表类型、验收标准。
先不要改文件。
```

3. 确认方案后，让 DSH 生成代码并运行。
4. 输出：
   - 1 张图；
   - 120 字以内结论；
   - 1 条局限。

自己动手时建议用这个提示词：

```text
请审查我的 mini_project：
1. 合并是否报告了未匹配和空值；
2. 指标是否写清是简单平均还是人口加权；
3. 结论是否带样本量和局限；
4. 给出修改后的代码。
```

## 3. 验证清单

- [ ] 合并前后 shape 和未匹配城市有记录
- [ ] 汇总表包含城市数、样本量和加权指标
- [ ] 结论写清指标定义
- [ ] 原文件未修改
- [ ] 迷你报告可复现

## 4. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| merge 后行数变多/变少 | 键不唯一或 how 选错 | 先检查键唯一性 |
| 出现 province_x / province_y | 两张表都有同名列 | 合并前重命名或选择列 |
| 简单平均和加权平均混用 | 没写口径 | 报告中明确指标公式 |
| 未匹配城市被悄悄丢弃 | 默认 inner/left 选择 | 主动报告未匹配列表 |
| 人口加权结果异常 | 缺失人口被忽略 | 检查缺失和总人口 |

## 5. 作业

提交 mini project，要求包含：

```text
projects/<姓名>/mini_project/
  README.md
  scripts/
  output/
```

README 写清楚数据来源、指标定义和 3 个步骤。

## 评分要点

| 项目 | 要求 |
|---|---|
| 合并 | 能解释未匹配和空值 |
| 汇总 | 能同时输出多指标，理解加权均值 |
| 项目 | 问题明确、代码可运行、结论 ≤120 字 |
| 局限 | 至少写出 1 条样本或口径局限 |
