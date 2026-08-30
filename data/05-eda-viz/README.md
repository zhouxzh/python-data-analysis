# 第 04 周数据：数据可视化

这些数据在第 04 周用于 matplotlib 和 pandas 集成绘图练习。

## `diamonds.csv`

- 来源：GitHub `selva86/datasets`
- 规模：53,940 行 × 10 列
- 领域：零售/消费
- 主要字段：`carat`、`cut`、`color`、`clarity`、`price`、`depth`、`table`
- 用途：直方图、箱线图、价格与重量关系，练习异常值识别。

## `midwest.csv`

- 来源：GitHub `selva86/datasets`
- 规模：437 行 × 28 列
- 领域：人口统计
- 主要字段：`county`、`state`、`poptotal`、`popdensity`、`percwhite`、`percbelowpoverty`
- 用途：分类对比、人口密度分布、贫困率与受教育程度的关系。

## `energy_dataset.csv`

- 来源：HuggingFace `vitaliy-sharandin/energy-consumption-hourly-spain`
- 镜像下载：`https://hf-mirror.com/datasets/vitaliy-sharandin/energy-consumption-hourly-spain/resolve/main/energy_dataset.csv`
- 规模：35,064 行 × 29 列
- 领域：能源时间序列
- 主要字段：`time`、`total load actual`、`price actual`、各类发电量
- 用途：折线图、缺失值处理、时间聚合，以及电价和负荷趋势。
