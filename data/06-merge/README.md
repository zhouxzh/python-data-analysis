# 第 05 周数据：合并与分组聚合

第 05 周练习多表合并、分组聚合和透视表，并基于多个数据源完成小型业务问题。

## `norway_new_car_sales_by_make.csv`

- 来源：GitHub `selva86/datasets`
- 规模：4,377 行 × 5 列
- 领域：汽车销售
- 主要字段：`Year`、`Month`、`Make`、`Quantity`、`Pct`

## `norway_new_car_sales_by_model.csv`

- 来源：GitHub `selva86/datasets`
- 规模：2,694 行 × 6 列
- 领域：汽车销售
- 主要字段：`Year`、`Month`、`Make`、`Model`、`Quantity`、`Pct`
- 用途：以 `Year`、`Month`、`Make` 为键合并两张表，比较按品牌和按车型汇总的一致性。

## `MarketArrivals.csv`

- 来源：GitHub `selva86/datasets`
- 规模：10,227 行 × 10 列
- 领域：零售/客流量
- 主要字段：`market`、`month`、`year`、`quantity`、`priceMin`、`priceMax`、`state`、`city`、`date`
- 用途：分组聚合、透视表，以及不同市场和月份的到货量对比。
