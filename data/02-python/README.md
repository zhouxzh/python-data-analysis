# 第 02 周数据：Python 编程基础

第 02 周不引入 pandas，用 Python 标准库和基础语法处理三个不同领域的小数据集。

## `stock_price.csv`

- 来源：GitHub `selva86/datasets`
- 规模：252 行 × 2 列，字段 `Date`、`Price`
- 领域：金融时间序列
- 课堂用法：列表、循环、日期字符串和数值计算，例如计算平均价格、最高最低价格。

## `supermarket_sales.csv`

- 来源：GitHub `selva86/datasets`
- 规模：1,000 行 × 17 列
- 领域：零售
- 主要字段：`Branch`、`City`、`Customer type`、`Product line`、`Unit price`、`Quantity`、`Total`、`Rating`
- 课堂用法：字典、条件判断、函数和汇总，例如按城市或商品线计算销售额。

## `breast_cancer.csv`

- 来源：GitHub `selva86/datasets`
- 规模：699 行 × 11 列
- 领域：医疗
- 主要字段：`Cl.thickness`、`Cell.size`、`Cell.shape`、`Bare.nuclei`、`Class`
- 课堂用法：读文件、错误处理和基础统计。注意 `Bare.nuclei` 中可能有 `?`，作为异常值判断练习。
