# 第 06/07 周数据：分类与回归模型

第 06 周用 `GermanCredit.csv` 和 `Churn_Modelling.csv` 练习分类模型，第 07 周用 `BostonHousing.csv` 练习回归模型与业务解释。

## `GermanCredit.csv`

- 来源：GitHub `selva86/datasets`
- 规模：1,000 行 × 21 列
- 领域：信贷风险
- 目标变量：`credit_risk`
- 主要字段：`duration`、`amount`、`age`、`credit_history`、`savings`、`housing`
- 用途：分类模型，理解准确率、召回率和业务代价。

## `BostonHousing.csv`

- 来源：GitHub `selva86/datasets`
- 规模：506 行 × 14 列
- 领域：房地产
- 目标变量：`medv`
- 主要字段：`crim`、`rm`、`lstat`、`ptratio`、`nox`、`age`
- 用途：线性回归，比较预测值与基线，计算 MAE、R2。

## `Churn_Modelling.csv`

- 来源：GitHub `selva86/datasets`
- 规模：10,000 行 × 14 列
- 领域：银行客户流失
- 目标变量：`Exited`
- 主要字段：`CreditScore`、`Geography`、`Age`、`Tenure`、`Balance`、`NumOfProducts`、`IsActiveMember`
- 用途：客户流失分类，练习类别不平衡、特征选择和业务解释。
