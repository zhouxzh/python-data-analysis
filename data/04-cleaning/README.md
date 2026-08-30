# 第 04/05 周数据：清洗与可视化案例

第 05 周识别并处理重复、缺失、类型错误、异常值和伪缺失值；`Life_Expectancy_Data.csv` 同时在第 04 周用于 pandas 集成绘图。

## `telco_customer_churn.csv`

- 来源：HuggingFace `aai510-group1/telco-customer-churn` 的 `train.csv`
- 镜像下载：`https://hf-mirror.com/datasets/aai510-group1/telco-customer-churn/resolve/main/train.csv`
- 规模：4,225 行 × 52 列
- 领域：电信
- 主要字段：`Customer ID`、`Churn`、`Tenure in Months`、`Monthly Charge`、`Total Charges`、`Contract`、`Internet Type`
- 课堂用法：重复记录、缺失值、字符串和数值类型转换，以及流失标签审计。

## `Cars93_miss.csv`

- 来源：GitHub `selva86/datasets`
- 规模：93 行 × 27 列
- 领域：汽车
- 课堂用法：小规模但真实带缺失的数据集，适合人工核对缺失位置和缺失原因。

## `Life_Expectancy_Data.csv`

- 来源：GitHub `selva86/datasets`
- 规模：1,649 行 × 22 列
- 领域：公共健康
- 主要字段：`Country`、`Year`、`Status`、`Life expectancy `、`GDP`、`Population`、`Schooling`
- 课堂用法：多国家多年份面板数据，练习字段名清理、异常值和逻辑一致性检查。
