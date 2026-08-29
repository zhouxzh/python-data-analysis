# 课程数据说明

## 主数据集：`nyc_airbnb.csv`

来源：HuggingFace 数据集 `gradio/NYC-Airbnb-Open-Data`，原始文件 `AB_NYC_2019.csv`。

下载地址：

```text
https://huggingface.co/datasets/gradio/NYC-Airbnb-Open-Data/resolve/main/AB_NYC_2019.csv
```

授权：AFL-3.0（Academic Free License 3.0），可用于教学、研究和再分发，使用时保留来源说明。

规模：48,895 行 × 16 列，约 7MB。

字段说明：

| 字段 | 含义 |
|---|---|
| `id` | 房源 ID |
| `name` | 房源名称 |
| `host_id` | 房东 ID |
| `host_name` | 房东姓名 |
| `neighbourhood_group` | 纽约市行政区（Bronx/Brooklyn/Manhattan/Queens/Staten Island） |
| `neighbourhood` | 具体街区 |
| `latitude` / `longitude` | 房源经纬度 |
| `room_type` | 房型：Entire home/apt、Private room、Shared room |
| `price` | 每晚价格（美元） |
| `minimum_nights` | 最少入住晚数 |
| `number_of_reviews` | 评论数量 |
| `last_review` | 最近一次评论日期，无评论时为缺失 |
| `reviews_per_month` | 每月平均评论数，无评论时为缺失 |
| `calculated_host_listings_count` | 同一房东在平台上的房源数 |
| `availability_365` | 一年内可订天数 |

已知数据问题：`name` 缺失 16 行，`host_name` 缺失 21 行，`last_review` 和 `reviews_per_month` 缺失 10052 行；`price` 存在 0 和大于 1000 的异常值。这些真实问题适合第 03 周清洗审计。

## 辅助表：`nyc_boroughs.csv`

第 06 周用于 `merge` 练习的行政区小表，字段来自纽约市公开统计的常见口径，课程中作为教学用近似数据，正式论文或项目应以官方最新统计为准。

## 旧数据

`data/` 中原有的成绩单、空气质量、Titanic、银行营销等文件保留为可选/旧案例，不再是第 01 周和后续各周主流程的依赖。
