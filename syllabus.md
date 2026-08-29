# 8 周课程总表（每晚 3 节，每节 45 分钟）

## 课程原则

- 每节 45 分钟按 `8 分钟演示 → 30 分钟练习 → 7 分钟复盘` 组织。
- 每节必须出现一次完整的 vibe loop：`定义问题 → 要求最小版本 → 运行 → 反馈 → 追问 → 验证`。
- 所有案例数据都在本仓库 `data/` 中，学生不需要联网下载也能完成主体练习。
- DSH 不是答案机。凡是 AI 生成的代码，学生必须运行并用自己的话解释 1 个关键步骤。
- 主数据集统一使用 `data/nyc_airbnb.csv`，旧数据文件保留为可选/进阶案例。

## Week 1：Agent 认知、环境安装与第一个数据问题

- 第 1 节：认识 AI Agent、常见 Agent、常见大模型、DeepSeek + DSH 的特点；先讲 5 条课堂红线和安全使用流程。
- 第 2 节：安装 Anaconda、Node.js、DeepSeek Harness；验证 `python` / `node` / DSH；配置 API Key 和工作区。
- 第 3 节：先体验传统 Python（`hello.py`），再用 DSH 读取 `data/nyc_airbnb.csv` 并做第一次 Vibe Coding 分析。
- 作业：完成安装并保存验证截图；写 Agent 安全说明；用 DSH 统计 `data/nyc_airbnb.csv` 并写出 3 个“这份数据能回答的问题”。

## Week 2：pandas 数据结构与读取

- 第 1 节：Series / DataFrame、`read_csv` / `read_excel`、`head` / `info` / `dtypes`。
- 第 2 节：列选择、行过滤、新增列；单位换算和简单算术。
- 第 3 节：Mini case：读取 `data/nyc_airbnb.csv`，按 `room_type` 找出平均价格最高的房型。
- 作业：让 DSH 生成一份“数据概览卡片”，包含行数、列数、每列类型、缺失值和前 5 行。

## Week 3：数据清洗与审计

- 第 1 节：识别缺失、重复、类型不一致、语义问题（0 与缺失的区别）；把审计写成清单。
- 第 2 节：`dropna` / `fillna` / `astype` / `to_datetime` / `drop_duplicates`；保存清洗函数。
- 第 3 节：Mini case：审计 `data/nyc_airbnb.csv` 的缺失、价格异常和日期列，输出清洗前后对比。
- 作业：让 DSH 对同一份数据做“双人审查”：一组用 AI 自动审计，一组人工核对，找出 AI 遗漏的质量问题。

## Week 4：EDA 与提出假设

- 第 1 节：`describe`、`value_counts`、`sort_values`、相关性；区分“描述”和“解释”。
- 第 2 节：带着问题做 EDA：行政区/房型分组、高价比例、缺失与异常值。
- 第 3 节：Mini case：分析 `data/nyc_airbnb.csv` 的价格分布，输出 3 个发现 + 3 个待验证假设。
- 作业：把“帮我做 EDA”改成“我要回答 3 个明确问题，请给我每张图对应的代码和结论”。

## Week 5：可视化表达

- 第 1 节：折线、柱状、直方、散点；一张图对应一个问题。
- 第 2 节：标题、轴标签、图例、颜色、子图；用 seaborn 快速做分组比较。
- 第 3 节：Mini case：用 `data/nyc_airbnb.csv` 做 2×2 价格面板，并让 DSH 审查图表信息是否完整。
- 作业：制作一张“数据来源、口径、样本量、结论”四要素齐全的图，写 2 句解读。

## Week 6：合并、分组与迷你项目

- 第 1 节：`merge` / `concat`、`on` / `how`、`groupby` / `agg`。
- 第 2 节：跨表整合：`nyc_airbnb.csv` + `nyc_boroughs.csv`，对比行政区价格、人口和收入。
- 第 3 节：Mini project：提出 2 个问题 → 清洗 → 分析 → 1 张图 → 120 字结论。
- 作业：用 DSH 把项目拆成“数据加载、清洗、分析、图表、报告”5 个子任务，分别给出验收标准。

## Week 7：第一个预测模型

- 第 1 节：特征 / 标签、训练集 / 测试集、为什么不能只报告训练得分。
- 第 2 节：用 `data/nyc_airbnb.csv` 跑通最小线性回归，预测 `price`，看 R² 和 MAE。
- 第 3 节：目标泄漏、样本不平衡、相关不等于因果；让 DSH 列出模型局限。
- 作业：写 100 字：这个模型能支持什么决策、不能支持什么决策。

## Week 8：结课项目与展示

- 第 1 节：选题、数据来源、问题与验收标准；用 plan mode 写出项目计划。
- 第 2 节：分小组并行执行：审计、EDA、可视化、报告；用 goal 追踪进度。
- 第 3 节：3 分钟展示 + 互评；完成“我验证了什么 / 我不确定什么 / 下一步做什么”。

## 推荐案例数据（已在仓库中）

- `data/nyc_airbnb.csv`：纽约 Airbnb 2019 挂牌数据，48,895 行 × 16 列，约 7MB，全课程主数据集。
- `data/nyc_boroughs.csv`：纽约市行政区小表，适合 Week 6 合并练习。
- `data/成绩单.xlsx`：成绩表，可选 Excel 读取热身案例。
- `data/air_quality_simple.csv`、`data/air_quality_dirty.csv`、`data/city_info.csv`、`data/synthetic_air_quality.csv`：旧空气质量案例，可选练习。
- `data/bank_marketing.zip`：UCI 银行营销数据，可选 Week 4 EDA 和 Week 7 分类。
- `data/titanic.csv`：Titanic 生存数据，可选 Week 7 分类和结课项目。
