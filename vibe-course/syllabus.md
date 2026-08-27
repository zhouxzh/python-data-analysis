# 8 周课程总表（每晚 3 节，每节 1 小时）

## 课程原则

- 每节 60 分钟按 `10 分钟演示 → 35 分钟练习 → 15 分钟复盘` 组织。
- 每节必须出现一次完整的 vibe loop：`定义问题 → 要求最小版本 → 运行 → 反馈 → 追问 → 验证`。
- 所有案例数据都在本仓库 `data/` 中，学生不需要联网下载也能完成主体练习。
- DSH 不是答案机。凡是 AI 生成的代码，学生必须运行并用自己的话解释 1 个关键步骤。

## Week 1：从 Excel 到第一个数据问题

- 第 1 节：数据分析是什么；AI 时代为什么还要学 Python；认识 DSH Web GUI、workspace 和数据文件。
- 第 2 节：Python 最基础类型与结构；让 DSH 读取 `data/成绩单.xlsx` 并做首次汇总。
- 第 3 节：Mini case：统计 5 名同学的平均分、最高分、最低分；保存第一个 notebook。
- 作业：用 DSH 生成代码，统计 `data/成绩单.xlsx` 并写出 3 个“从这份成绩能回答的问题”。

## Week 2：pandas 数据结构与读取

- 第 1 节：Series / DataFrame、`read_csv` / `read_excel`、`head` / `info` / `dtypes`。
- 第 2 节：列选择、行过滤、新增列；单位换算和简单算术。
- 第 3 节：Mini case：读取 `data/air_quality_simple.csv`，找出 PM2.5 平均最高的 3 个城市。
- 作业：让 DSH 生成一份“数据概览卡片”，包含行数、列数、每列类型、缺失值和前 5 行。

## Week 3：数据清洗与审计

- 第 1 节：识别缺失、重复、类型不一致、伪缺失 `unknown`；把审计写成清单。
- 第 2 节：`dropna` / `fillna` / `astype` / `to_datetime` / `drop_duplicates`；保存清洗函数。
- 第 3 节：Mini case：修复 `data/air_quality_dirty.csv`，输出清洗前后对比。
- 作业：让 DSH 对同一份数据做“双人审查”：一组用 AI 自动审计，一组人工核对，找出 AI 遗漏的质量问题。

## Week 4：EDA 与提出假设

- 第 1 节：`describe`、`value_counts`、`sort_values`、相关性；区分“描述”和“解释”。
- 第 2 节：带着问题做 EDA：分组转化率、交叉表、缺失与不平衡、可疑字段。
- 第 3 节：Mini case：分析 `data/bank_marketing.zip` 中的客户订阅率，输出 3 个发现 + 3 个待验证假设。
- 作业：把“帮我做 EDA”改成“我要回答 3 个明确问题，请给我每张图对应的代码和结论”。

## Week 5：可视化表达

- 第 1 节：折线、柱状、直方、散点；一张图对应一个问题。
- 第 2 节：标题、轴标签、图例、颜色、子图；用 seaborn 快速做分组比较。
- 第 3 节：Mini case：用 `data/synthetic_air_quality.csv` 做 2×2 监测面板，并让 DSH 审查图表信息是否完整。
- 作业：制作一张“数据来源、口径、样本量、结论”四要素齐全的图，写 2 句解读。

## Week 6：合并、分组与迷你项目

- 第 1 节：`merge` / `concat`、`on` / `how`、`groupby` / `agg`。
- 第 2 节：跨表整合：`air_quality_simple.csv` + `city_info.csv`，计算人口加权指标。
- 第 3 节：Mini project：提出 2 个问题 → 清洗 → 分析 → 1 张图 → 120 字结论。
- 作业：用 DSH 把项目拆成“数据加载、清洗、分析、图表、报告”5 个子任务，分别给出验收标准。

## Week 7：第一个预测模型

- 第 1 节：特征 / 标签、训练集 / 测试集、为什么不能只报告训练得分。
- 第 2 节：Titanic 生存预测或银行营销分类；跑通最小模型并看混淆矩阵。
- 第 3 节：目标泄漏、样本不平衡、相关不等于因果；让 DSH 列出模型局限。
- 作业：写 100 字：这个模型能支持什么决策、不能支持什么决策。

## Week 8：结课项目与展示

- 第 1 节：选题、数据来源、问题与验收标准；用 plan mode 写出项目计划。
- 第 2 节：分小组并行执行：审计、EDA、可视化、报告；用 goal 追踪进度。
- 第 3 节：3 分钟展示 + 互评；完成“我验证了什么 / 我不确定什么 / 下一步做什么”。

## 推荐案例数据（已在仓库中）

- `data/成绩单.xlsx`：成绩表，适合 Week 1 Python 和 Excel 读取。
- `data/air_quality_simple.csv`：城市空气数据，适合 Week 2 基础 pandas。
- `data/air_quality_dirty.csv`：含缺失、重复、类型问题的空气数据，适合 Week 3。
- `data/city_info.csv`：城市人口 / 区域信息，适合 Week 6 合并。
- `data/synthetic_air_quality.csv`：小时级空气数据，适合 Week 5 可视化和 Week 7 回归。
- `data/bank_marketing.zip`：UCI 银行营销数据，适合 Week 4 EDA 和 Week 7 分类。
- `report/data/titanic.csv`：Titanic 生存数据，适合 Week 7 分类和结课项目。
