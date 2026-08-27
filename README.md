# Vibe Programming × Python 数据分析（8 周 / 24 学时）

## 周贤中 zhouxzh@gdut.edu.cn

面向低基础学生的 Python 数据分析课程。新版课程以 **DeepSeek Harness 协作式分析**为核心：先定义问题，再让 DSH 生成第一版代码，学生负责运行、验证、追问、解释和决策。

- 时长：8 周，每周一个晚上，每晚上 3 节课，共 24 学时。
- 案例：成绩单、城市空气质量、银行营销、Titanic、学生自选项目。
- 交付：每周 3 节课的完整教案、提示词库、DSH 手册、参考代码、结课项目模板。

## 课程入口

新版课程全部放在 [`vibe-course/`](vibe-course/README.md)：

| 内容 | 路径 |
|---|---|
| 课程总览 | [vibe-course/README.md](vibe-course/README.md) |
| 8 周课表 | [vibe-course/syllabus.md](vibe-course/syllabus.md) |
| 每周教案 | [vibe-course/sessions/](vibe-course/sessions/) |
| DSH 使用手册 | [vibe-course/dsh/harness-playbook.md](vibe-course/dsh/harness-playbook.md) |
| 核心提示词库 | [vibe-course/prompts/core-prompts.md](vibe-course/prompts/core-prompts.md) |
| 参考代码 | [vibe-course/examples/](vibe-course/examples/) |
| 结课项目 | [vibe-course/assignments/final-project.md](vibe-course/assignments/final-project.md) |

## 8 周课程地图

1. 从 Excel 到第一个数据问题
2. pandas 数据结构与读取
3. 数据清洗与审计
4. EDA 与提出假设
5. 可视化表达
6. 合并、分组与迷你项目
7. 第一个预测模型
8. 结课项目与展示

## 环境

```bash
pip install -r requirements.txt
```

然后打开 DeepSeek Harness Web GUI，把本仓库作为 workspace。

## 旧版材料

旧版 notebook 仍保留在 [`notebooks/`](notebooks/)，Titanic 完整示例仍保留在 [`report/`](report/)。新课程会复用其中的数据与案例，但教学主线已改为 Vibe Programming。

## 学习迁移提示

1. 把数据分析看成循环：问题 → 数据 → 清洗 → 探索 → 表达 → 复盘。
2. AI 生成第一版，人负责验证和决策。
3. 报错和追问记录是学习资产，不要只保留最终答案。
4. 结论必须能指出依据：哪个字段、什么计算、多少样本。
5. 可复现 > 炫技，注释解释“为什么”而不是“做什么”。
