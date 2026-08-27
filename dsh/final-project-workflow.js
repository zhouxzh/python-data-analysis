// DSH workflow 示例：结课项目三阶段流水线。
// 使用时把本文件作为 workflow 工具的 script body 传入，
// 并在 workflow 调用参数中提供 datasetPath / projectDir / reportTitle。
// meta 参数示例：
// {
//   "name": "final-project-analysis",
//   "description": "结课项目：审计、EDA、报告",
//   "phases": [
//     { "title": "数据审计", "detail": "检查数据质量" },
//     { "title": "EDA 与可视化", "detail": "回答核心问题并出图" },
//     { "title": "报告撰写", "detail": "形成可验证报告" }
//   ]
// }

const datasetPath = args.datasetPath || "data/air_quality_dirty.csv";
const projectDir = args.projectDir || "projects/example/final";
const reportTitle = args.reportTitle || "城市空气质量分析";

phase("数据审计");
log(`开始审计 ${datasetPath}`);

const audit = await agent(
  `你是数据分析审计员。请读取 ${datasetPath}，不要修改原文件。
   输出 JSON：shape、problems（按严重程度排序的问题）、suggested_cleaning（建议处理）。
   每个问题都要给出判断依据。`,
  {
    label: "audit",
    schema: {
      type: "object",
      properties: {
        shape: { type: "string" },
        problems: { type: "array", items: { type: "string" } },
        suggested_cleaning: { type: "array", items: { type: "string" } },
      },
      required: ["shape", "problems", "suggested_cleaning"],
      additionalProperties: false,
    },
  }
);

phase("EDA 与可视化");
log("开始 EDA 与可视化");

const eda = await agent(
  `你是数据分析 EDA 工程师。请读取 ${datasetPath}。
   回答 2 个明确问题，并把结果保存到 ${projectDir}/output/eda_summary.csv。
   同时生成 ${projectDir}/output/dashboard.png，每张图必须对应一个问题。
   输出 JSON：questions、findings、chart_path。`,
  {
    label: "eda",
    schema: {
      type: "object",
      properties: {
        questions: { type: "array", items: { type: "string" } },
        findings: { type: "array", items: { type: "string" } },
        chart_path: { type: "string" },
      },
      required: ["questions", "findings", "chart_path"],
      additionalProperties: false,
    },
  }
);

phase("报告撰写");
log("开始报告撰写");

const report = await agent(
  `你是数据分析报告作者。请基于以下审计和 EDA 结果撰写 ${projectDir}/report.md。
   标题：${reportTitle}。
   结构：问题、数据来源、方法、发现、局限、下一步。
   要求：区分已验证事实与待验证推断，每个发现注明依据和样本量。
   审计结果：${JSON.stringify(audit)}
   EDA 结果：${JSON.stringify(eda)}`,
  {
    label: "report",
  }
);

return {
  audit,
  eda,
  report_path: `${projectDir}/report.md`,
  report,
};
