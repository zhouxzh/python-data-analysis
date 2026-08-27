import { viteBundler } from '@vuepress/bundler-vite'
import { defaultTheme } from '@vuepress/theme-default'

export default {
  lang: 'zh-CN',
  title: 'Vibe Programming × Python 数据分析',
  description: '8 周 Vibe Programming × Python 数据分析课程',
  base: '/python-data-analysis/',
  bundler: viteBundler(),
  pagePatterns: [
    'README.md',
    'syllabus.md',
    '第*.md',
    'scripts/**/*.md',
    'prompts/**/*.md',
    'dsh/**/*.md',
    'examples/**/*.md',
    'assignments/**/*.md',
    'projects/**/*.md',
    '!**/node_modules/**',
    '!**/.git/**',
    '!**/.vuepress/**'
  ],
  theme: defaultTheme({
    repo: 'https://github.com/zhouxzh/python-data-analysis',
    docsDir: '.',
    sidebarDepth: 2,
    navbar: [
      { text: '首页', link: '/' },
      { text: '8 周课表', link: '/syllabus.html' },
      { text: '每周实践', link: '/第01周-从Excel到第一个数据问题.html' },
      { text: 'DSH 手册', link: '/dsh/harness-playbook.html' }
    ],
    sidebar: [
      { text: '首页', link: '/' },
      { text: '8 周课表', link: '/syllabus.html' },
      { text: '核心提示词', link: '/prompts/core-prompts.html' },
      { text: 'DSH 手册', link: '/dsh/harness-playbook.html' },
      {
        text: '每周实践教程',
        children: [
          '/第01周-从Excel到第一个数据问题.html',
          '/第02周-pandas与数据读取.html',
          '/第03周-数据清洗与审计.html',
          '/第04周-EDA与提出假设.html',
          '/第05周-可视化表达.html',
          '/第06周-合并分组与迷你项目.html',
          '/第07周-第一个预测模型.html',
          '/第08周-结课项目与展示.html'
        ]
      },
      {
        text: '结课项目',
        children: [
          '/assignments/final-project.html',
          '/assignments/rubric.html'
        ]
      }
    ]
  })
}
