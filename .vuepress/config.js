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
    'sessions/**/*.md',
    'scripts/**/*.md',
    'prompts/**/*.md',
    'dsh/**/*.md',
    'examples/**/*.md',
    'assignments/**/*.md',
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
      { text: '每周实践', link: '/sessions/week01-vibe-and-first-data.html' },
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
          '/sessions/week01-vibe-and-first-data.html',
          '/sessions/week02-pandas-and-csv.html',
          '/sessions/week03-cleaning-and-audit.html',
          '/sessions/week04-eda-and-hypotheses.html',
          '/sessions/week05-visualization.html',
          '/sessions/week06-merge-and-group.html',
          '/sessions/week07-first-model.html',
          '/sessions/week08-final-project.html'
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
