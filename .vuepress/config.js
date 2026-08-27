import { viteBundler } from '@vuepress/bundler-vite'
import { defaultTheme } from '@vuepress/theme-default'

export default {
  lang: 'zh-CN',
  title: 'Python 数据分析教程',
  description: '8 周 Vibe Programming × Python 数据分析',
  base: '/python-data-analysis/',
  bundler: viteBundler(),
  pagePatterns: [
    'README.md',
    'vibe-course/**/*.md',
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
      { text: '课程总览', link: '/vibe-course/' },
      { text: '8 周课表', link: '/vibe-course/syllabus.html' },
      { text: 'DSH 手册', link: '/vibe-course/dsh/harness-playbook.html' }
    ],
    sidebar: {
      '/': [
        { text: '首页', link: '/' },
        { text: '课程总览', link: '/vibe-course/' },
        { text: '8 周课表', link: '/vibe-course/syllabus.html' },
        { text: '核心提示词', link: '/vibe-course/prompts/core-prompts.html' },
        { text: 'DSH 手册', link: '/vibe-course/dsh/harness-playbook.html' }
      ],
      '/vibe-course/': [
        {
          text: '课程入口',
          children: [
            '/vibe-course/',
            '/vibe-course/syllabus.html'
          ]
        },
        {
          text: '每周教案',
          children: [
            '/vibe-course/sessions/week01-vibe-and-first-data.html',
            '/vibe-course/sessions/week02-pandas-and-csv.html',
            '/vibe-course/sessions/week03-cleaning-and-audit.html',
            '/vibe-course/sessions/week04-eda-and-hypotheses.html',
            '/vibe-course/sessions/week05-visualization.html',
            '/vibe-course/sessions/week06-merge-and-group.html',
            '/vibe-course/sessions/week07-first-model.html',
            '/vibe-course/sessions/week08-final-project.html'
          ]
        },
        {
          text: '工具与资料',
          children: [
            '/vibe-course/prompts/core-prompts.html',
            '/vibe-course/dsh/harness-playbook.html'
          ]
        },
        {
          text: '结课项目',
          children: [
            '/vibe-course/assignments/final-project.html',
            '/vibe-course/assignments/rubric.html'
          ]
        }
      ]
    }
  })
}
