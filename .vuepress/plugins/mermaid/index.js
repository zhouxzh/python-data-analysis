import { fileURLToPath } from 'url'

export default () => {
  return {
    name: 'vuepress-plugin-mermaid',
    extendsMarkdown(md) {
      const defaultFence = md.renderer.rules.fence
      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const lang = (token.info || '').trim().split(/\s+/)[0]
        if (lang !== 'mermaid') {
          return defaultFence(tokens, idx, options, env, self)
        }
        const code = md.utils.escapeHtml(token.content.trim())
        return `<pre class="mermaid-raw"><code>${code}</code></pre>\n`
      }
    },
    clientConfigFile: fileURLToPath(new URL('./clientAppEnhance.js', import.meta.url)).replaceAll('\\', '/')
  }
}
