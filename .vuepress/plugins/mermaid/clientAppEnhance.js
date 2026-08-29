import { defineClientConfig, useRouter } from 'vuepress/client'
import mermaid from 'mermaid'

// Mermaid defaults to scanning `.mermaid` elements on load. The VuePress client
// module is loaded after the SSR content already exists, so disable auto-start
// before it can replace diagram source with an error placeholder.
mermaid.initialize({
  startOnLoad: false,
  securityLevel: 'loose',
  theme: 'base',
  themeVariables: {
    fontFamily: 'Inter, system-ui, "Segoe UI", "Microsoft YaHei", sans-serif'
  }
})

let initialized = false

function renderMermaidBlocks(root = document) {
  const blocks = root.querySelectorAll('pre.mermaid-raw')
  if (!blocks.length) return

  blocks.forEach(async (block) => {
    const codeBlock = block.querySelector('code')
    const code = (codeBlock ? codeBlock.textContent : block.textContent) || ''
    if (!code.trim() || block.getAttribute('data-rendered')) return

    block.setAttribute('data-rendered', 'true')
    block.classList.add('is-rendering')
    block.classList.remove('mermaid-raw')
    block.classList.add('mermaid')

    try {
      const { svg } = await mermaid.render(`mermaid-${Date.now()}-${Math.random().toString(16).slice(2)}`, code)
      const wrapper = document.createElement('div')
      wrapper.className = 'mermaid-wrapper'
      wrapper.innerHTML = svg
      block.replaceWith(wrapper)
    } catch (error) {
      block.classList.remove('is-rendering')
      block.classList.add('mermaid-error')
      block.setAttribute('data-error', error.message || String(error))
    }
  })
}

export default defineClientConfig({
  setup() {
    // Mermaid must run in the browser after the page content is available.
    if (typeof window === 'undefined') return

    // Temporarily hide the mermaid class so mermaid's own loader ignores them.
    document.querySelectorAll('pre.mermaid').forEach((block) => {
      block.classList.remove('mermaid')
      block.classList.add('mermaid-raw')
    })

    const renderAll = () => renderMermaidBlocks(document)

    // The client setup may run before VuePress hydrates the SSR content.
    renderAll()
    setTimeout(renderAll, 0)
    setTimeout(renderAll, 300)
    window.addEventListener('load', renderAll)

    // VuePress SPA navigation replaces page content without a new window load.
    const router = useRouter()
    router.afterEach(renderAll)

    const observer = new MutationObserver((mutations) => {
      const shouldRender = mutations.some((mutation) => {
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE && node.querySelector('pre.mermaid')) return true
        }
        return false
      })
      if (shouldRender) renderAll()
    })
    observer.observe(document.body, { childList: true, subtree: true })
  }
})
