# scripts

课堂脚本按 `data/` 同样的主题目录组织。每个目录下“一个例子一个文件”，不要把所有 Python 程序塞进一个文件。

## 目录结构

```text
scripts/
├── download_course_data.py
├── deploy-github-pages.ps1
├── 01-agent/
│   ├── 01-python-list-loop.py
│   ├── 02-read-airbnb.py
│   └── 03-room-type-price.py
├── 02-python/
├── 03-pandas/
├── 04-viz/
├── 05-cleaning-merge/
├── 06-classification/
├── 07-regression/
└── 08-final-project/
```

每个子目录内还有自己的 `README.md`，说明该周每个文件对应哪个数据和课堂问题。

## 运行方式

所有脚本都从仓库根目录运行，例如：

```bash
python scripts/01-agent/01-python-list-loop.py
python scripts/04-viz/01-diamonds-price.py
```

## 数据准备与部署

| 脚本 | 用途 | 运行 |
|---|---|---|
| `download_course_data.py` | 下载课程数据 | `python scripts/download_course_data.py` |
| `deploy-github-pages.ps1` | 构建并部署 GitHub Pages | `pwsh -File scripts/deploy-github-pages.ps1 -Message "..."` |
