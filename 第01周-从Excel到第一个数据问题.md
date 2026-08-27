# Week 1：环境安装与第一个数据问题

> **本章导读**
> 时长：3 节课，每节 60 分钟
> 数据：`data/成绩单.xlsx`
> 你将学到：用自然语言让 DSH 读 Excel、写第一版 Python、发现数据格式问题，并自己验证结果
> 本周产出：`projects/<姓名>/first_analysis.py`

## 1. 环境安装（第 1 节课）

第一节课先把环境装好。本课程需要三样东西：**Anaconda（Python 环境）**、**Node.js（DSH 的运行环境）**、**DeepSeek Harness（DSH）**。

### 1.1 下载并安装 Anaconda

打开 Anaconda 官方下载页：

<https://www.anaconda.com/download>

页面会显示当前版本的 Python，点击下载 64-Bit Graphical Installer：

![Anaconda 官方下载页](./images/week01/anaconda-download.png)

> 官方页面会随版本更新，按钮位置可能略有变化，认准 “Anaconda” 官方网址即可。

安装步骤：

1. 双击下载好的安装包。
2. Windows 一路点 `Next`，到许可协议时点 `I Agree`。
3. 选择 `Just Me`，安装到默认路径。
4. 到 `Advanced Installation Options` 时，勾选 `Add Anaconda3 to my PATH environment variable`。
5. 等待安装完成，不要提前关闭窗口。

安装完成后打开命令行（Windows 可以用 `cmd`、PowerShell 或 Anaconda Prompt），输入：

```bash
python --version
conda --version
```

能看到类似输出，说明 Python 环境已经装好：

```text
Python 3.12.8
conda 24.11.3
```

版本号不一样也没关系，只要两个命令都能打印版本就行。

### 1.2 打开 Documents，创建专用目录，测试 Python

安装完 Python 后，不要急着写数据分析代码，先建一个自己的练习目录，避免文件散落在桌面或下载文件夹里。

Windows 打开“文件资源管理器”，进入 Documents：

```text
C:\Users\<你的用户名>\Documents
```

macOS 打开访达，进入：

```text
~/Documents
```

在 Documents 里新建一个专用目录，例如：

```text
python-course
```

然后打开命令行，进入这个目录：

```bash
cd Documents
mkdir python-course
cd python-course
```

在目录里新建一个 `hello.py` 文件，写入：

```python
print("Hello, Python!")
print(1 + 2)
```

保存后回到命令行运行：

```bash
python hello.py
```

预期输出：

```text
Hello, Python!
3
```

能输出 `Hello, Python!`，说明 Python 安装、目录创建、文件保存和命令行运行都正常。

### 1.3 安装课程依赖

进入本仓库根目录，运行：

```bash
pip install -r requirements.txt
```

这里会安装 pandas、numpy、matplotlib、seaborn、scikit-learn、openpyxl、pytest。安装过程中出现黄色提示通常是升级提示，不一定是错误。

### 1.4 安装 Node.js

DeepSeek Harness 是一个 npm 工具，先安装 Node.js：

1. 打开 <https://nodejs.org/en/download>。
2. 下载 `LTS` 版本的 Windows/macOS 安装包。
3. 一路 `Next` 安装，保持默认选项。
4. 打开命令行，验证：

```bash
node --version
npm --version
```

能输出版本号即可，例如：

```text
v20.18.0
10.8.2
```

### 1.5 安装并启动 DeepSeek Harness

DSH 的官方仓库在：

<https://github.com/deepseek-ai/deepseek-harness>

![DeepSeek Harness 官方仓库](./images/week01/dsh-github.png)

> 截图来自官方 GitHub 仓库；安装命令以官方 README 为准。

最稳妥的启动方式是使用 `npx`，第一次会自动下载，不需要手动配置路径：

```bash
npx @deepseek-ai/dsh web
```

也可以全局安装一次，之后直接使用 `dsh`：

```bash
npm install -g @deepseek-ai/dsh
dsh web
```

看到下面这些信息，说明 DSH 已经启动：

![DSH 启动与验证示例](./images/week01/terminal-verify.png)

```text
URL: http://127.0.0.1:3080
```

然后浏览器打开：

<http://127.0.0.1:3080>

### 1.6 配置 DSH

DSH 需要两样配置才能开始工作：**模型 API Key** 和 **工作区**。

先到 DeepSeek 开放平台申请 API Key：

<https://platform.deepseek.com/>

申请后打开 DSH 的 `Settings → Models`，粘贴 API Key 并保存。然后在主界面点击 `Choose workspace`，选择本仓库根目录，例如：

```text
/home/zhong/Documents/python-data-analysis
```

Windows 上类似：

```text
D:\python-data-analysis
```

![DSH 模型与工作区配置示意图](./images/week01/webui-config.png)

保存 API Key、选中工作区后，会话输入框才可以使用。注意：**API Key 只保存在本地，不要写进教程、代码或 GitHub。**

## 2. 跟着老师做

### 2.1 先确定要回答的问题

我们拿到一份 3 人成绩单，先不急着背 pandas，先问一句：

```text
这份成绩单里，哪个科目最需要补课？
```

“需要补课”可以先定义为：**科目平均分最低，或者存在明显低于及格线的成绩**。

### 2.2 把这个任务发给 DSH

在 DSH 里输入：

```text
请读取 data/成绩单.xlsx。
任务：先输出 shape、dtypes、缺失值、完整表格；
然后计算语文、数学、英语的平均分；
最后用一句话回答“哪个科目最需要补课”。
代码保存为 projects/<姓名>/first_analysis.py。
```

### 2.3 老师的第一版代码

DSH 通常会生成类似代码：

```python
import pandas as pd

df = pd.read_excel('data/成绩单.xlsx')

print('shape:', df.shape)
print()
print(df.dtypes)
print()
print('缺失值:')
print(df.isna().sum())
print()
print(df)
```

预期输出：

```text
shape: (3, 5)
学号      int64
姓名     object
语文      int64
数学     object
英语    float64
缺失值:
 学号    0
姓名    0
语文    0
数学    0
英语    0
  学号  姓名  语文     数学    英语
0  2401  张三  59  59..5  60.5
1  2402  李四  90     95  100.0
2  2403  王五  95    100   90.0
```

### 2.4 发现第一个数据问题

`数学` 列是 `object`，因为 `张三` 的分数写成了 `59..5`。直接求平均会报错：

```text
TypeError: can only concatenate str (not "int") to str
```

这不是代码写错了，而是数据格式有问题。让 DSH 修复这一列：

```text
数学列里有一个值是 59..5，导致它被读成文本。
请把 59..5 修复成 59.5，再转成 float。
不要修改原 Excel，只修复读入后的 DataFrame。
```

### 2.5 修复并完成第一个结论

```python
df['数学'] = df['数学'].astype(str).str.replace('..', '.', regex=False)
df['数学'] = pd.to_numeric(df['数学'], errors='coerce')

print(df)
print()
print('平均分:')
print(df[['语文', '数学', '英语']].mean().round(2))
```

预期输出：

```text
  学号  姓名  语文    数学    英语
0  2401  张三  59  59.5  60.5
1  2402  李四  90  95.0 100.0
2  2403  王五  95 100.0  90.0

平均分:
语文    81.33
数学    84.83
英语    83.50
```

对“哪个科目最需要补课”这个问题，可以这样回答：

```text
按平均分看，语文 81.33 最低，最需要优先关注；
同时张三的数学、英语都低于 70，还需要看单科不合格情况。
```

## 3. 你自己动手做

1. 新建 `projects/<姓名>/first_analysis.py`，把上面的完整流程整理成可运行脚本。
2. 增加一列 `total = 语文 + 数学 + 英语`，排出总分名次。
3. 把结论改成“谁的哪一科最需要补课”。
4. 让 DSH 审查你的脚本，并给出 2 个你没发现的风险。

自己动手时建议用这个提示词：

```text
请审查 projects/<姓名>/first_analysis.py：
1. 是否按“读取 → 清洗 → 计算 → 结论”组织；
2. 数学列修复是否安全；
3. 平均分结论是否支持“哪个科目最需要补课”；
4. 列出 2 个可能的坑。
```

## 4. 验证清单

- [ ] `python --version` 和 `conda --version` 能输出版本
- [ ] 已在 Documents 里创建 `python-course` 专用目录
- [ ] `python hello.py` 能输出 `Hello, Python!` 和 `3`
- [ ] `node --version` 和 `npm --version` 能输出版本
- [ ] `npx @deepseek-ai/dsh web` 能启动，浏览器打开 `http://127.0.0.1:3080`
- [ ] 已保存 DeepSeek API Key
- [ ] 已选择本仓库作为工作区，会话输入框可输入
- [ ] 脚本能从仓库根目录用 `python scripts/week01_practice.py` 或你的项目脚本运行
- [ ] 表格能打印出来
- [ ] `59..5` 被修复成 `59.5`
- [ ] 平均分、总分、排名都有输出
- [ ] 每个结论都能指出“用了哪一列、怎么算的”

## 5. 常见错误与坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `python` 不是内部或外部命令 | Anaconda 没加入 PATH | 重装时勾选 `Add Anaconda3 to my PATH`，或使用 Anaconda Prompt |
| `python hello.py` 提示找不到文件 | 命令行不在文件所在目录 | 先 `cd` 到 `python-course`，再运行 `python hello.py` |
| `npx` 不是内部或外部命令 | Node.js 没安装或没生效 | 重装 Node.js LTS，重新打开命令行 |
| DSH 页面打不开 | 启动命令还在下载，或端口被占用 | 等命令显示 `URL: http://127.0.0.1:3080` 后再刷新浏览器 |
| 页面能开但不能输入对话 | 没保存 API Key 或没选工作区 | 完成 `Settings → Models` 和 `Choose workspace` |
| 数学列是 object | 数据里有 `59..5` | 先观察，再清洗，不直接强转 |
| 平均分报字符串拼接错误 | 数值列被读成文本 | `to_numeric` 前先看异常值 |
| AI 删掉原始数据 | 提示词没说明 | 明确写“不要修改原文件” |
| 第一版结论太宽泛 | 问题没定义 | 先定义“需要补课”的口径 |

## 6. 作业

1. 按第 1 节完成 Anaconda、Node.js 和 DSH 安装，把 `hello.py` 运行结果和验证命令截图保存到 `projects/<姓名>/environment.png`。
2. 让 DSH 生成代码，统计这份成绩单，然后写出 3 个“第一版提示词没有回答”的问题。

示例：

- 哪门课最需要补课？
- 单科不及格的人有几科不及格？
- 总分前两名是否说明所有科目都更强？

## 评分要点

| 项目 | 要求 |
|---|---|
| 环境 | `python`、`node`、DSH 都能启动，工作区已选择 |
| 运行 | 每个代码块都能运行 |
| 清洗 | 能发现 `59..5` 并修复 |
| 结果 | 平均分、总分、排名都有输出 |
| AI 协作 | 保留 DSH 修改代码的记录 |
