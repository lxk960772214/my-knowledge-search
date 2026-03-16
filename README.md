# 🧠 Smart Knowledge MCP Server

这是一个专为个人知识库设计的智能 MCP (Model Context Protocol) 服务器。它结合了**极速本地模糊搜索**与**云端语义 RAG** 技术，帮助 AI 助手高效地“回忆”和利用你本地的 Markdown 笔记、技术文档及客户资料。

# ✨ 核心特性

## 1. ⚡️ 极速结构化搜索 (Smart Search)
不同于普通的全文搜索，本工具针对知识库结构进行了深度优化：
- **目录感知 (Context Aware)**: 智能识别 `/A客户名/readme*.md` 结构。搜索“A客户名”时，会自动高优匹配其目录下的 `readme.md`、`README_v2.md` 等文件。
- **目录递归遍历**: 支持递归扫描指定根目录下的所有子目录和文件。
- **模糊匹配增强**: 
  - 目录名模糊匹配：搜  `python实用` 可命中 `python实用命令` 目录。
  - 文件名模糊匹配：搜 `python` 可命中 `python_notes.md`。
- **YAML 标签增强**: 支持解析 Markdown 文件头的 YAML Front Matter。
- **性能优先**: 默认采用“IO 优化模式”，仅读取文件头 500 字符进行索引。

## 2. 🤖 语义 RAG 阅读 (Semantic Reading)
当找到相关文档后，不再需要人工阅读全文：
- **云端大脑**: 集成 DeepInfra/OpenRouter 的云端 Embedding 模型 (BGE-M3)，无需本地显卡，轻量高效。
- **智能分块**: 使用 LangChain 按 Markdown 标题层级（H1/H2/H3）精准切分文档，保留上下文。
- **按需提取**: 根据你的自然语言问题（如“查看登录密码”或“部署流程”），自动计算语义相似度，仅提取最相关的段落给 AI。

# 🛠️ 安装指南

## 前置要求
- Python 3.10+
- 推荐使用 `uv` 包管理器（也可使用 `pip`）

## ⚙️ Cherry Studio 配置

在 Cherry Studio 的 **设置 > MCP 服务器** 中添加以下配置：

```json
{
  "mcpServers": {
    "my-knowledge-search": {
      "command": "uv", 
      "args": [
        "--directory",
        "custom_mcp_server.py文件所在目录绝对路径",
        "run",
        "custom_mcp_server.py",
        "/Users/你的用户名/Documents/Notes/",
        "/Users/你的用户名/Documents/USER/",
        "--exclude-dirs",
        "test,logs,tmp,/Users/你的用户名/Documents/Private",
        "--exclude-files",
        "secret.md,draft.md,/Users/你的用户名/Documents/Notes/unfinished.md"
      ],
      "registryUrl": "https://pypi.tuna.tsinghua.edu.cn/simple"
    }
  }
}
```
> **注意**: 
> 1. `command` 请替换为你本地 uv 的绝对路径 (可通过 `which uv` 查看)。
> 2. `custom_mcp_server.py` 后续参数是你希望被索引的**文件夹路径**（支持多个）。
> 3. `--exclude-dirs` 和 `--exclude-files` 支持传入目录/文件名（如 `test`）或完整绝对路径（如 `/Users/.../test`），多个路径用逗号分隔。

# 📖 最佳实践

## 1. 客户目录结构
推荐将客户信息统一管理，搜索效率最高：
```text
/VIP/
  ├── A客户/
  │    └── readme.md   <-- 包含项目概况、服务器信息
  ├── B客户/
  │    └── readme.md
  └── C客户/
       └── readme.md
```

## 2. 技术文档打标
在笔记开头添加 YAML 信息，方便检索：
```markdown
---
tags: [linux,运维,常用命令]
description: 记录生产环境常用的 Linux 排查命令
---
# Linux 运维手册
...
```
此时搜索“运维”或“常用命令”均可命中该文件。

# 💁‍♀️ 推荐系统提示词 (System Prompt)

建议将以下内容复制到 AI 助手的 System Prompt 中，以获得最佳使用体验：

```text
请严格按照以下【速度优先】策略执行：

⚡️ 快速定位 (Quick Search):

提取用户问题中的 1-2 个最核心名词（如 "Linux", "A客户"）作为关键词。
调用 search_files。
🚫 禁止使用长句或复杂描述进行搜索。

🎯 单点突破 (One-Shot Read):

从搜索结果中，直接选择 得分最高 (Top 1) 的那个文件。
仅当 Top 1 明显不相关时，才考虑 Top 2。
使用具体的问题细节（如 "docker nginx 配置", "cuda 安装命令"）调用 read_relevant_content。

🚀 即时响应 (Instant Reply):

获取 RAG 返回的片段后，立即总结回答。
即使相似度分数不高，只要内容相关，就直接使用，不要进行二次检索。
🚫 除非完全未找到相关信息，否则严禁尝试打开第二个文件。

✅ 案例示范:

用户: "给我linux文档中docker+nginx部分"
-> 动作 1: search_files("Linux") (只搜核心词)
-> 动作 2: 在 Top 1 文件检索"docker nginx 配置"

用户: "查一下安装包目录里的 CUDA 安装方法"
-> 动作 1: search_files("安装包")
-> 动作 2: 找到 安装包/readme.md，检索"CUDA 安装命令"

```




# 🧾下一步计划
- 新增忽略文件，忽略目录。已完成
- embedding 模型通过环境变量配置。已完成
- 优化得分逻辑，优先最高得分文档匹配，提高检索效率。已完成

