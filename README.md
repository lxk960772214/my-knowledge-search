# 🧠 Smart Knowledge MCP Server

这是一个专为个人知识库设计的智能 MCP (Model Context Protocol) 服务器。它结合了**极速本地模糊搜索**与**云端语义 RAG** 技术，帮助 AI 助手高效地“回忆”和利用你本地的 Markdown 笔记、技术文档及客户资料。

## ✨ 核心特性

### 1. ⚡️ 极速结构化搜索 (Smart Search)
不同于普通的全文搜索，本工具针对知识库结构进行了深度优化：
- **目录感知 (Context Aware)**: 智能识别 `/A客户名/readme*.md` 结构。搜索“A客户名”时，会自动高优匹配其目录下的 `readme.md`、`README_v2.md` 等文件。
- **目录递归遍历**: 支持递归扫描指定根目录下的所有子目录和文件。
- **模糊匹配增强**: 
  - 目录名模糊匹配：搜  `python实用` 可命中 `python实用命令` 目录。
  - 文件名模糊匹配：搜 `python` 可命中 `python_notes.md`。
- **YAML 标签增强**: 支持解析 Markdown 文件头的 YAML Front Matter。
- **性能优先**: 默认采用“IO 优化模式”，仅读取文件头 500 字符进行索引。

### 2. 🤖 语义 RAG 阅读 (Semantic Reading)
当找到相关文档后，不再需要人工阅读全文：
- **云端大脑**: 集成 DeepInfra/OpenRouter 的云端 Embedding 模型 (BGE-M3)，无需本地显卡，轻量高效。
- **智能分块**: 使用 LangChain 按 Markdown 标题层级（H1/H2/H3）精准切分文档，保留上下文。
- **按需提取**: 根据你的自然语言问题（如“查看登录密码”或“部署流程”），自动计算语义相似度，仅提取最相关的段落给 AI。

## 🛠️ 安装指南

### 前置要求
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
        "/Users/你的用户名/Documents/USER/"
      ],
      "registryUrl": "https://pypi.tuna.tsinghua.edu.cn/simple"
    }
  }
}
```
> **注意**: 
> 1. `command` 请替换为你本地 uv 的绝对路径 (可通过 `which uv` 查看)。
> 2. custom_mcp_server.py 后续参数是你希望被索引的**文件夹路径**（支持多个）。

## 📖 最佳实践

### 1. 客户目录结构
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

### 2. 技术文档打标
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

# 💁‍♀️推荐提示词

请按照以下步骤回答用户问题：

1. **定位文档 (Broad Search)**:
   - 先分析用户问题属于哪个技术领域或客户项目。
   - 使用最核心的关键词（如 "python", "linux", "A客户", "B客户"）调用 `search_files` 工具。
   - *不要* 一开始就使用长难句搜索，因为文件名通常只包含核心词。
2. **精准提取 (Deep Read)**:
   - 根据第 1 步返回的文件列表，选择最可能包含答案的文件。
   - 使用具体的长尾关键词（如 "pytorch 安装验证", "login password"）调用 `read_relevant_content` 工具。
   - 这一步利用 RAG 技术，可以在文档内部进行精准的语义搜索。
3. **综合回答**:
   - 根据提取到的内容回答用户问题。

比如“给我linux文档中docker+nginx部分”。 客户已经指明是“linux文档”，此任务优先检索带有“linux”关键词的文档，不需要查找文件名中带有docker nginx关键词的文档。 再在上一步文档中查找和“docker+nginx”相关的部分给用户。
