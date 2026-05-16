# Hermes Agent 代码架构分析

> 版本: v0.10.0 | 作者: Nous Research | "The Self-Improving AI Agent"

---

## 1. 项目概述

Hermes Agent 是一个基于 Python 的 LLM Agent 框架，提供终端交互界面、多平台消息网关、技能（Skills）系统、记忆（Memory）系统以及研究工具链（批处理轨迹生成、RL 训练环境）。

**技术栈：** Python 3.11+、`uv` 包管理器、pytest (xdist)、OpenAI/Anthropic SDK、prompt_toolkit + Rich (CLI)、React Ink (TUI)、Docusaurus (文档)

---

## 2. 顶层目录结构

```
hermes-agent/
│
├── run_agent.py                 # AIAgent 核心对话循环
├── cli.py                       # HermesCLI 交互式终端
├── model_tools.py               # 工具编排层 — 发现+调度
├── toolsets.py                  # 工具集定义 (Toolset 系统)
│
├── hermes_constants.py          # 路径常量和环境检测
├── hermes_state.py              # SessionDB — SQLite+FTS5 会话存储
├── hermes_logging.py            # 集中式日志（滚动文件+脱敏格式化）
├── hermes_time.py               # 时间工具
├── utils.py                     # 共享工具函数
│
├── batch_runner.py              # 并行批处理（研究/数据生成）
├── trajectory_compressor.py     # 轨迹压缩（RL token 预算）
├── toolset_distributions.py     # 工具集概率分布（批处理用）
├── rl_cli.py                    # RL 训练 CLI 运行器
│
├── agent/                       # Agent 内部逻辑
├── hermes_cli/                  # CLI 子命令系统
├── tools/                       # 工具实现（每个文件一个工具族）
├── gateway/                     # 消息网关
├── tui_gateway/                 # React TUI 后端 (JSON-RPC)
├── cron/                        # 内置调度器
├── environments/                # RL 训练环境
├── plugins/                     # 插件系统
├── acp_adapter/                 # ACP 服务器（编辑器集成）
├── skills/                      # 内置技能
├── tests/                       # 测试套件
├── docker/                      # Docker 配置
├── .github/                     # CI/CD
├── acp_registry/                # ACP 注册元数据
└── assets/                      # 资源
```

---

## 3. 入口点体系

### 3.1 三个 CLI 入口点

| 命令 | 模块 | 用途 |
|------|------|------|
| `hermes` | `hermes_cli/main.py:main` | **主入口** — 所有子命令 |
| `hermes-agent` | `run_agent.py:main` | 独立 Agent 运行器 |
| `hermes-acp` | `acp_adapter/entry.py:main` | ACP 服务器（编辑器集成） |

### 3.2 `hermes` 命令体系 (`hermes_cli/main.py`)

主入口使用 `argparse` 分发子命令：

```
hermes                           # 默认启动交互式聊天
hermes chat                      # 交互式聊天（显式）
hermes gateway [start|stop|...]  # 消息网关管理
hermes setup                     # 安装向导
hermes model                     # 切换模型/提供商
hermes tools                     # 工具配置
hermes doctor                    # 诊断
hermes cron                      # 定时任务管理
hermes update                    # 更新
hermes uninstall                 # 卸载
hermes sessions browse           # 会话浏览
hermes acp                       # ACP 服务器模式
...
```

关键特性：`_apply_profile_override()` 在导入任何模块**之前**拦截 `--profile` 参数，设置 `HERMES_HOME` 环境变量以实现多 Profile 隔离。

### 3.3 文件依赖链

```
tools/registry.py                # 无依赖 — 被所有工具文件导入
       ↑
tools/*.py                       # 每个在模块级调用 registry.register()
       ↑
model_tools.py                   # 导入 registry + 触发工具发现
       ↑
run_agent.py / cli.py / batch_runner.py / environments/    # 消费者
```

---

## 4. 核心组件详解

### 4.1 AIAgent (`run_agent.py`) — 核心对话循环

这是整个系统的核心，负责：

1. **消息循环**：同步循环，调用 LLM API，处理工具调用，管理消息历史
2. **上下文压缩**：通过 `ContextCompressor` 自动压缩超长对话
3. **Prompt 缓存**：支持 Anthropic 的 prompt caching
4. **迭代预算**：`IterationBudget` 线程安全计数器（父级默认 90 次，子代理默认 50 次）
5. **错误处理**：通过 `error_classifier.py` 分类 API 错误，支持故障转移
6. **轨迹保存**：通过 `trajectory.py` 保存对话轨迹供 RL 训练使用

核心数据结构：

```python
class AIAgent:
    def run_conversation(self, message, ...) -> Dict[str, Any]:
        # 1. 构建消息历史
        # 2. 调用 LLM API (同步循环)
        # 3. 处理工具调用 → 执行 → 返回结果
        # 4. 重复直到完成或达到迭代上限
```

### 4.2 HermesCLI (`cli.py`) — 交互式终端

基于 `prompt_toolkit` 的交互式 REPL：

- **固定输入区**：底部输入栏，上方滚动输出区
- **历史记录**：文件持久化的命令历史
- **自动补全**：命令补全和路径补全
- **Rich 格式化**：彩色输出、进度条
- **皮肤引擎**：通过 `skin_engine.py` 自定义视觉风格
- **KawaiiSpinner**：`agent/display.py` 中的酷炫加载动画

### 3.3 两种入口路径

```
hermes CLI 启动                    | 直接 Agent 调用
                                   |
hermes_cli/main.py                 | run_agent.py
       |                           |      |
hermes_cli/commands.py             | AIAgent.run_conversation()
       |                           |      |
cli.py (HermesCLI)                 | model_tools 层
       |                           |      |
model_tools 层                     | tools/registry
       |
tools/registry
```

---

## 5. 层次架构与数据流

### 5.1 六层架构

```
┌─────────────────────────────────────────────────────────┐
│  第1层: 入口层 (Entry Points)                             │
│  hermes_cli/main.py  run_agent.py  cli.py  gateway/run.py │
│  batch_runner.py  rl_cli.py  acp_adapter/entry.py        │
│  tui_gateway/entry.py                                    │
├─────────────────────────────────────────────────────────┤
│  第2层: Agent 核心 (Agent Core)                           │
│  run_agent.py  —  AIAgent 对话循环                       │
│  agent/prompt_builder.py  —  系统提示词组装               │
│  agent/context_engine.py  —  上下文窗口管理               │
│  agent/context_compressor.py  —  自动压缩                 │
│  agent/memory_manager.py  —  记忆上下文块构建             │
│  agent/retry_utils.py  —  重试逻辑                        │
│  agent/error_classifier.py  —  错误分类/故障转移           │
├─────────────────────────────────────────────────────────┤
│  第3层: 工具编排 (Tool Orchestration)                     │
│  model_tools.py  —  get_tool_definitions/                │
│                     handle_function_call                 │
│  tools/registry.py  —  注册/发现/调度                     │
│  toolsets.py  —  工具集定义/解析                          │
├─────────────────────────────────────────────────────────┤
│  第4层: 传输层 (Transport Layer)                          │
│  agent/transports/                                       │
│  ├── base.py  —  抽象基类                                │
│  ├── anthropic.py  —  Anthropic Messages API             │
│  ├── chat_completions.py  —  OpenAI 兼容 API             │
│  ├── bedrock.py  —  AWS Bedrock                         │
│  └── codex.py  —  GitHub Copilot Codex                  │
├─────────────────────────────────────────────────────────┤
│  第5层: 工具实现 (Tool Implementations)                   │
│  tools/*.py  —  50+ 个工具文件                           │
│  tools/environments/  —  6 种执行后端                    │
│  tools/browser_providers/  —  3 种浏览器提供者            │
├─────────────────────────────────────────────────────────┤
│  第6层: 外部集成 (External Integrations)                  │
│  gateway/platforms/  —  20+ 消息平台适配器                │
│  plugins/  —  插件系统（记忆后端、图像生成等）             │
│  cron/  —  定时任务                                      │
│  acp_adapter/  —  编辑器集成                             │
│  tui_gateway/  —  React TUI 后端                         │
└─────────────────────────────────────────────────────────┘
```

### 5.2 请求数据流

```
用户输入
    │
    ▼
┌──────────────────────────────────────┐
│  入口层                               │
│  (CLI / Gateway / ACP / TUI)          │
│  解析输入格式，调用 AIAgent            │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  AIAgent.run_conversation()           │
│  1. 构建消息历史 (含系统提示词)        │
│  2. 调用 LLM API (通过 Transport)      │
│  3. 解析响应中的 tool_calls            │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  model_tools.handle_function_call()   │
│  1. 在 registry 中查找工具            │
│  2. 检查权限/可用性                   │
│  3. 执行工具处理函数                   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  tools/*.py 工具实现                   │
│  返回 JSON 字符串结果                  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  AIAgent 将结果加入消息历史            │
│  继续循环 -> 直到完成或达到迭代上限     │
└──────────────────────────────────────┘
```

---

## 6. 工具系统 (`tools/`)

### 6.1 注册模式

Hermes Agent 使用自注册（self-registration）模式。每个工具模块在**模块级**调用 `registry.register()`：

```python
# tools/web_tools.py
from tools.registry import registry

registry.register(
    name="web_search",
    toolset="web",
    schema={...},        # JSON Schema
    handler=handle_web_search,  # 处理函数
    check_fn=check_web_search,  # 可用性检查
)
```

### 6.2 `tools/registry.py` — 中央注册表

`ToolEntry` 数据结构：

```python
class ToolEntry:
    name: str           # 工具名称
    toolset: str        # 所属工具集
    schema: dict        # JSON Schema
    handler: callable   # 处理函数
    check_fn: callable  # 可用性检查
    cache_strategy: str # 缓存策略
    # ...
```

核心方法：
- `register()` — 注册工具
- `discover_builtin_tools()` — 通过 AST 分析自动发现 `tools/*.py` 中的注册调用
- `get_tool_definitions(...)` — 获取已启用的工具定义
- `dispatch()` — 调度工具调用
- `get_tool_names_for_toolset()` — 查询某工具集的所有工具

### 6.3 `model_tools.py` — 编排层

提供以下公共 API：

```python
get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)
handle_function_call(function_name, function_args, task_id, user_task)
get_toolset_for_tool(name)
check_toolset_requirements()
check_tool_availability(quiet)
```

### 6.4 `toolsets.py` — 工具集系统

工具集（Toolset）是工具的命名分组，支持**组合**：

```python
TOOLSETS = {
    "web": {"tools": ["web_search", "web_extract"], "includes": []},
    "terminal": {"tools": ["terminal", "process"], "includes": []},
    "debugging": {"tools": [], "includes": ["web", "file"]},  # 组合
    "hermes-cli": {"tools": _HERMES_CORE_TOOLS, "includes": []},
    "hermes-gateway": {  # 所有消息平台的联合
        "tools": [],
        "includes": ["hermes-telegram", "hermes-discord", ...]
    },
}
```

`_HERMES_CORE_TOOLS` 包含约 20 个核心工具（web、文件、终端、浏览器、技能、视觉、TTS、规划等），是所有平台工具集的基础。

### 6.5 工具分类

| 类别 | 工具数量 | 代表工具 |
|------|---------|---------|
| Web | 2 | `web_search`, `web_extract` |
| 终端 | 2 | `terminal`, `process` |
| 文件 | 4 | `read_file`, `write_file`, `patch`, `search_files` |
| 浏览器 | 10+ | `browser_navigate`, `browser_click`, `browser_snapshot` 等 |
| 代码执行 | 1 | `execute_code` |
| 委托 | 1 | `delegate_task` |
| 技能 | 3 | `skills_list`, `skill_view`, `skill_manage` |
| 视觉 | 1 | `vision_analyze` |
| 图像生成 | 1 | `image_generate` |
| 记忆 | 1 | `memory` |
| 规划 | 1 | `todo` |
| 消息发送 | 1 | `send_message` |
| Cron | 1 | `cronjob` |
| TTS | 1 | `text_to_speech` |
| 会话搜索 | 1 | `session_search` |
| 澄清 | 1 | `clarify` |

### 6.6 执行后端 (`tools/environments/`)

终端工具支持 6 种执行后端：

```
tools/environments/
├── base.py          # 抽象基类
├── local.py         # 本地终端
├── docker.py        # Docker 容器
├── ssh.py           # SSH 远程
├── modal.py         # Modal 无服务器
├── daytona.py       # Daytona 工作区
└── singularity.py   # Singularity 容器
```

### 6.7 浏览器提供者 (`tools/browser_providers/`)

```
tools/browser_providers/
├── base.py          # 抽象基类
├── browser_use.py   # Browser Use 提供者
├── browserbase.py   # Browserbase 云浏览器
└── firecrawl.py     # Firecrawl 浏览器
```

---

## 7. 传输层 — LLM 适配器 (`agent/transports/`)

负责抽象不同 LLM API 的差异：

```
agent/transports/
├── base.py              # 抽象基类 Transport
├── anthropic.py         # Anthropic Messages API
├── chat_completions.py  # OpenAI 兼容 API
├── bedrock.py           # AWS Bedrock
├── codex.py             # GitHub Copilot Codex
└── types.py             # 类型定义
```

此外还有各提供商的特定适配器：

```
agent/anthropic_adapter.py     # Anthropic 适配
agent/bedrock_adapter.py       # AWS Bedrock 适配
agent/gemini_native_adapter.py # Gemini 原生适配
agent/gemini_cloudcode_adapter.py
agent/codex_responses_adapter.py
agent/copilot_acp_client.py
agent/google_code_assist.py
```

---

## 8. 消息网关 (`gateway/`)

### 8.1 架构

```
gateway/
├── run.py              # 主循环、消息分发、斜杠命令
├── session.py          # SessionStore — 会话持久化
├── session_context.py  # 会话上下文管理
├── config.py           # 网关配置
├── delivery.py         # 消息投递路由
├── hooks.py            # 钩子系统
├── channel_directory.py # 频道→会话映射
├── pairing.py          # DM 配对（安全）
├── mirror.py           # 会话镜像
├── stream_consumer.py  # 流消费
├── platforms/          # 20+ 平台适配器
└── builtin_hooks/      # 内置钩子实现
```

### 8.2 支持的消息平台 (20+)

| 平台 | 适配器 | 类型 |
|------|--------|------|
| Telegram | `telegram.py` + `telegram_network.py` | 即时消息 |
| Discord | `discord.py` | 即时消息 |
| Slack | `slack.py` | 团队协作 |
| WhatsApp | `whatsapp.py` | 即时消息 |
| Signal | `signal.py` | 加密消息 |
| Email | `email.py` | 邮件 |
| Matrix | `matrix.py` | 去中心化 |
| Mattermost | `mattermost.py` | 自托管团队消息 |
| DingTalk | `dingtalk.py` | 企业消息 |
| Feishu/Lark | `feishu.py` + `feishu_comment*` | 企业协作 |
| WeCom (企业微信) | `wecom.py` + `wecom_callback*` | 企业消息 |
| Weixin (微信) | `weixin.py` | 个人微信 |
| QQ | `qqbot/` | QQ 消息 |
| SMS | `sms.py` | 短信 |
| BlueBubbles (iMessage) | `bluebubbles.py` | Apple 消息 |
| Home Assistant | `homeassistant.py` | 智能家居 |
| Webhook | `webhook.py` | 通用 Webhook |
| REST API | `api_server.py` | HTTP API |

### 8.3 Gateway 工作流

```
外部消息 → Platform Adapter → Gateway 主循环 → AIAgent → 回复 → Platform Adapter
                                │
                          ┌─────┴─────┐
                          │ 斜杠命令处理 │
                          └───────────┘
```

---

## 9. 记忆系统

### 9.1 核心组件

```
agent/memory_manager.py   # 构建记忆上下文块、清理上下文
agent/memory_provider.py  # 记忆提供者抽象
agent/hermes_state.py     # SessionDB — SQLite + FTS5 全文搜索
```

### 9.2 记忆架构

- **SessionDB** (`hermes_state.py`): 基于 SQLite 的会话存储，支持 FTS5 全文搜索
- **Memory Manager** (`memory_manager.py`): 构建记忆上下文块，在执行开始时注入到系统提示
- **Memory Provider** (`memory_provider.py`): 记忆提供者的抽象接口
- **记忆后端插件** (`plugins/memory/`):
  - Honcho
  - Mem0
  - Byterover
  - Holographic
  - OpenViking
  - RetainDB
  - SuperMemory
  - Hindsight

### 9.3 记忆数据流

```
1. 用户发送消息
2. AIAgent 开始处理
3. memory_manager 从 SQLite 加载相关记忆
4. 构建记忆上下文块 → 注入系统提示
5. 调用 LLM
6. 对话结束后，AI 可能调用 memory 工具写入新记忆
```

---

## 10. 技能系统 (`skills/`)

### 10.1 架构

```
skills/                          # 内置技能文档
├── creative/                    # 创意写作技能
├── media/                       # 媒体处理技能
├── mlops/                       # ML Ops 技能
├── productivity/               # 生产力技能
├── red-teaming/                # 红队测试技能
├── research/                    # 研究技能
└── ...

agent/skill_commands.py          # 斜杠命令处理（CLI + Gateway 共享）
agent/skill_utils.py             # 工具函数
tools/skills_tool.py             # 技能执行工具
tools/skill_manager_tool.py      # 技能管理工具
tools/skills_guard.py            # 安全守卫
tools/skills_hub.py              # Skills Hub 集成
tools/skills_sync.py             # 技能同步
```

### 10.2 技能管理

技能是存储在文件系统中的 Markdown 文档，每个技能包含：
- **元数据** (YAML frontmatter)：名称、描述、标签
- **指令** (Markdown)：LLM 应遵循的具体指令

通过工具调用进行管理：`skills_list`、`skill_view`、`skill_manage`。

---

## 11. 插件系统 (`plugins/`)

### 11.1 架构

```
plugins/
├── memory/                      # 记忆后端
│   ├── honcho/
│   ├── mem0/
│   ├── byterover/
│   ├── holographic/
│   ├── openviking/
│   ├── retaindb/
│   ├── supermemory/
│   └── hindsight/
├── context_engine/              # 上下文引擎扩展
├── image_gen/                   # 图像生成提供者
│   ├── openai/
│   └── openai-codex/
├── disk-cleanup/               # 磁盘清理工具
└── example-dashboard/          # 示例仪表盘
```

插件系统提供了记忆后端、图像生成提供者等可插拔组件，根据配置文件动态加载。

---

## 12. Cron 调度器 (`cron/`)

```
cron/
├── __init__.py
├── jobs.py        # 任务定义和执行
└── scheduler.py   # 基于 croniter 的调度循环
```

- 内置 cron 调度器，基于 `croniter` 库
- 支持投递到任何消息平台
- 通过 `cronjob` 工具或 CLI 进行管理

---

## 13. RL 训练环境 (`environments/`)

```
environments/
├── hermes_base_env.py       # 基础 RL 环境
├── agentic_opd_env.py       # OPD (开放问题难度) 环境
├── web_research_env.py      # 网络研究环境
├── agent_loop.py            # RL 训练的 Agent 循环
├── tool_context.py          # 工具上下文管理
├── patches.py               # 补丁工具
├── hermes_swe_env/          # SWE-bench 环境
├── terminal_test_env/       # 终端测试环境
├── tool_call_parsers/       # 每模型的工具调用解析器 (11个)
└── benchmarks/              # 基准测试环境
    ├── tblite/              # Tablebench lite
    ├── terminalbench_2/     # TerminalBench v2
    └── yc_bench/            # Y Combinator 基准
```

### 工具调用解析器 (`tool_call_parsers/`)

针对不同模型的工具调用格式进行解析：

```
tool_call_parsers/
├── hermes_parser.py
├── llama_parser.py
├── mistral_parser.py
├── qwen_parser.py
├── qwen3_coder_parser.py
├── deepseek_v3_parser.py
├── deepseek_v3_1_parser.py
├── glm45_parser.py
├── glm47_parser.py
├── kimi_k2_parser.py
└── longcat_parser.py
```

---

## 14. ACP 服务器 (`acp_adapter/`)

```
acp_adapter/
├── entry.py        # CLI 入口 (`hermes-acp`)
├── server.py       # ACP 服务器实现
├── session.py      # 会话管理
├── tools.py        # 通过 ACP 暴露工具
├── events.py       # 事件处理
├── auth.py         # 多提供商认证
└── permissions.py  # 权限管理
```

ACP 是 Anthropic 的 Agent Communication Protocol，用于 IDE 集成（VS Code、Zed、JetBrains）。

---

## 15. TUI 后端 (`tui_gateway/`)

```
tui_gateway/
├── entry.py            # stdio JSON-RPC 入口
├── server.py           # RPC 处理器和会话逻辑
├── render.py           # Rich/ANSI 桥接
└── slash_worker.py     # 持久化 HermesCLI 子进程（斜杠命令）
```

前端是 React Ink 应用 (`ui-tui/`)，通过 JSON-RPC over stdio 与后端通信。

---

## 16. 配置系统

### 16.1 配置文件

- **`~/.hermes/config.yaml`** — 用户配置（模型、提供商、工具集、网关等）
- **`~/.hermes/.env`** — API 密钥

### 16.2 Profile 系统

通过 `HERMES_HOME` 环境变量支持多 Profile 隔离。例如：

```
~/.hermes/                     # 默认
~/.hermes/profiles/coder/      # "coder" profile
~/.hermes/profiles/researcher/ # "researcher" profile
```

所有状态相关路径使用 `get_hermes_home()`（来自 `hermes_constants.py`），确保 Profile 兼容。

### 16.3 配置加载顺序

```
hermes_cli/env_loader.py  # 加载 .env 文件
hermes_cli/config.py      # 加载 config.yaml + DEFAULT_CONFIG + 迁移
```

---

## 17. 安全机制

- **`tools/approval.py`** — 危险命令检测（终端命令审批）
- **`tools/url_safety.py`** — URL 安全检查
- **`tools/path_security.py`** — 路径遍历防护
- **`tools/tirith_security.py`** — Tirith 安全沙箱
- **`tools/skills_guard.py`** — 技能执行安全守卫
- **`agent/redact.py`** — 敏感数据脱敏
- **`tools/file_safety.py`** — 文件操作安全检查

---

## 18. 其他重要组件

| 组件 | 文件 | 用途 |
|------|------|------|
| 日志系统 | `hermes_logging.py` | 滚动文件日志 + 脱敏格式化 |
| 对话标题生成 | `agent/title_generator.py` | 自动生成对话标题 |
| 对话洞察 | `agent/insights.py` | 对话洞察生成 |
| 费率限制 | `agent/rate_limit_tracker.py` | API 费率限制追踪 |
| 凭据池 | `agent/credential_pool.py` | 多密钥轮换/路由 |
| 用量计价 | `agent/usage_pricing.py` | 用量估算和成本计算 |
| 模型元数据 | `agent/model_metadata.py` | 模型上下文长度、Token 估算 |
| 提示缓存 | `agent/prompt_caching.py` | Anthropic 提示缓存支持 |
| 图像生成 | `agent/image_gen_provider/registry.py` | 多提供商图像生成 |
| Shell 钩子 | `agent/shell_hooks.py` | Shell 钩子执行 |
| 检查点 | `tools/checkpoint_manager.py` | Agent 检查点管理 |

---

## 19. 设计模式总结

| 模式 | 使用场景 | 示例 |
|------|---------|------|
| **自注册 (Self-Registration)** | 工具系统 | 每个 `tools/*.py` 在模块级 `registry.register()` |
| **策略模式 (Strategy)** | 执行后端 | `tools/environments/` 的 6 种后端 |
| **适配器模式 (Adapter)** | 消息平台、LLM 传输 | `gateway/platforms/*`、`agent/transports/*` |
| **抽象工厂** | 记忆/图像生成提供者 | `plugins/memory/*`、`plugins/image_gen/*` |
| **组合模式 (Composite)** | 工具集 | 工具集可以包含其他工具集 (`includes`) |
| **发布-订阅** | 网关钩子 | `gateway/hooks.py` |
| **桥接模式 (Bridge)** | 浏览器自动化 | `tools/browser_providers/*` 统一接口 |
| **单例模式** | 工具注册表 | `registry` 全局实例 |

---

## 20. 数据流全景图

```
                    ┌────────────────────────────────────────────────────┐
                    │                  用户交互界面                       │
                    │  ┌──────────┐  ┌────────┐  ┌────────┐  ┌─────────┐│
                    │  │CLI (TUI) │  │Gateway │  │ACP IDE │  │Batch/Rl ││
                    │  └────┬─────┘  └───┬────┘  └───┬────┘  └────┬────┘│
                    └───────┼────────────┼───────────┼────────────┼──────┘
                            │            │           │            │
                            ▼            ▼           ▼            ▼
                    ┌────────────────────────────────────────────────────┐
                    │              AIAgent (run_agent.py)                │
                    │  ┌──────────────────────────────────────────────┐  │
                    │  │  对话循环: 发送消息 → 解析响应 → 执行工具 →   │  │
                    │  │  → 继续直到完成                               │  │
                    │  └──────────────────────────────────────────────┘  │
                    │              │              │                      │
                    │         ┌────┴────┐   ┌────┴────┐                 │
                    │         │ 记忆系统  │   │ 技能系统  │                 │
                    │         └─────────┘   └─────────┘                 │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────┴────────────────────────────┐
                    │             model_tools.py (编排层)                 │
                    │              tools/registry.py (注册表)              │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────┴────────────────────────────┐
                    │              tools/*.py (工具实现)                  │
                    │  终端  │  文件  │  Web  │  浏览器  │  代码  │  ...  │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────┴────────────────────────────┐
                    │              传输层 (LLM API)                       │
                    │  Anthropic │ OpenAI │ Bedrock │ Gemini │ Codex     │
                    └────────────────────────────────────────────────────┘
```
