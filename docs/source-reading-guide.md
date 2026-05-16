# Hermes Agent 源码阅读指南

> 版本: 0.10.0 | 许可证: MIT | Python 3.11+ | 作者: Nous Research

Hermes Agent 是一个自改进型 AI Agent 框架，支持 20+ 聊天平台、多 LLM 提供商、内置工具系统、记忆系统和技能系统。本文档梳理其核心特性与代码结构，辅助源码阅读。

---

## 目录

1. [全局架构总览](#1-全局架构总览)
2. [核心数据流：一次对话请求的完整生命周期](#2-核心数据流一次对话请求的完整生命周期)
3. [核心模块详解](#3-核心模块详解)
4. [工具系统](#4-工具系统)
5. [LLM 提供商抽象层](#5-llm-提供商抽象层)
6. [多平台网关](#6-多平台网关)
7. [上下文与记忆系统](#7-上下文与记忆系统)
8. [插件系统](#8-插件系统)
9. [安全与权限体系](#9-安全与权限体系)
10. [推荐源码阅读顺序](#10-推荐源码阅读顺序)
11. [关键文件速查表](#11-关键文件速查表)

---

## 1. 全局架构总览

### 1.1 项目结构

```
hermes-agent/
├── run_agent.py          # 核心: AIAgent 类 (12,172 行)
├── cli.py                # 交互式 CLI REPL (11,096 行)
├── model_tools.py        # 工具编排层
├── toolsets.py           # 工具集定义
├── hermes_state.py       # SQLite 会话存储
├── mcp_serve.py          # MCP Server 实现
├── batch_runner.py       # 批量运行器 (RL/评测)
├── trajectory_compressor.py  # 轨迹压缩 (训练数据)
│
├── agent/                # Agent 内部模块 (~43 个)
│   ├── prompt_builder.py       # 系统提示词组装
│   ├── context_compressor.py   # 上下文压缩器
│   ├── context_engine.py       # 上下文引擎抽象
│   ├── memory_manager.py       # 记忆管理器
│   ├── memory_provider.py      # 外部记忆提供商抽象
│   ├── credential_pool.py      # API 密钥池
│   ├── auxiliary_client.py     # 辅助 LLM 客户端
│   ├── anthropic_adapter.py    # Anthropic 适配器
│   ├── bedrock_adapter.py      # AWS Bedrock 适配器
│   ├── gemini_native_adapter.py # Gemini 适配器
│   ├── codex_responses_adapter.py # Codex 适配器
│   ├── shell_hooks.py          # Shell 钩子
│   ├── transports/             # 传输层抽象
│   │   ├── base.py             # ProviderTransport ABC
│   │   ├── types.py            # NormalizedResponse, ToolCall, Usage
│   │   ├── chat_completions.py # OpenAI 兼容
│   │   ├── anthropic.py        # Anthropic Messages API
│   │   ├── codex.py            # OpenAI Responses API
│   │   └── bedrock.py          # AWS Bedrock Converse
│   └── ...
│
├── tools/                # 工具实现 (~60 个)
│   ├── registry.py             # 工具注册中心 (单例)
│   ├── terminal_tool.py        # 终端/Shell
│   ├── file_tools.py           # 文件读写/搜索/Patch
│   ├── web_tools.py            # 网络搜索/提取
│   ├── browser_tool.py         # 浏览器自动化
│   ├── mcp_tool.py             # MCP 客户端集成
│   ├── delegate_tool.py        # 子 Agent 委派
│   ├── approval.py             # 危险命令审批
│   ├── todo_tool.py            # 任务规划
│   ├── memory_tool.py          # 记忆操作
│   ├── skills_tool.py          # 技能管理
│   ├── code_execution_tool.py  # 沙盒代码执行
│   ├── environments/           # 终端后端 (local/docker/ssh/modal/daytona)
│   └── ...
│
├── hermes_cli/           # CLI 子命令和配置 (~52 个)
│   ├── main.py                 # hermes CLI 入口
│   ├── config.py               # 配置加载/保存/迁移
│   ├── setup.py                # 交互式设置向导
│   ├── commands.py             # 斜杠命令注册中心
│   ├── models.py               # 模型目录
│   ├── auth.py                 # 多提供商认证
│   └── plugins.py              # 插件系统核心
│
├── gateway/              # 多平台消息网关
│   ├── run.py                  # 网关主循环 (11,200 行)
│   ├── session.py              # 会话持久化
│   ├── config.py               # 网关配置
│   ├── hooks.py                # 网关事件钩子
│   ├── pairing.py              # DM 配对认证
│   └── platforms/              # 20 个平台适配器
│       ├── base.py             # BasePlatformAdapter ABC
│       ├── telegram.py         # Telegram
│       ├── discord.py          # Discord
│       ├── slack.py            # Slack
│       ├── whatsapp.py         # WhatsApp
│       ├── feishu.py           # 飞书
│       ├── weixin.py           # 微信
│       ├── matrix.py           # Matrix
│       ├── dingtalk.py         # 钉钉
│       ├── wecom.py            # 企业微信
│       └── ... (还有 10+ 平台)
│
├── skills/               # 内置技能 (27+ 分类)
├── optional-skills/      # 可选技能
├── plugins/              # 插件 (8 个记忆后端 + 其他)
│   └── memory/               # byterover, honcho, mem0, supermemory 等
│
├── ui-tui/               # Ink/React 终端 UI
├── tui_gateway/          # TUI 的 Python JSON-RPC 后端
├── web/                  # Vite + React Web 仪表盘
├── acp_adapter/          # ACP 协议 (VS Code/Zed/JetBrains 集成)
├── cron/                 # 内置定时调度器
├── environments/         # RL 训练环境 (Atropos)
├── tests/                # 测试套件 (~3000+)
└── website/              # Docusaurus 文档站
```

### 1.2 依赖链

代码按严格单向依赖组织，自底向上：

```
tools/registry.py         (零依赖，被所有工具导入)
       ↑
tools/*.py                (每个文件顶层调用 registry.register())
       ↑
model_tools.py            (导入 registry，触发工具发现)
       ↑
run_agent.py / cli.py / batch_runner.py / environments/
```

### 1.3 入口点

| 入口 | 命令 | 说明 |
|------|------|------|
| CLI 主入口 | `hermes` | `hermes_cli.main:main` |
| 独立 Agent | `hermes-agent` | `run_agent:main` |
| ACP 服务器 | `hermes-acp` | `acp_adapter.entry:main` |
| 消息网关 | `hermes gateway start` | `gateway/run.py` |
| 终端 UI | `hermes --tui` | `ui-tui/` (Ink/React) |
| Web 仪表盘 | 内嵌于 CLI 构建产物 | `web/` (Vite/React) |

---

## 2. 核心数据流：一次对话请求的完整生命周期

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  AIAgent.run_conversation(user_message, history)        │
│  (run_agent.py:8649)                                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 会话初始化                                            │
│     ├─ 清洗输入 (移除注入风险)                              │
│     ├─ 构建/恢复系统提示词 (缓存复用)                        │
│     ├─ 通知记忆提供商 (prefetch)                           │
│     └─ 预压缩上下文 (如果历史过长)                           │
│                                                         │
│  2. 主循环 while (api_calls < max_iterations):           │
│     │                                                   │
│     ├─ 2a. 构建 API Messages                             │
│     │   ├─ 注入临时记忆上下文                              │
│     │   ├─ 添加 prompt caching 标记 (Anthropic)           │
│     │   ├─ 归一化空白和 JSON (提升缓存命中率)                │
│     │   └─ 清理孤立 tool results                          │
│     │                                                   │
│     ├─ 2b. 调用 LLM API (流式优先)                        │
│     │   ├─ _build_api_kwargs() → transport.build_kwargs()│
│     │   ├─ _interruptible_streaming_api_call()           │
│     │   └─ transport.normalize_response() → 标准化        │
│     │                                                   │
│     ├─ 2c. 处理响应                                       │
│     │   ├─ 跟踪 token 用量，更新上下文压缩器                │
│     │   ├─ 处理截断 (finish_reason == "length")            │
│     │   └─ 处理空响应/仅 thinking 响应                     │
│     │                                                   │
│     ├─ 2d. 如果有工具调用:                                │
│     │   ├─ 验证工具名 (模糊匹配自修复)                      │
│     │   ├─ 验证 JSON 参数 (拒绝截断调用)                    │
│     │   ├─ _execute_tool_calls()                         │
│     │   │   ├─ 并发执行 (ThreadPoolExecutor)              │
│     │   │   ├─ 或顺序执行                                 │
│     │   │   └─ 插件钩子: pre_tool_call / post_tool_call    │
│     │   ├─ 将结果追加到 messages                           │
│     │   ├─ 检查是否需要上下文压缩                           │
│     │   └─ continue → 下一轮迭代                          │
│     │                                                   │
│     └─ 2e. 如果无工具调用 (最终回复):                       │
│         ├─ 剥离 think 块                                 │
│         └─ break → 结束循环                               │
│                                                         │
│  3. 后处理                                               │
│     ├─ 持久化会话到 SQLite                                 │
│     ├─ 刷新记忆 (MEMORY.md / USER.md)                     │
│     ├─ 保存轨迹 (训练数据)                                 │
│     └─ 返回结果字典                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
  │
  ▼
结果字典 {
    "final_response": str,
    "messages":      list,     # 完整对话历史
    "api_calls":     int,      # LLM 调用次数
    "completed":     bool,     # 是否正常完成
    "partial":       bool,     # 是否被截断
    "interrupted":   bool,     # 是否被用户中断
    "error":         str,      # 错误描述
}
```

---

## 3. 核心模块详解

### 3.1 AIAgent 类 (run_agent.py)

**位置**: `run_agent.py:679`

这是整个系统的核心，一个 ~11,000 行的编排类。构造函数接受 60+ 参数：

| 参数类别 | 示例参数 | 说明 |
|----------|---------|------|
| 模型/提供商 | `base_url`, `api_key`, `provider`, `model` | LLM 连接配置 |
| API 模式 | `api_mode` | `chat_completions` / `anthropic_messages` / `codex_responses` / `bedrock_converse` |
| 工具 | `enabled_toolsets`, `disabled_toolsets` | 工具集开关 |
| 迭代 | `max_iterations` (默认 90) | 最大工具调用轮次 |
| 回调 | `stream_delta_callback`, `tool_progress_callback` | 流式输出和进度通知 |
| 会话 | `session_id`, `session_db`, `platform` | 会话持久化 |
| 子 Agent | `iteration_budget`, `credential_pool` | 委派和凭证管理 |

**关键方法**:
- `run_conversation()` (line 8649) — 公共入口，执行完整对话循环
- `_build_system_prompt()` (line 4045) — 组装系统提示词
- `_invoke_tool()` (line 7694) — 工具调用路由
- `_execute_tool_calls()` (line 7652) — 工具并发/顺序执行
- `_build_api_kwargs()` (line 6918) — 构建提供商特定请求
- `_interruptible_streaming_api_call()` (line 5468) — 可中断的流式 API 调用

### 3.2 CLI (cli.py)

**位置**: `cli.py` — HermesCLI 类

基于 prompt_toolkit 的富终端 REPL，提供：
- 斜杠命令自动补全
- 流式 Markdown 输出
- 工具执行可视化（进度条、预览）
- 会话管理（/new, /reset, /model）
- 皮肤/主题引擎

### 3.3 会话存储 (hermes_state.py)

**位置**: `hermes_state.py:122` — SessionDB 类

- SQLite WAL 模式
- FTS5 全文搜索（跨会话召回）
- 线程安全写入（随机抖动重试）
- 存储：会话元数据、消息、token 用量、成本

---

## 4. 工具系统

### 4.1 工具注册中心 (tools/registry.py)

**设计模式**: 单例 + 自注册

```python
# 每个工具文件在模块顶层注册
registry.register(
    name="read_file",           # 工具名
    toolset="file",             # 所属工具集
    schema=READ_FILE_SCHEMA,    # OpenAI function-calling 格式
    handler=_handle_read_file,  # 执行函数
    check_fn=_check_file_reqs,  # 可用性检查 (如 API key)
    emoji="📖",
    max_result_size_chars=float('inf'),
)
```

**工具发现机制** (`discover_builtin_tools()`): AST 扫描 `tools/*.py` 中的 `registry.register()` 调用，只导入有注册的文件。

### 4.2 工具集 (toolsets.py)

工具按场景分组，可整体启用/禁用：

| 工具集 | 包含工具 | 说明 |
|--------|---------|------|
| `file` | read_file, write_file, patch, search_files | 文件操作 |
| `terminal` | terminal, process | Shell 执行 |
| `web` | web_search, web_extract | 网络搜索/提取 |
| `browser` | browser_navigate, browser_click, ... | 浏览器自动化 (11 个子工具) |
| `vision` | vision_analyze, image_generate | 视觉/图像 |
| `skills` | skills_list, skill_view, skill_manage | 技能管理 |
| `memory` | memory | 持久记忆 |
| `todo` | todo | 任务规划 |
| `delegate` | delegate_task | 子 Agent 委派 |
| `cron` | cronjob | 定时任务 |
| `messaging` | send_message | 跨平台消息 |
| `code_execution` | execute_code | 沙盒代码执行 |
| `session_search` | session_search | 会话历史搜索 |
| `tts` | text_to_speech | 文本转语音 |
| `homeassistant` | ha_* | 智能家居控制 |
| `mixture_of_agents` | mixture_of_agents | 多模型共识推理 |
| `rl_training` | rl_* | RL 训练管理 |

### 4.3 工具执行流程

```
LLM 返回 tool_calls
      │
      ▼
model_tools.handle_function_call(name, args)
      │
      ├─ coerce_tool_args()     # 类型转换 ("42" → 42)
      ├─ plugin pre_tool_call   # 插件可阻止执行
      │
      ▼
registry.dispatch(name, args)
      │
      ├─ 查找 ToolEntry
      ├─ 调用 handler(args, **kwargs)
      ├─ _run_async()           # 异步桥接
      └─ 异常捕获 → {"error": "..."}
      │
      ▼
maybe_persist_tool_result()    # 三层预算系统
      ├─ 每工具阈值 (100K 默认)
      ├─ 每轮预算 (200K 聚合)
      └─ 预览大小 (1.5K 内联摘要)
      │
      ▼
追加到 messages: {"role": "tool", "tool_call_id": ..., "content": ...}
```

**特殊工具路由** (`_invoke_tool()` line 7694):
`todo`, `memory`, `session_search`, `delegate_task`, `clarify` 由 AIAgent 直接处理（需要 agent 级状态），其余走 `model_tools.handle_function_call()`。

### 4.4 终端后端

终端工具支持多种执行环境：

| 后端 | 文件 | 说明 |
|------|------|------|
| 本地 | `tools/environments/local.py` | 本机 Shell |
| Docker | `tools/environments/docker.py` | Docker 容器 |
| SSH | `tools/environments/ssh.py` | 远程 SSH |
| Modal | `tools/environments/modal.py` | Modal 无服务器 |
| Daytona | `tools/environments/daytona.py` | Daytona 沙盒 |
| Singularity | `tools/environments/singularity.py` | 容器 (HPC) |
| Managed Modal | `tools/environments/managed_modal.py` | 托管 Modal |

---

## 5. LLM 提供商抽象层

### 5.1 传输层 (agent/transports/)

```
                   ┌──────────────────────┐
                   │   ProviderTransport   │  (base.py ABC)
                   │   - convert_messages  │
                   │   - convert_tools     │
                   │   - build_kwargs      │
                   │   - normalize_response│
                   └──────┬───────────────┘
                          │
          ┌───────────────┼───────────────┬──────────────┐
          │               │               │              │
   ChatCompletions   AnthropicTransport  CodexTransport  BedrockTransport
   (OpenAI 兼容)    (Messages API)     (Responses API)  (Converse API)
```

**NormalizedResponse** (types.py) — 标准化所有提供商的响应：

```python
@dataclass
class NormalizedResponse:
    content: Optional[str]              # 文本内容
    tool_calls: Optional[List[ToolCall]] # 工具调用
    finish_reason: str                  # "stop" / "tool_calls" / "length"
    reasoning: Optional[str]            # 推理/思考内容
    usage: Optional[Usage]              # token 用量
    provider_data: Optional[Dict]       # 提供商特定数据
```

### 5.2 API 模式自动检测

AIAgent 构造函数根据 `provider` 和 `base_url` 自动选择 API 模式：

| 提供商 | API 模式 | SDK |
|--------|---------|-----|
| OpenRouter, OpenAI, Qwen, Groq, ... | `chat_completions` | `openai` |
| Anthropic | `anthropic_messages` | `anthropic` |
| Codex | `codex_responses` | `openai` (Responses API) |
| AWS Bedrock | `bedrock_converse` | `boto3` |

### 5.3 流式输出

所有 API 模式均支持流式，运行在独立线程中，0.3 秒轮询中断请求：

- **chat_completions**: `stream=True`, SSE chunk 迭代
- **anthropic_messages**: `client.messages.stream()` 回调
- **codex_responses**: `_run_codex_stream` 专用流
- **bedrock_converse**: `converse_stream()` delta 回调

### 5.4 凭证池 (agent/credential_pool.py)

支持多密钥自动切换，策略包括：
- `fill_first` — 优先使用第一个可用密钥
- `round_robin` — 轮询
- `random` — 随机
- `least_used` — 最少使用

耗尽的凭证 (429/402) 冷却 1 小时。

---

## 6. 多平台网关

### 6.1 支持的平台 (20 个)

| 平台 | 适配器文件 | 大小 |
|------|-----------|------|
| Telegram | `gateway/platforms/telegram.py` | 139K |
| Discord | `gateway/platforms/discord.py` | 168K |
| WhatsApp | `gateway/platforms/whatsapp.py` | 46K |
| Slack | `gateway/platforms/slack.py` | 73K |
| Signal | `gateway/platforms/signal.py` | 40K |
| 飞书/Lark | `gateway/platforms/feishu.py` | 197K |
| 微信 | `gateway/platforms/weixin.py` | 79K |
| 钉钉 | `gateway/platforms/dingtalk.py` | 57K |
| 企业微信 | `gateway/platforms/wecom.py` | 66K |
| Matrix | `gateway/platforms/matrix.py` | 88K |
| Mattermost | `gateway/platforms/mattermost.py` | 28K |
| Home Assistant | `gateway/platforms/homeassistant.py` | 17K |
| Email | `gateway/platforms/email.py` | 24K |
| SMS (Twilio) | `gateway/platforms/sms.py` | 15K |
| API Server | `gateway/platforms/api_server.py` | 115K |
| Webhook | `gateway/platforms/webhook.py` | 31K |
| BlueBubbles (iMessage) | `gateway/platforms/bluebubbles.py` | 34K |
| WeCom (回调) | `gateway/platforms/wecom_callback.py` | 17K |
| QQ Bot | `gateway/platforms/qqbot/` | 子包 |

### 6.2 平台适配器抽象 (gateway/platforms/base.py)

所有适配器继承 `BasePlatformAdapter`：

```python
class BasePlatformAdapter(ABC):
    # 必须实现
    async def connect(self) -> bool
    async def disconnect(self)
    async def send(self, chat_id, content, reply_to, metadata) -> SendResult
    async def get_chat_info(self, chat_id) -> Dict

    # 可选实现 (有默认桩)
    async def edit_message(...)
    async def send_typing(...)
    async def send_image(...)
    async def send_document(...)
    async def send_voice(...)

    # 框架提供
    def set_message_handler(handler)
    def build_source(...) -> SessionSource  # 构造会话源
    def truncate_message(...)               # 智能消息分割
```

### 6.3 网关主循环 (gateway/run.py)

**位置**: `gateway/run.py` (11,200 行, 530KB)

网关是异步事件循环，负责：
- 连接所有启用的平台
- 分发入站消息到 AIAgent
- 处理斜杠命令 (/model, /new, /approve 等)
- 会话管理 (重置策略、活跃会话追踪)
- 流式消息推送 (progressive editMessageText)
- 事件钩子分发

### 6.4 DM 配对认证 (gateway/pairing.py)

未知用户需通过配对码授权：
- 8 字符码，32 字符无歧义字母表
- `secrets.choice()` 密码学随机
- 1 小时过期，每平台最多 3 个待处理
- 速率限制: 1 次/10 分钟/用户
- 5 次失败锁定 1 小时

---

## 7. 上下文与记忆系统

### 7.1 上下文引擎抽象 (agent/context_engine.py)

```python
class ContextEngine(ABC):
    def update_from_response(self, response)     # 跟踪 token 用量
    def should_compress(self) -> bool             # 判断是否需要压缩
    def compress(self, messages) -> list          # 执行压缩
    def on_session_start/end/reset(self)          # 生命周期
    def update_model(self, model)                 # 模型切换
```

通过 `context.engine` 配置项选择实现，默认 `compressor`。

### 7.2 上下文压缩器 (agent/context_compressor.py)

默认实现，使用廉价辅助模型压缩中间轮次：

```
原始消息: [系统提示] [用户1] [助手1] [工具1] [助手2] [用户2] ... [用户N] [助手N]
                                │                            │
                          保护头部 (前3条)              保护尾部 (后6条)
                                │                            │
                    ┌───────────┘                            │
                    ▼                                        │
              用辅助模型摘要中间内容                           │
              (含 Resolved/Pending 问题追踪)                   │
                    │                                        │
                    ▼                                        ▼
压缩后: [系统提示] [用户1] [助手1] [摘要: "已完成X，待处理Y"] [用户N] [助手N]
```

**关键配置**:
- `compression.enabled`: true
- `compression.threshold`: 0.50 (超过上下文 50% 时压缩)
- `compression.target_ratio`: 0.20
- `compression.protect_last_n`: 20 (保留最近 20 条消息)

### 7.3 记忆系统

**两层架构**:

```
┌─────────────────────────────────────────────┐
│            MemoryManager                     │
│  (agent/memory_manager.py)                   │
├─────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  内置记忆         │  │  外部记忆提供商    │  │
│  │  MEMORY.md (2200字)│  │  (8 个可选)       │  │
│  │  USER.md  (1375字) │  │  honcho, mem0,   │  │
│  │                   │  │  supermemory,     │  │
│  │  每 10 轮提醒保存  │  │  byterover,      │  │
│  │  压缩前/重置前刷新 │  │  hindsight,      │  │
│  └─────────────────┘  │  holographic,     │  │
│                        │  openviking,      │  │
│                        │  retaindb         │  │
│                        └──────────────────┘  │
└─────────────────────────────────────────────┘
```

**外部记忆提供商生命周期** (MemoryProvider ABC):
```
is_available() → initialize() → system_prompt_block()
  → prefetch(query) ← 每次 API 调用前
  → sync_turn(user, asst) ← 每轮后
  → shutdown()
```

### 7.4 会话重置策略

| 模式 | 行为 |
|------|------|
| `daily` | 每天固定时间重置 (默认凌晨 4 点) |
| `idle` | 不活跃 N 分钟后重置 (默认 24 小时) |
| `both` | 先触发者生效 |
| `none` | 从不自动重置，依赖上下文压缩 |

---

## 8. 插件系统

### 8.1 插件来源 (优先级从低到高)

1. **内置插件**: `plugins/<name>/` (随项目发布)
2. **用户插件**: `~/.hermes/plugins/<name>/`
3. **项目插件**: `./.hermes/plugins/<name>/` (需显式启用)
4. **Pip 插件**: 通过 `hermes_agent.plugins` entry-point 组暴露

### 8.2 插件接口

每个插件需要:
- `plugin.yaml` — 清单文件 (name, version, provides_tools, provides_hooks)
- `__init__.py` — `register(ctx: PluginContext)` 函数

**PluginContext API**:
- `register_tool()` — 注册工具 (与内置工具同等地位)
- `register_hook()` — 注册生命周期钩子
- `register_cli_command()` — 添加 CLI 子命令
- `register_command()` — 添加会话内斜杠命令
- `register_memory_provider()` — 注册记忆后端
- `register_context_engine()` — 注册上下文引擎
- `inject_message()` — 向活跃对话注入消息

### 8.3 生命周期钩子 (13 个)

| 钩子 | 触发时机 | 可阻止? |
|------|---------|--------|
| `pre_tool_call` | 工具执行前 | **是** (返回消息阻止) |
| `post_tool_call` | 工具执行后 | 否 |
| `transform_terminal_output` | 终端输出后 | 否 (可修改) |
| `transform_tool_result` | 工具结果后 | 否 (可替换) |
| `pre_llm_call` | LLM 调用前 | 否 |
| `post_llm_call` | LLM 调用后 | 否 |
| `pre_api_request` | HTTP 请求前 | 否 |
| `post_api_request` | HTTP 请求后 | 否 |
| `on_session_start` | 会话开始 | 否 |
| `on_session_end` | 会话结束 | 否 |
| `on_session_finalize` | 会话最终化 | 否 |
| `on_session_reset` | 会话重置 | 否 |
| `subagent_stop` | 子 Agent 完成 | 否 |

### 8.4 三层钩子系统

Hermes 有三个独立但可组合的钩子系统：

| 钩子层 | 位置 | 触发场景 |
|--------|------|---------|
| 网关事件钩子 | `gateway/hooks.py` | gateway:startup, session:*, agent:*, command:* |
| 插件生命周期钩子 | `hermes_cli/plugins.py` | 工具调用、LLM 调用、会话生命周期 |
| Shell 脚本钩子 | `agent/shell_hooks.py` | 自定义 shell 脚本，接收/返回 JSON |

---

## 9. 安全与权限体系

### 9.1 危险命令审批 (tools/approval.py)

```
终端命令进入
    │
    ▼
┌─ 规范化 (去除 ANSI、null 字节、全角字符)
│
├─ 30+ 正则模式匹配:
│  ├─ 递归删除 (rm -rf)
│  ├─ 文件系统格式化 (mkfs)
│  ├─ SQL DROP/DELETE 无 WHERE
│  ├─ Fork 炸弹
│  ├─ curl|sh
│  ├─ git reset --hard / push --force
│  └─ 自终止命令
│
├─ 敏感路径检查 (/etc/, ~/.ssh/, ~/.hermes/.env)
│
├─ 审批流程:
│  ├─ CLI: 交互式提示
│  ├─ Gateway: 异步阻塞队列 → 用户 /approve 或 /deny
│  ├─ 智能审批: 辅助 LLM 判断低风险命令
│  └─ 审批模式: once / session / permanent / deny
│
└─ contextvars 隔离 → 并发安全
```

### 9.2 多层权限防护

| 层 | 机制 |
|----|------|
| 工具集过滤 | 只启用的工具集中的工具才会提供给模型 |
| check_fn 门控 | 运行时检查 (如 API key 是否存在) |
| 插件钩子 | `pre_tool_call` 可阻止任何工具调用 |
| 子 Agent 限制 | `DELEGATE_BLOCKED_TOOLS` 禁止递归委派、用户交互等 |
| 代码执行沙盒 | `execute_code` 仅允许 7 个安全工具 |
| URL/路径安全 | `url_safety.py` + `path_security.py` 验证 |
| MCP 环境过滤 | stdio 子进程获得过滤后的环境变量 |
| 上下文文件扫描 | 10+ 娅提示注入模式检测 |

### 9.3 认证体系 (hermes_cli/auth.py)

三种认证类型:

| 类型 | 提供商 | 机制 |
|------|--------|------|
| OAuth Device Code | Nous Portal | 设备授权 + 轮询 + Token 刷新 |
| OAuth External | OpenAI Codex, Qwen, Gemini | 浏览器 OAuth + 本地回调服务器 |
| API Key | OpenRouter, Gemini, Kimi, ... | 环境变量 |

认证状态持久化到 `~/.hermes/auth.json`，跨进程文件锁保护。

---

## 10. 推荐源码阅读顺序

### 阶段一: 理解核心循环 (2-3 小时)

```
1. run_agent.py
   ├─ 192: IterationBudget 类 (理解迭代控制)
   ├─ 679: AIAgent.__init__() (理解所有配置)
   ├─ 4045: _build_system_prompt() (理解提示词组装)
   └─ 8649: run_conversation() (理解主循环)
```

### 阶段二: 理解工具系统 (1-2 小时)

```
2. tools/registry.py
   ├─ 76: ToolEntry 数据类
   ├─ 100: ToolRegistry 类
   └─ 56: discover_builtin_tools() (AST 发现)

3. model_tools.py
   ├─ get_tool_definitions() (工具 schema 过滤)
   ├─ handle_function_call() (工具分发)
   └─ coerce_tool_args() (参数类型转换)

4. tools/file_tools.py (示例工具，理解注册模式)
5. tools/approval.py (理解安全审批)
```

### 阶段三: 理解传输层 (1 小时)

```
6. agent/transports/base.py (ProviderTransport ABC)
7. agent/transports/types.py (NormalizedResponse)
8. agent/transports/chat_completions.py (最常用的传输)
9. agent/transports/anthropic.py (Anthropic 特有逻辑)
```

### 阶段四: 理解上下文管理 (1 小时)

```
10. agent/context_engine.py (ContextEngine ABC)
11. agent/context_compressor.py (默认压缩实现)
12. agent/memory_manager.py (记忆编排)
13. agent/memory_provider.py (外部记忆抽象)
```

### 阶段五: 理解多平台 (1-2 小时)

```
14. gateway/platforms/base.py (BasePlatformAdapter ABC)
15. gateway/config.py (Platform 枚举, PlatformConfig)
16. gateway/run.py: 前 200 行 (网关初始化和主循环)
17. gateway/session.py (会话持久化)
18. gateway/pairing.py (DM 配对)
```

### 阶段六: 理解插件与扩展 (30 分钟)

```
19. hermes_cli/plugins.py (PluginContext, 注册机制)
20. plugins/memory/__init__.py (记忆插件发现)
21. tools/mcp_tool.py: 前 100 行 (MCP 客户端入口)
22. mcp_serve.py: 前 100 行 (MCP 服务端入口)
```

### 阶段七: 可选深入

```
23. cli.py: HermesCLI 类 (交互式 REPL)
24. hermes_cli/commands.py (斜杠命令注册)
25. hermes_cli/config.py (配置系统)
26. agent/shell_hooks.py (Shell 钩子)
27. agent/credential_pool.py (凭证池)
28. tools/delegate_tool.py (子 Agent 委派)
```

---

## 11. 关键文件速查表

### 核心编排

| 文件 | 行数 | 说明 |
|------|------|------|
| `run_agent.py` | 12,172 | AIAgent 类，主对话循环 |
| `cli.py` | 11,096 | HermesCLI 交互式 REPL |
| `model_tools.py` | 617 | 工具编排层 |
| `toolsets.py` | 720 | 工具集定义 |

### Agent 内部模块

| 文件 | 说明 |
|------|------|
| `agent/prompt_builder.py` | 系统提示词片段 |
| `agent/context_compressor.py` | 上下文压缩器 |
| `agent/context_engine.py` | 上下文引擎抽象 |
| `agent/memory_manager.py` | 记忆管理器 |
| `agent/memory_provider.py` | 外部记忆提供商抽象 |
| `agent/credential_pool.py` | API 密钥池 |
| `agent/auxiliary_client.py` | 辅助 LLM 客户端 (134K) |
| `agent/shell_hooks.py` | Shell 脚本钩子 |
| `agent/transports/base.py` | ProviderTransport ABC |
| `agent/transports/types.py` | 标准化响应类型 |

### 工具系统

| 文件 | 说明 |
|------|------|
| `tools/registry.py` | 工具注册中心 |
| `tools/terminal_tool.py` | 终端执行 (89K) |
| `tools/file_tools.py` | 文件操作 |
| `tools/web_tools.py` | 网络搜索 (89K) |
| `tools/browser_tool.py` | 浏览器自动化 (103K) |
| `tools/mcp_tool.py` | MCP 客户端 (107K) |
| `tools/delegate_tool.py` | 子 Agent 委派 (91K) |
| `tools/approval.py` | 危险命令审批 (42K) |
| `tools/code_execution_tool.py` | 沙盒代码执行 |
| `tools/tool_result_storage.py` | 结果预算系统 |

### 网关与平台

| 文件 | 说明 |
|------|------|
| `gateway/run.py` | 网关主循环 (530K) |
| `gateway/config.py` | 网关配置 |
| `gateway/platforms/base.py` | 平台适配器抽象 |
| `gateway/session.py` | 会话持久化 |
| `gateway/pairing.py` | DM 配对认证 |
| `gateway/hooks.py` | 网关事件钩子 |

### 配置与 CLI

| 文件 | 说明 |
|------|------|
| `hermes_cli/config.py` | 配置加载/保存 (166K) |
| `hermes_cli/main.py` | CLI 入口/子命令 (8,865 行) |
| `hermes_cli/commands.py` | 斜杠命令注册中心 |
| `hermes_cli/auth.py` | 多提供商认证 (139K) |
| `hermes_cli/plugins.py` | 插件系统 (46K) |
| `hermes_cli/setup.py` | 交互式设置向导 (135K) |
| `hermes_state.py` | SQLite 会话存储 |

### 持久化与状态

| 文件 | 说明 |
|------|------|
| `hermes_state.py` | SQLite + FTS5 会话 DB |
| `hermes_constants.py` | HERMES_HOME 等常量 |
| `cron/scheduler.py` | 定时任务调度 (48K) |
| `cron/jobs.py` | 任务定义 (28K) |

### MCP

| 文件 | 说明 |
|------|------|
| `tools/mcp_tool.py` | MCP 客户端 (107K) |
| `mcp_serve.py` | MCP 服务端 (867 行) |
| `tools/mcp_oauth.py` | MCP OAuth 2.1 |
| `hermes_cli/mcp_config.py` | MCP 配置管理 |

### 集成与扩展

| 文件 | 说明 |
|------|------|
| `acp_adapter/server.py` | ACP 服务器 (编辑器集成) |
| `acp_adapter/permissions.py` | ACP 权限桥接 |
| `batch_runner.py` | 批量运行器 |
| `trajectory_compressor.py` | 轨迹压缩 (训练数据) |
| `environments/` | RL 训练环境 |

---

> **提示**: 此文档基于 v0.10.0 源码分析。建议配合 `AGENTS.md` (项目自带的 AI 开发者指南) 和 `cli-config.yaml.example` (50K 配置参考) 一起阅读。
