# Hermes Agent 代码架构梳理

> 版本: v0.10.0 | Nous Research 出品 | "自我进化的 AI Agent"

---

## 整体定位

Hermes Agent 是一个 Python 3.11+ 实现的 LLM Agent 框架，具备：
- 终端交互式 CLI（prompt_toolkit + Rich）
- React Ink TUI（TypeScript，通过 JSON-RPC 与 Python 通信）
- 多平台消息网关（Telegram、Discord、Slack、WhatsApp、Signal、Email、Matrix、飞书、企业微信、钉钉等 16+ 平台）
- 40+ 内置工具（终端执行、文件操作、浏览器自动化、网页搜索、MCP 客户端、子代理委派等）
- 技能系统（自主创建和改进技能，过程性记忆）
- 记忆系统（SQLite + FTS5 全文检索）
- 定时任务调度（内置 cron）
- RL 训练环境（Atropos 集成）
- IDE 集成（VS Code / Zed / JetBrains 通过 ACP 协议）

---

## 进程模型

```
hermes CLI 入口 (hermes_cli/main.py)
    │
    ├─ 交互式 CLI 模式 → HermesCLI (cli.py) → AIAgent (run_agent.py)
    │
    ├─ --tui 模式 → Node/Ink (ui-tui/) ──stdio JSON-RPC──→ tui_gateway/ → AIAgent
    │
    ├─ gateway start → gateway/run.py → AIAgent（多平台消息循环）
    │
    ├─ acp adapter → acp_adapter/ → AIAgent（IDE 集成）
    │
    └─ batch_runner.py → AIAgent（批量/并行处理）
```

所有模式最终都汇聚到同一个 `AIAgent` 核心循环。

---

## 核心依赖链

```
tools/registry.py          ← 零依赖，所有工具文件的注册入口
       ↑
tools/*.py                 ← 每个文件在模块级别调用 registry.register()
       ↑
model_tools.py             ← 导入 registry，触发工具发现，暴露 get_tool_definitions() / handle_function_call()
       ↑
run_agent.py               ← AIAgent 类，核心对话循环
cli.py / batch_runner.py / gateway/run.py / tui_gateway/ / acp_adapter/
```

这个依赖链设计得很清晰：`registry.py` 是叶子节点，不依赖任何其他内部模块，避免了循环导入。每个工具文件自注册，`model_tools.py` 通过 `discover_builtin_tools()` 自动发现并导入所有带 `registry.register()` 的模块。

---

## 目录结构与职责

### 核心层

| 目录/文件 | 职责 | 文件数 |
|---|---|---|
| `run_agent.py` | AIAgent 类 — 核心对话循环、LLM 调用、工具执行、上下文压缩 | 1 |
| `model_tools.py` | 工具编排层 — 发现工具、获取定义、处理函数调用分发 | 1 |
| `toolsets.py` | 工具集定义 — _HERMES_CORE_TOOLS 列表，控制哪些工具在哪启用 | 1 |
| `cli.py` | HermesCLI 类 — 交互式终端编排器（prompt_toolkit + Rich） | 1 |
| `hermes_state.py` | SessionDB — SQLite 会话存储 + FTS5 全文检索 | 1 |
| `hermes_constants.py` | 全局常量、路径辅助函数（get_hermes_home 等） | 1 |
| `batch_runner.py` | 并行批处理 | 1 |

### agent/ — Agent 内部实现（48 文件）

Agent 的内部模块，从 `run_agent.py` 中拆分出来以保持模块化：

| 模块 | 职责 |
|---|---|
| `prompt_builder.py` | 系统提示词组装（身份、平台提示、记忆指引、技能指引） |
| `context_compressor.py` | 自动上下文压缩（当 token 接近窗口限制时） |
| `prompt_caching.py` | Anthropic prompt caching 控制 |
| `auxiliary_client.py` | 辅助 LLM 客户端（视觉理解、摘要生成等轻量任务） |
| `model_metadata.py` | 模型上下文长度信息、token 估算 |
| `memory_manager.py` | 记忆系统 — 构建记忆上下文块、清理上下文 |
| `error_classifier.py` | API 错误分类，决定是否 failover |
| `retry_utils.py` | 带抖动的指数退避重试 |
| `usage_pricing.py` | 用量成本估算 |
| `transports/` | 多 provider 传输层适配（Anthropic、Bedrock、ChatCompletions、Codex） |
| `anthropic_adapter.py` | Anthropic API 专用适配 |
| `bedrock_adapter.py` | AWS Bedrock 适配 |
| `gemini_native_adapter.py` | Google Gemini 原生适配 |
| `credential_pool.py` | 多凭证轮换池 |
| `display.py` | KawaiiSpinner、工具预览格式化 |

### tools/ — 工具实现（77 文件）

自注册的工具系统，每个文件对应一个或一组工具：

**文件操作类：**
- `file_tools.py` — 文件读写/搜索/补丁
- `file_operations.py` — 文件操作辅助
- `file_state.py` — 文件状态跟踪
- `patch_parser.py` — 补丁解析

**终端执行类：**
- `terminal_tool.py` — 终端编排（6 种后端：local, docker, ssh, modal, daytona, singularity）
- `process_registry.py` — 后台进程管理
- `code_execution_tool.py` — 代码沙箱执行

**Web/Browser 类：**
- `web_tools.py` — 网页搜索/提取（Parallel + Firecrawl）
- `browser_tool.py` — Browserbase 浏览器自动化
- `browser_camofox.py` / `browser_cdp_tool.py` — 其他浏览器实现
- `browser_providers/` — 浏览器提供者抽象层

**通信类：**
- `mcp_tool.py` — MCP 客户端（~1050 行）
- `delegate_tool.py` — 子代理委派
- `send_message_tool.py` — 消息发送

**技能/记忆类：**
- `skills_tool.py` — 技能管理
- `skill_manager_tool.py` — 技能管理器
- `skills_sync.py` / `skills_guard.py` / `skills_hub.py` — 技能同步、保护、搜索
- `memory_tool.py` — 记忆读写
- `session_search_tool.py` — 会话搜索

**多媒体类：**
- `vision_tools.py` — 图像理解
- `image_generation_tool.py` — 图像生成
- `tts_tool.py` — 文字转语音
- `transcription_tools.py` — 语音转文字
- `voice_mode.py` — 语音交互模式

**安全类：**
- `approval.py` — 危险命令检测
- `path_security.py` — 路径安全检查
- `url_safety.py` — URL 安全检查
- `tirith_security.py` — 安全扫描
- `osv_check.py` — 漏洞检查

**环境后端（tools/environments/，11 文件）：**
- `base.py` → `local.py`, `docker.py`, `ssh.py`, `modal.py`, `daytona.py`, `singularity.py`
- `file_sync.py` — 远程文件同步
- `managed_modal.py` / `modal_utils.py` — Modal 托管环境

### hermes_cli/ — CLI 子命令与配置（49 文件）

| 模块 | 职责 |
|---|---|
| `main.py` | 主入口 — 所有 `hermes` 子命令 |
| `commands.py` | 斜杠命令注册表（CommandDef） |
| `config.py` | DEFAULT_CONFIG, OPTIONAL_ENV_VARS, 配置迁移 |
| `setup.py` | 交互式设置向导 |
| `callbacks.py` | 终端回调（确认、sudo、审批） |
| `skin_engine.py` | 皮肤/主题引擎 |
| `banner.py` | 启动 Banner 展示 |
| `models.py` | 模型目录 |
| `model_switch.py` | 共享模型切换逻辑 |
| `auth.py` | Provider 凭证解析 |
| `profiles.py` | 多实例配置文件管理 |
| `doctor.py` | 诊断工具 |
| `skills_hub.py` | 技能搜索/浏览/安装 |

### gateway/ — 消息网关（31+ 文件）

| 模块 | 职责 |
|---|---|
| `run.py` | 主循环、斜杠命令、消息分发 |
| `session.py` | SessionStore — 对话持久化 |
| `platforms/base.py` | 平台适配器基类 |
| `platforms/telegram.py` | Telegram Bot 适配 |
| `platforms/discord.py` | Discord 适配 |
| `platforms/slack.py` | Slack 适配 |
| `platforms/whatsapp.py` | WhatsApp 适配 |
| `platforms/signal.py` | Signal 适配 |
| `platforms/email.py` | 邮件适配 |
| `platforms/matrix.py` | Matrix 适配 |
| `platforms/feishu.py` | 飞书适配 |
| `platforms/wecom.py` | 企业微信适配 |
| `platforms/weixin.py` | 微信适配 |
| `platforms/dingtalk.py` | 钉钉适配 |
| `platforms/homeassistant.py` | Home Assistant 适配 |
| `platforms/mattermost.py` | Mattermost 适配 |
| `platforms/bluebubbles.py` | BlueBubbles (iMessage) 适配 |
| `platforms/sms.py` | SMS 适配 |

### ui-tui/ + tui_gateway/ — React Ink TUI

```
hermes --tui
  └─ Node (Ink) ──stdio JSON-RPC── Python (tui_gateway)
       │                                └─ AIAgent + tools + sessions
       └─ 渲染：transcript, composer, prompts, activity
```

- `ui-tui/`：TypeScript/React Ink 前端（输入、补全、主题、Markdown 渲染）
- `tui_gateway/`：Python JSON-RPC 后端（5 文件：entry, server, render, slash_worker）

### acp_adapter/ — IDE 集成（9 文件）

通过 ACP（Agent Communication Protocol）为 VS Code / Zed / JetBrains 提供 agent 能力。

### cron/ — 定时调度（3 文件）

`jobs.py`（任务定义）+ `scheduler.py`（调度引擎），支持向任意消息平台投递。

### environments/ — RL 训练环境

为 Atropos RL 框架提供训练环境，包括 Agent loop、基准测试（TerminalBench、YC Bench 等）。

### tests/ — 测试套件（706 个 .py 文件）

覆盖所有模块，使用 pytest + xdist（4 workers 匹配 CI）。通过 `scripts/run_tests.sh` 确保本地和 CI 环境一致。

---

## AIAgent 核心循环

`run_agent.py` 中的 `AIAgent` 类是整个系统的引擎。核心循环是完全同步的：

```python
while api_call_count < max_iterations and iteration_budget.remaining > 0:
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tool_schemas
    )
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args, task_id)
            messages.append(tool_result_message(result))
        api_call_count += 1
    else:
        return response.content
```

关键设计点：
- 消息格式统一为 OpenAI 格式：`{"role": "system/user/assistant/tool", ...}`
- 推理内容存储在 `assistant_msg["reasoning"]` 字段
- 支持自动上下文压缩（当 token 接近窗口限制时）
- 支持 Anthropic prompt caching（system prompt 和工具定义可缓存）
- 支持多 provider 自动 failover（错误分类 → 切换 provider）

---

## 工具注册机制

工具系统是整个项目最精妙的设计之一：

```python
# tools/xxx_tool.py
from tools.registry import registry

registry.register(
    name="tool_name",
    toolset="toolset_name",
    schema={...},           # OpenAI function calling schema
    handler=lambda args, **kw: ...,
    check_fn=check_requirements,  # 检查依赖是否满足
    requires_env=["API_KEY"],     # 依赖的环境变量
)
```

自动发现机制：
1. `registry.py` 用 AST 解析扫描 `tools/*.py` 中是否有顶层 `registry.register()` 调用
2. `discover_builtin_tools()` 自动导入这些模块（触发注册）
3. `model_tools.py` 调用 `discover_builtin_tools()`，然后从 registry 获取工具定义
4. 新增工具只需创建文件 + 在 `toolsets.py` 注册，零手动导入

---

## 配置系统

| 配置文件 | 内容 |
|---|---|
| `~/.hermes/config.yaml` | 用户设置（模型、显示、工具等） |
| `~/.hermes/.env` | API 密钥和凭证 |
| `~/.hermes/skins/*.yaml` | 自定义皮肤 |
| `~/.hermes/skills/` | 用户技能 |

配置加载有三套独立的加载器：
- `load_cli_config()` — CLI 模式使用
- `load_config()` — `hermes tools` / `hermes setup` 使用
- Gateway 直接加载 YAML

---

## 斜杠命令系统

所有斜杠命令在 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY` 中集中定义为 `CommandDef` 对象。一次定义，多处自动派生：

- CLI 的 `process_command()` 通过 `resolve_command()` 分发
- Gateway 的 `GATEWAY_KNOWN_COMMANDS` 集合
- Telegram 的 BotCommand 菜单
- Slack 的 `/hermes` 子命令路由
- TUI 的自动补全
- 帮助文本自动生成

---

## 多实例（Profiles）系统

通过 `HERMES_HOME` 环境变量实现完全隔离的多实例：
- `_apply_profile_override()` 在模块导入前设置 `HERMES_HOME`
- 所有 119+ 处 `get_hermes_home()` 引用自动指向正确的 profile 目录
- 每个 profile 有独立的 config、API keys、memory、sessions、skills、gateway

---

## 皮肤/主题系统

纯数据驱动的 CLI 主题系统（`hermes_cli/skin_engine.py`）：
- 内置皮肤：default（金色卡哇伊）、ares（深红战神）、mono（灰度极简）、slate（蓝色开发者）
- 用户自定义：`~/.hermes/skins/<name>.yaml`
- 可定制：Banner 颜色、Spinner 动画/文字、工具前缀、工具 emoji、Agent 名称、欢迎消息

---

## 测试策略

- 706 个测试文件，约 3000 个测试用例
- 必须通过 `scripts/run_tests.sh` 运行（确保与 CI 环境一致）
- 4 个 xdist worker（匹配 GitHub Actions）
- `tests/conftest.py` 提供 `_isolate_hermes_home` 自动 fixture，将 `~/.hermes` 重定向到临时目录
- 强调行为测试，禁止变更检测测试（不断言特定模型名、版本号等会变化的数据）
