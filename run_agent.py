#!/usr/bin/env python3
"""
Hermes Agent 核心运行模块 — AI Agent Runner with Tool Calling
=================================================================

本模块是 Hermes Agent 的**核心引擎**，提供了一个完整的 AI Agent 实现，
支持工具调用 (Tool Calling)、多模型提供商、流式响应、上下文压缩、
会话持久化、中断/steer 机制、子 Agent 委托等功能。

## 核心架构

```
                        ┌─────────────────────────┐
                        │     run_conversation()   │  ← 主入口: 接收用户消息, 返回最终响应
                        │   (核心对话循环)          │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ _build_system_   │  │ _interruptible_  │  │ _execute_tool_   │
   │ prompt()         │  │ api_call()       │  │ calls()          │
   │ 系统提示词构建     │  │ LLM API 调用      │  │ 工具调用执行       │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
                                              │
                           ┌──────────────────┼──────────────────┐
                           ▼                  ▼                  ▼
                    ┌──────────┐     ┌──────────────┐    ┌──────────────┐
                    │ 并发执行   │     │ 顺序执行       │    │ _invoke_tool │
                    │ (线程池)  │     │ (逐个调用)     │    │ 单工具路由    │
                    └──────────┘     └──────────────┘    └──────────────┘
```

## 核心对话循环 (run_conversation) 流程图

```

用户消息输入
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段一: 初始化                                                │
│   • 安装安全 IO 包装器 (_SafeWriter)                          │
│   • 清洗用户输入 (surrogate 字符, memory-context 泄漏)         │
│   • 重置重试计数器, 迭代预算, task_id                          │
│   • 清理死连接 (_cleanup_dead_connections)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段二: 系统提示词构建                                         │
│   • 检查缓存 (_cached_system_prompt)                          │
│   • 若为持续会话: 从 SQLite 加载已存储的 system prompt         │
│   • 若为新会话: 调用 _build_system_prompt() 从头构建           │
│     (SOUL.md → 工具指导 → 记忆 → 技能 → 上下文文件 → 元数据)    │
│   • 将 system prompt 快照存入 SQLite                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段三: 预压缩 (Preflight Compression)                        │
│   • 估算消息 + system prompt + tool schema 的总 token 数      │
│   • 若超出阈值: 调用 _compress_context() 压缩历史              │
│   • 最多尝试 3 轮压缩 (适配超大 session → 小 context 场景)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段四: 插件钩子 + 外部记忆预取                                 │
│   • 调用 pre_llm_call 插件钩子 (注入临时上下文到 user message)  │
│   • 外部记忆提供商预取 (Honcho, Mem0 等)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │       阶段五: 主循环                  │
         │   while api_call_count < max_iter    │
         │   AND iteration_budget > 0:          │
         │                                     │
         │   ┌─────────────────────────────┐   │
         │   │ 检查中断 (interrupt_requested)│   │
         │   │ 消耗迭代预算 (budget.consume) │   │
         │   │ 排空 /steer 消息              │   │
         │   └─────────────┬───────────────┘   │
         │                 │                   │
         │                 ▼                   │
         │   ┌─────────────────────────────┐   │
         │   │ 构建 API 消息 (api_messages)  │   │
         │   │ • 注入临时上下文到 user msg   │   │
         │   │ • 复制 reasoning_content     │   │
         │   │ • 清理内部字段                │   │
         │   └─────────────┬───────────────┘   │
         │                 │                   │
         │                 ▼                   │
         │   ┌─────────────────────────────┐   │
         │   │ _interruptible_api_call()   │   │
         │   │ (或流式版本)                  │   │
         │   │ • 调用 LLM API               │   │
         │   │ • 处理 4xx/5xx 错误           │   │
         │   │ • 失败时自动切换 fallback     │   │
         │   └─────────────┬───────────────┘   │
         │                 │                   │
         │                 ▼                   │
         │   ┌─────────────────────────────┐   │
         │   │ 处理 API 响应                 │   │
         │   │ • 有 tool_calls? ────┐       │   │
         │   │ • 纯文本? ──→ 退出循环 │       │   │
         │   └─────────────┬───────┼───┘   │
         │                 │       │       │
         │                 │       ▼       │
         │                 │  ┌──────────────────┐
         │                 │  │ _execute_tool_   │
         │                 │  │ calls()          │
         │                 │  │ • 并发/顺序判断   │
         │                 │  │ • _invoke_tool() │
         │                 │  │ • 追加结果到 msg  │
         │                 │  └────────┬─────────┘
         │                 │           │
         │                 └───→ 继续循环 ←──┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段六: 后处理                                                │
│   • 持久化会话到 SQLite                                       │
│   • 保存 trajectory (如开启)                                   │
│   • 保存 session log                                         │
│   • 触发 post_llm_call 插件钩子                                │
│   • 返回最终结果 Dict: {final_response, messages, api_calls}   │
└─────────────────────────────────────────────────────────────┘
```

## 关键组件说明

| 组件 | 类/方法 | 职责 |
|------|---------|------|
| 安全 IO | `_SafeWriter` | 防止断管道/stdout 关闭导致 agent 崩溃 |
| 迭代预算 | `IterationBudget` | 线程安全的迭代计数器, 防止无限循环 |
| 上下文压缩 | `_compress_context()` | 对话历史过长时自动压缩, 拆分 SQLite session |
| 工具执行 | `_execute_tool_calls()` | 工具批次的并发/顺序调度入口 |
| 单工具路由 | `_invoke_tool()` | 根据工具名分发到不同处理器 |
| 中断机制 | `interrupt()` / `is_interrupted()` | 用户中断当前 agent 循环 |
| Steer 机制 | `steer()` | 注入用户提示到工具结果中 (不中断 agent) |
| 子 Agent | `delegate_task` | 委托子任务给独立 AIAgent 实例 |
| 记忆提供商 | MemoryManager | 外部记忆插件 (Honcho, Mem0 等) |
| 提示缓存 | Anthropic prompt caching | 减少多轮对话输入成本 ~75% |

## 支持的 API 模式

- **chat_completions**: OpenAI 兼容端点 (OpenRouter, Ollama, vLLM 等)
- **anthropic_messages**: Anthropic Messages API (原生 + 第三方兼容)
- **codex_responses**: OpenAI Codex Responses API (GPT-5, xAI 等)
- **bedrock_converse**: AWS Bedrock Converse API

Features:
- Automatic tool calling loop until completion
- Configurable model parameters
- Error handling and recovery (credential pool, fallback model)
- Message history management with session persistence
- Support for multiple model providers (Anthropic, OpenAI, Bedrock, OpenRouter, etc.)
- Concurrent tool execution via thread pool
- Streaming token delivery with interrupt support
- Context compression for long-running sessions
- Sub-agent delegation (delegate_task)
- Plugin hook system (pre_llm_call, post_llm_call, on_session_start)

Usage:
    from run_agent import AIAgent

    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

import asyncio
import base64
import concurrent.futures
import copy
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import random
import re
import sys
import tempfile
import time
import threading
from types import SimpleNamespace
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home

# 从 ~/.hermes/.env 优先加载 .env 文件，然后回退到项目根目录。
# 用户管理的 env 文件应在重启时覆盖过时的 shell 导出环境变量。
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.timeouts import (
    get_provider_request_timeout,
    get_provider_stale_timeout,
)

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")


## 导入工具系统
from model_tools import (
    get_tool_definitions,
    get_toolset_for_tool,
    handle_function_call,
    check_toolset_requirements,
)
from tools.terminal_tool import cleanup_vm, get_active_env, is_persistent_env
from tools.tool_result_storage import maybe_persist_tool_result, enforce_turn_budget
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser


## Agent 内部逻辑提取到 agent/ 包中以提高模块化
from agent.memory_manager import build_memory_context_block, sanitize_context
from agent.retry_utils import jittered_backoff
from agent.error_classifier import classify_api_error, FailoverReason
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS,
    MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE,
    build_nous_subscription_prompt,
)
from agent.model_metadata import (
    fetch_model_metadata,
    estimate_tokens_rough, estimate_messages_tokens_rough, estimate_request_tokens_rough,
    get_next_probe_tier, parse_context_limit_from_error,
    parse_available_output_tokens_from_error,
    save_context_length, is_local_endpoint,
    query_ollama_num_ctx,
)
from agent.context_compressor import ContextCompressor
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.prompt_caching import apply_anthropic_cache_control
from agent.prompt_builder import build_skills_system_prompt, build_context_files_prompt, build_environment_hints, load_soul_md, TOOL_USE_ENFORCEMENT_GUIDANCE, TOOL_USE_ENFORCEMENT_MODELS, GOOGLE_MODEL_OPERATIONAL_GUIDANCE, OPENAI_MODEL_EXECUTION_GUIDANCE
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from agent.codex_responses_adapter import (
    _derive_responses_function_call_id as _codex_derive_responses_function_call_id,
    _deterministic_call_id as _codex_deterministic_call_id,
    _split_responses_tool_id as _codex_split_responses_tool_id,
    _summarize_user_message_for_log,
)
from agent.display import (
    KawaiiSpinner, build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
    _detect_tool_failure,
    get_tool_emoji as _get_tool_emoji,
)
from agent.trajectory import (
    convert_scratchpad_to_think, has_incomplete_scratchpad,
    save_trajectory as _save_trajectory_to_file,
)
from utils import atomic_json_write, base_url_host_matches, base_url_hostname, env_var_enabled, normalize_proxy_url



class _SafeWriter:
    """安全 IO 包装器 — 静默捕获断管道错误, 防止 agent 崩溃.

    当 hermes-agent 作为 systemd 服务、Docker 容器或 headless 守护进程运行时,
    stdout/stderr 管道可能因空闲超时、缓冲区耗尽或 socket 重置而不可用。
    任何 print() 调用将抛出 ``OSError: [Errno 5] Input/output error``,
    这会崩溃 agent 的初始化或 run_conversation(), 尤其当 except 处理程序
    也尝试打印时会发生双重错误。

    此外, 子 agent 在 ThreadPoolExecutor 线程中运行时, 共享的 stdout 句柄
    可能在线程销毁和清理之间关闭, 引发 ``ValueError: I/O operation on closed file``。

    本包装器将所有写入委托给底层流, 并静默捕获 OSError 和 ValueError。
    当包装的流正常时完全透明。"""

    __slots__ = ("_inner",)

    def __init__(self, inner):
        """包装底层流对象 — 存储内部流引用。"""
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        """写入数据 — 静默捕获 OSError/ValueError, 防止断管道崩溃。"""
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        """刷新底层流 — 静默捕获 OSError/ValueError。"""
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        """返回底层流的文件描述符。"""
        return self._inner.fileno()

    def isatty(self):
        """检查底层流是否为 TTY — 失败时返回 False 而非崩溃。"""
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        """代理未知属性的访问到内部流。"""
        return getattr(self._inner, name)


def _get_proxy_from_env() -> Optional[str]:
    """从环境变量读取代理 URL — 支持 HTTPS_PROXY/HTTP_PROXY/ALL_PROXY。

    [中文] 按优先级检查代理环境变量, 返回第一个有效的。忽略大小写变体。
    返回 None 表示未配置代理。"""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = os.environ.get(key, "").strip()
        if value:
            return normalize_proxy_url(value)
    return None


def _install_safe_stdio() -> None:
    """安装安全 IO 包装器 — 防止断管道 stdout/stderr 导致 agent 崩溃。

    [中文] 在 systemd/Docker/headless 守护进程环境下, stdout/stderr 可能不可用。
    本函数将两者包装为 _SafeWriter, 使 print() 失败时静默处理而非崩溃。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


class IterationBudget:
    """线程安全的迭代计数器 — 防止 agent 无限循环.

    每个 agent (父 agent 或子 agent) 拥有独立的 ``IterationBudget``。
    父 agent 的预算上限为 ``max_iterations`` (默认 90)。
    每个子 agent 拥有独立预算, 上限为 ``delegation.max_iterations`` (默认 50)。
    这意味着父 agent + 子 agent 的总迭代次数可能超过父 agent 的上限。
    用户通过 config.yaml 中的 ``delegation.max_iterations`` 控制每个子 agent 的限制。

    ``execute_code`` (编程式工具调用) 的迭代通过 :meth:`refund` 退还,
    因此不消耗预算。这是为了鼓励模型在执行代码时进行多轮调试。"""

    def __init__(self, max_total: int):
        """初始化迭代预算 — max_total 为最大允许迭代次数。"""
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """消费一次迭代 — 返回 True 表示允许继续, False 表示预算耗尽。"""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """退还一次迭代 — 用于 execute_code 等不应消耗预算的编程式调用。"""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        """已消费的迭代次数。"""
        return self._used

    @property
    def remaining(self) -> int:
        """剩余可用迭代次数 — 线程安全, 返回 max(0, max_total - _used)。"""
        with self._lock:
            return max(0, self.max_total - self._used)


# ═══════════════════════════════════════════════════════════════════════════════
# 工具并行执行策略 — 定义哪些工具可以安全并发运行
# ═══════════════════════════════════════════════════════════════════════════════

# 禁止并发运行的工具 (交互式/面向用户的工具)。
# 当批次中出现这些工具时, 回退到顺序执行。
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# 只读工具 — 无共享可变会话状态, 始终安全并发。
_PARALLEL_SAFE_TOOLS = frozenset({
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
})

## 文件工具在操作独立路径时可以并发运行。
_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

## 并行工具执行的最大并发工作线程数。
_MAX_TOOL_WORKERS = 8

## 匹配终端命令中可能修改/删除文件的模式。
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
## 覆盖文件的输出重定向（> 而非 >>）
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def _is_destructive_command(cmd: str) -> bool:
    """启发式判断终端命令是否可能修改/删除文件 — Heuristic check for destructive commands.

    [中文] 检测命令中是否包含 rm, mv, sed -i, git reset/clean/checkout 等破坏性操作,
    以及 > 重定向覆盖。用于并发工具执行时的安全检查。"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """判断工具批次是否可以安全并发执行 — Check if a tool-call batch is safe for parallel execution.

    [中文] 检查规则:
      1. 单个工具 → 不并发
      2. 包含交互式工具 (clarify) → 不并发
      3. 文件工具 → 检查目标路径不重叠才并发
      4. 其他工具 → 必须在 _PARALLEL_SAFE_TOOLS 白名单中
      5. 参数解析失败 → 保守回退到顺序执行"""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """提取文件工具的操作目标路径 — Extract the normalized file target for path-scoped tools.

    [中文] 从工具调用参数中提取 'path' 字段, 展开 ~ 和相对路径, 返回绝对 Path。
    文件可能尚未存在, 因此不调用 resolve()。"""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """判断两个路径是否指向同一子树 — Check if two paths may refer to the same subtree.

    [中文] 通过比较路径 parts 的前缀是否相同来判断路径重叠。
    用于并发执行时检测文件工具之间的冲突。"""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]



_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')




def _sanitize_surrogates(text: str) -> str:
    """清理字符串中的孤立 surrogate 字符 — Replace lone surrogate code points with U+FFFD.

    [中文] Surrogate 在 UTF-8 中无效, 会导致 OpenAI SDK 的 json.dumps() 崩溃。
    无 surrogate 时快速跳过 (几乎零开销)。"""
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('�', text)
    return text
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


## _summarize_user_message_for_log 从 agent.codex_responses_adapter 导入
##（参见上方导入块）。保留从 run_agent 导入以向后兼容。


def _sanitize_structure_surrogates(payload: Any) -> bool:
    """递归清理嵌套字典/列表中的 surrogate 字符 — Recursively replace surrogate code points in-place.

    [中文] 与 _sanitize_messages_surrogates 配合使用, 处理扁平字段检查无法触及的
    嵌套结构化字段 (如 reasoning_details 数组)。返回 True 表示有 surrogate 被替换。"""
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[key] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    if _SURROGATE_RE.search(value):
                        node[idx] = _SURROGATE_RE.sub('\ufffd', value)
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found


def _sanitize_messages_surrogates(messages: list) -> bool:
    """清理消息列表中所有字符串内容的 surrogate 字符 — Sanitize surrogates from all message fields.

    [中文] 原地遍历消息字典, 检查 content/text/name/tool_calls/reasoning_content 等字段。
    字节级推理模型 (xiaomi/mimo, kimi, glm) 可能在推理输出中产生孤立 surrogate,
    这些 surrogate 会流入下一轮 api_messages["reasoning_content"], 导致 SDK json.dumps() 崩溃。
    返回 True 表示有 surrogate 被替换。"""
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
        name = msg.get("name")
        if isinstance(name, str) and _SURROGATE_RE.search(name):
            msg["name"] = _SURROGATE_RE.sub('\ufffd', name)
            found = True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and _SURROGATE_RE.search(tc_id):
                    tc["id"] = _SURROGATE_RE.sub('\ufffd', tc_id)
                    found = True
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and _SURROGATE_RE.search(fn_name):
                        fn["name"] = _SURROGATE_RE.sub('\ufffd', fn_name)
                        found = True
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str) and _SURROGATE_RE.search(fn_args):
                        fn["arguments"] = _SURROGATE_RE.sub('\ufffd', fn_args)
                        found = True
        # 遍历额外的字符串/嵌套字段（reasoning、reasoning_content、
        # reasoning_details 等）——字节级推理模型（xiaomi/mimo、kimi、glm）
        # 可能在这些字段中留下 surrogate 字符，上述逐字段检查无法覆盖。
        # 匹配 _sanitize_messages_non_ascii 的覆盖范围（PR #10537）。
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                if _SURROGATE_RE.search(value):
                    msg[key] = _SURROGATE_RE.sub('\ufffd', value)
                    found = True
            elif isinstance(value, (dict, list)):
                if _sanitize_structure_surrogates(value):
                    found = True
    return found


def _repair_tool_call_arguments(raw_args: str, tool_name: str = "?") -> str:
    """尝试修复格式错误的工具调用参数 JSON — Repair malformed tool_call argument JSON.

    [中文] GLM-5.1 等模型通过 Ollama 可能产生截断的 JSON、尾部逗号、Python None 等。
    修复步骤: 去尾部逗号 → 补全未闭合的大括号/方括号 → 去除多余的闭合括号。
    全部失败则返回 "{}", 避免整个会话崩溃。"""
    raw_stripped = raw_args.strip() if isinstance(raw_args, str) else ""

    # 快速路径：空/纯空白 → 空对象
    if not raw_stripped:
        logger.warning("Sanitized empty tool_call arguments for %s", tool_name)
        return "{}"

    # Python 字面量 None → 标准化为 {}
    if raw_stripped == "None":
        logger.warning("Sanitized Python-None tool_call arguments for %s", tool_name)
        return "{}"

    # 尝试常见 JSON 修复
    fixed = raw_stripped
    # 1. 去除 } 或 ] 前的尾部逗号
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # 2. 补全未闭合的结构
    open_curly = fixed.count('{') - fixed.count('}')
    open_bracket = fixed.count('[') - fixed.count(']')
    if open_curly > 0:
        fixed += '}' * open_curly
    if open_bracket > 0:
        fixed += ']' * open_bracket
    # 3. 移除多余的闭合括号（最多 50 次迭代）
    for _ in range(50):
        try:
            json.loads(fixed)
            break
        except json.JSONDecodeError:
            if fixed.endswith('}') and fixed.count('}') > fixed.count('{'):
                fixed = fixed[:-1]
            elif fixed.endswith(']') and fixed.count(']') > fixed.count('['):
                fixed = fixed[:-1]
            else:
                break

    try:
        json.loads(fixed)
        logger.warning(
            "Repaired malformed tool_call arguments for %s: %s → %s",
            tool_name, raw_stripped[:80], fixed[:80],
        )
        return fixed
    except json.JSONDecodeError:
        pass

    # 最后手段：替换为空对象，避免 API 请求
    # 导致整个会话崩溃。
    logger.warning(
        "Unrepairable tool_call arguments for %s — "
        "replaced with empty object (was: %s)",
        tool_name, raw_stripped[:80],
    )
    return "{}"


def _strip_non_ascii(text: str) -> str:
    """移除所有非 ASCII 字符 — Last-resort ASCII-only encoding fallback.

    [中文] 用于 LANG=C / Chromebook 等 ASCII-only 系统编码的极端场景。
    移除全部非 ASCII 字符, 保证字符串可通过 ASCII 编码。"""
    return text.encode('ascii', errors='ignore').decode('ascii')


def _sanitize_messages_non_ascii(messages: list) -> bool:
    """清理消息列表中所有字符串的非 ASCII 字符 — Strip non-ASCII from all message fields.

    [中文] ASCII-only 系统的最后手段恢复机制 (LANG=C, Chromebook, 最小容器)。
    返回 True 表示有非 ASCII 内容被清理。"""
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # 清理 content（字符串）
        content = msg.get("content")
        if isinstance(content, str):
            sanitized = _strip_non_ascii(content)
            if sanitized != content:
                msg["content"] = sanitized
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        sanitized = _strip_non_ascii(text)
                        if sanitized != text:
                            part["text"] = sanitized
                            found = True
        # 清理 name 字段（工具结果中可能含非 ASCII 字符）
        name = msg.get("name")
        if isinstance(name, str):
            sanitized = _strip_non_ascii(name)
            if sanitized != name:
                msg["name"] = sanitized
                found = True
        # 清理 tool_calls
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        fn_args = fn.get("arguments")
                        if isinstance(fn_args, str):
                            sanitized = _strip_non_ascii(fn_args)
                            if sanitized != fn_args:
                                fn["arguments"] = sanitized
                                found = True
        # 清理额外的顶级字符串字段（如 reasoning_content）
        for key, value in msg.items():
            if key in {"content", "name", "tool_calls", "role"}:
                continue
            if isinstance(value, str):
                sanitized = _strip_non_ascii(value)
                if sanitized != value:
                    msg[key] = sanitized
                    found = True
    return found


def _sanitize_tools_non_ascii(tools: list) -> bool:
    """清理工具定义中的非 ASCII 字符 — Strip non-ASCII from tool payloads in-place.

    [中文] 原地清理, 确保工具 schema 与 ASCII-only 系统兼容。"""
    return _sanitize_structure_non_ascii(tools)


def _sanitize_structure_non_ascii(payload: Any) -> bool:
    """递归清理嵌套结构中的非 ASCII 字符 — Recursively strip non-ASCII from nested dict/list.

    [中文] 深度优先遍历, 同 _sanitize_structure_surrogates 镜像实现。
    返回 True 表示有非 ASCII 内容被清理。"""
    found = False

    def _walk(node):
        nonlocal found
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[key] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                if isinstance(value, str):
                    sanitized = _strip_non_ascii(value)
                    if sanitized != value:
                        node[idx] = sanitized
                        found = True
                elif isinstance(value, (dict, list)):
                    _walk(value)

    _walk(payload)
    return found





# =========================================================================
# 大工具结果处理器 — 将超大输出保存到临时文件
# =========================================================================


# =========================================================================
# Qwen Portal 请求头 — 模拟 QwenCode CLI，兼容 portal.qwen.ai。
# 提取为模块级辅助函数，供 __init__ 和
# _apply_client_headers_for_base_url 共用。
# =========================================================================
_QWEN_CODE_VERSION = "0.14.1"


def _qwen_portal_headers() -> dict:
    """构建 Qwen Portal API 请求所需的 HTTP 头 — Return required HTTP headers for Qwen Portal.

    [中文] 模拟 QwenCode CLI 的 User-Agent 签名, 包含 DashScope 缓存控制和 OAuth 认证类型。"""
    import platform as _plat

    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }


class AIAgent:
    """
    Hermes AI Agent 主类 — 具备完整工具调用能力的 AI 智能体.

    本类是 Hermes Agent 的**核心控制器**, 负责管理整个对话生命周期:

    **六大核心职责:**
    1. **系统提示词管理** — 按层级组装系统提示词 (身份/记忆/技能/上下文/环境)
    2. **LLM API 调用** — 支持 4 种 API 模式 (chat_completions / anthropic_messages / codex_responses / bedrock_converse)
    3. **工具调用执行** — 自动并发/顺序调度工具批次, 最多 8 个并发线程
    4. **上下文压缩** — 对话历史过长时自动压缩, 拆分 SQLite 会话
    5. **会话持久化** — SQLite + FTS5 全文搜索, JSON session log
    6. **中断/Steer/委托** — 用户中断、注入提示、子 Agent 任务委托

    **关键生命周期:**
    - ``__init__()`` → 配置 + 客户端初始化
    - ``run_conversation()`` → 对话主循环 (可多次调用)
    - ``close()`` → 释放资源

    **线程模型:**
    - 主线程: run_conversation 主循环
    - API 调用线程: 流式 API 在独立线程运行, 主线程 0.3s 轮询中断
    - 工具执行线程: ThreadPoolExecutor (最多 8 个 worker)
    - 子 Agent 线程: delegate_task 创建独立 AIAgent 实例
    """

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""
        self._base_url_hostname = base_url_hostname(value)

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        acp_command: str = None,
        acp_args: list[str] | None = None,
        command: str = None,
        args: list[str] | None = None,
        model: str = "",
        max_iterations: int = 90,  # Default tool-calling iterations (shared with subagents)
        tool_delay: float = 1.0,
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
        provider_require_parameters: bool = False,
        provider_data_collection: str = None,
        session_id: str = None,
        tool_progress_callback: callable = None,
        tool_start_callback: callable = None,
        tool_complete_callback: callable = None,
        thinking_callback: callable = None,
        reasoning_callback: callable = None,
        clarify_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None,
        status_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        service_tier: str = None,
        request_overrides: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        user_id: str = None,
        user_name: str = None,
        chat_id: str = None,
        chat_name: str = None,
        chat_type: str = None,
        thread_id: str = None,
        gateway_session_key: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db=None,
        parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None,
        fallback_model: Dict[str, Any] = None,
        credential_pool=None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 50,
        pass_session_id: bool = False,
        persist_session: bool = True,
    ):
        """
        初始化 AI Agent — 解析全部配置, 建立客户端连接, 加载工具和记忆系统.

        Args:
            base_url: 模型 API 的基础 URL (可选)
            api_key: API 认证密钥 (可选, 未提供时从环境变量读取)
            provider: 提供商标识符 (用于遥测/路由提示)
            api_mode: API 模式覆盖: "chat_completions" / "codex_responses" / "anthropic_messages" / "bedrock_converse"
            model: 模型名称 (默认: "anthropic/claude-opus-4.6")
            max_iterations: 最大工具调用迭代次数 (默认 90)
            tool_delay: 工具调用间延迟 (秒, 默认 1.0)
            enabled_toolsets: 仅启用这些工具集 (可选)
            disabled_toolsets: 禁用这些工具集 (可选)
            save_trajectories: 是否保存对话轨迹到 JSONL 文件
            verbose_logging: 启用详细调试日志
            quiet_mode: 静默模式 — 抑制进度输出
            ephemeral_system_prompt: 临时系统提示 (仅 API 调用时注入, 不持久化)
            log_prefix_chars: 日志预览中工具调用/响应的字符数
            log_prefix: 并行处理中标识消息的日志前缀
            session_id: 预生成的会话 ID (可选, 未提供时自动生成)
            clarify_callback: 交互式澄清工具的回调函数(问题, 选项) → str
            prefill_messages: 预填充对话历史 (few-shot priming, 不持久化)
            platform: 用户界面平台 (如 "cli", "telegram", "discord", "whatsapp")
            skip_context_files: True 时跳过 SOUL.md/AGENTS.md/.cursorrules 注入
            credential_pool: 凭证池 — API key 耗尽时自动切换
            pass_session_id: True 时在系统提示中暴露 session_id
            persist_session: True 时持久化会话到 SQLite
        """
        _install_safe_stdio()

        # ═══════════════════════════════════════════════════════════════
        # [初始化阶段 1/5] 基础属性存储 — 模型、预算、回调
        # ═══════════════════════════════════════════════════════════════

        self.model = model
        self.max_iterations = max_iterations
        # 共享迭代预算 — 父 agent 创建, 子 agent 继承。
        # 每次 LLM 轮次由父 agent + 所有子 agent 共同消耗。
        self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform  # "cli", "telegram", "discord", "whatsapp", etc.
        self._user_id = user_id  # Platform user identifier (gateway sessions)
        self._user_name = user_name
        self._chat_id = chat_id
        self._chat_name = chat_name
        self._chat_type = chat_type
        self._thread_id = thread_id
        self._gateway_session_key = gateway_session_key  # Stable per-chat key (e.g. agent:main:telegram:dm:123)
        # 可插拔的 print 函数 — CLI 会替换为 _cprint，以便
        # 原始的 ANSI 状态行通过 prompt_toolkit 的渲染器路由，
        # 而不是直接输出到 stdout——patch_stdout 的 StdoutProxy
        # 会破坏转义序列。None = 使用内置 print。
        self._print_fn = None
        self.background_review_callback = None  # Optional sync callback for gateway delivery
        self.skip_context_files = skip_context_files
        self.pass_session_id = pass_session_id
        self.persist_session = persist_session
        self._credential_pool = credential_pool
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""

        # ═══════════════════════════════════════════════════════════════
        # [初始化阶段 2/5] API 模式自动检测 — 根据 provider/URL 选择协议
        # ═══════════════════════════════════════════════════════════════
        # 四种 API 模式:
        #   chat_completions  — OpenAI 兼容端点 (OpenRouter, Ollama, vLLM)
        #   anthropic_messages — Anthropic Messages API (原生 + DashScope/MiniMax 兼容)
        #   codex_responses   — OpenAI Codex Responses API (GPT-5, xAI)
        #   bedrock_converse  — AWS Bedrock Converse API
        self.base_url = base_url or ""
        provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        self.provider = provider_name or ""
        self.acp_command = acp_command or command
        self.acp_args = list(acp_args or args or [])
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif self.provider == "xai":
            self.api_mode = "codex_responses"
        elif (provider_name is None) and (
            self._base_url_hostname == "chatgpt.com"
            and "/backend-api/codex" in self._base_url_lower
        ):
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif (provider_name is None) and self._base_url_hostname == "api.x.ai":
            self.api_mode = "codex_responses"
            self.provider = "xai"
        elif self.provider == "anthropic" or (provider_name is None and self._base_url_hostname == "api.anthropic.com"):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        elif self._base_url_lower.rstrip("/").endswith("/anthropic"):
            # 第三方兼容 Anthropic 的端点（如 MiniMax、DashScope）
            # 使用以 /anthropic 结尾的 URL 约定。自动检测这些端点，以便
            # 使用 Anthropic Messages API 适配器而非 chat completions。
            self.api_mode = "anthropic_messages"
        elif self.provider == "bedrock" or (
            self._base_url_hostname.startswith("bedrock-runtime.")
            and base_url_host_matches(self._base_url_lower, "amazonaws.com")
        ):
            # AWS Bedrock — 从提供商名称或 base URL 自动检测
            # (bedrock-runtime.<region>.amazonaws.com)。
            self.api_mode = "bedrock_converse"
        else:
            self.api_mode = "chat_completions"

        # 预先预热传输缓存，使导入错误在初始化时就能暴露出来，
        # 而非在对话中途才发现。同时验证 api_mode 已注册。
        try:
            self._get_transport()
        except Exception:
            pass  # Non-fatal — transport may not exist for all modes yet

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS,
                normalize_model_for_provider,
            )

            if self.provider not in _AGGREGATOR_PROVIDERS:
                self.model = normalize_model_for_provider(self.model, self.provider)
        except Exception:
            pass

        # GPT-5.x 模型通常需要 Responses API 路径，但部分
        # 提供商存在例外（例如 Copilot 的 gpt-5-mini 仍使用
        # chat completions）。对直连 OpenAI URL 也会自动升级，
        # 因为所有较新的工具调用模型更偏好 Responses API。
        # ACP 运行时除外：CopilotACPClient
        # 自行处理路由，不实现 Responses API
        # 接口。
        # 当明确指定了 api_mode 时，遵循用户选择——用户
        # 最清楚自己的端点支持什么（#10473）。
        if (
            api_mode is None
            and self.api_mode == "chat_completions"
            and self.provider != "copilot-acp"
            and not str(self.base_url or "").lower().startswith("acp://copilot")
            and not str(self.base_url or "").lower().startswith("acp+tcp://")
            and (
                self._is_direct_openai_url()
                or self._provider_model_requires_responses_api(
                    self.model,
                    provider=self.provider,
                )
            )
        ):
            self.api_mode = "codex_responses"
            # 使预热的传输缓存失效——api_mode 已从
            # __init__ 预热后由 chat_completions 变更为 codex_responses。
            if hasattr(self, "_transport_cache"):
                self._transport_cache.clear()

        # 在后台线程中预热 OpenRouter 模型元数据缓存。
        # fetch_model_metadata() 缓存 1 小时；避免在首次 API 响应
        # 估算价格时发出阻塞的 HTTP 请求。
        if self.provider == "openrouter" or self._is_openrouter_url():
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.suppress_status_output = False
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self.interim_assistant_callback = interim_assistant_callback
        self.status_callback = status_callback
        self.tool_gen_callback = tool_gen_callback

        
        # 工具执行状态——允许在工具执行期间通过 _vprint 输出
        # 即使已注册了流消费者（此时没有 token 流传输）
        self._executing_tools = False

        # ═══════════════════════════════════════════════════════════════
        # [初始化阶段 3/5] 中断/Steer/子Agent 状态初始化
        # ═══════════════════════════════════════════════════════════════
        # Interrupt 机制 — 用户发送消息时立即中断 agent
        self._interrupt_requested = False
        self._interrupt_message = None  # 触发中断的可选消息
        self._execution_thread_id: int | None = None  # 在 run_conversation() 开始时设置
        self._interrupt_thread_signal_pending = False
        self._client_lock = threading.RLock()

        # /steer 机制 — 注入用户提示到工具结果中 (不中断 agent)
        # 与 interrupt() 不同, steer() 不设置 _interrupt_requested；
        # 它等待当前工具批次自然完成, 然后 drain hook 将文本
        # 追加到最后一条工具结果的 content 中, 使模型在下次
        # 迭代时看到提示。通过修改现有 tool 消息来保持角色交替,
        # 而不是插入新的 user 轮次。
        self._pending_steer: Optional[str] = None
        self._pending_steer_lock = threading.Lock()

        # 并发工具工作线程追踪。`_execute_tool_calls_concurrent`
        # 在独立的 ThreadPoolExecutor 工作线程中运行每个工具——这些工作线程
        # 的 tid 与 `_execution_thread_id` 不同，因此
        # 仅靠 `_set_interrupt(True, _execution_thread_id)` 不会使
        # 工作线程内的 `is_interrupted()` 返回 True。在此追踪
        # 工作线程，以便 `interrupt()` / `clear_interrupt()` 可以分发到
        # 这些 tid。
        self._tool_worker_threads: set[int] = set()
        self._tool_worker_threads_lock = threading.Lock()
        
        # 子 Agent 委托状态
        self._delegate_depth = 0        # 0 = top-level agent, incremented for children
        self._active_children = []      # Running child AIAgents (for interrupt propagation)
        self._active_children_lock = threading.Lock()
        
        # 存储 OpenRouter 提供商偏好
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.provider_require_parameters = provider_require_parameters
        self.provider_data_collection = provider_data_collection

        # 存储工具集过滤选项
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # 模型响应配置
        self.max_tokens = max_tokens  # None = use model default
        self.reasoning_config = reasoning_config  # None = use default (medium for OpenRouter)
        self.service_tier = service_tier
        self.request_overrides = dict(request_overrides or {})
        self.prefill_messages = prefill_messages or []  # Prefilled conversation turns
        self._force_ascii_payload = False
        
        # Anthropic 提示缓存：对原生 Anthropic、OpenRouter 和支持
        # Anthropic 协议的第三方网关上的 Claude 模型自动启用
        # (``api_mode == 'anthropic_messages'``)。多轮对话
        # 可降低约 75% 的输入成本。使用 system_and_3 策略
        # (4 个缓存断点)。参见 ``_anthropic_prompt_cache_policy``
        # 了解布局与传输的决策。
        self._use_prompt_caching, self._use_native_cache_layout = (
            self._anthropic_prompt_cache_policy()
        )
        self._cache_ttl = "5m"  # Default 5-minute TTL (1.25x write cost)
        
        # 迭代预算：仅当 LLM 实际耗尽迭代预算时
        # (api_call_count >= max_iterations) 才通知它。届时
        # 注入一条消息，允许最后一次 API 调用，如果模型
        # 仍未生成文本响应，则强制发送用户消息要求其总结。
        # 不发送中间压力警告——这曾导致模型在复杂任务上
        # 过早"放弃"（#7915）。
        self._budget_exhausted_injected = False
        self._budget_grace_call = False

        # 活动追踪——每次 API 调用、工具执行和流数据块时更新。
        # 网关超时处理器用它报告 Agent 被杀时正在执行的操作，
        # 以及"仍在工作中"通知也用它展示进度。
        
        self._last_activity_ts: float = time.time()
        self._last_activity_desc: str = "initializing"
        self._current_tool: str | None = None
        self._api_call_count: int = 0

        # 速率限制追踪——每次 API 调用后从 x-ratelimit-* 响应头更新。
        # 由 /usage 斜杠命令访问。
        self._rate_limit_state: Optional["RateLimitState"] = None

        # 集中日志——agent.log (INFO+) 和 errors.log (WARNING+)
        # 均存放在 ~/.hermes/logs/ 下。幂等设计，网关模式
        #（每条消息创建新 AIAgent）不会重复注册处理器。
        from hermes_logging import setup_logging, setup_verbose_logging
        setup_logging(hermes_home=_hermes_home)

        if self.verbose_logging:
            setup_verbose_logging()
            logger.info("Verbose logging enabled (third-party library logs suppressed)")
        else:
            if self.quiet_mode:
                # 静默模式下（CLI 默认），抑制所有工具/基础设施日志
                # 在*控制台*的输出。TUI 有自己的 rich 显示用于
                # 状态展示；logger INFO/WARNING 消息只会造成干扰。
                # 文件处理器（agent.log、errors.log）仍然捕获全部内容。
                for quiet_logger in [
                    'tools',               # all tools.* (terminal, browser, web, file, etc.)
                    'run_agent',            # agent runner internals
                    'trajectory_compressor',
                    'cron',                 # scheduler (only relevant in daemon mode)
                    'hermes_cli',           # CLI helpers
                ]:
                    logging.getLogger(quiet_logger).setLevel(logging.ERROR)
        
        # 内部流回调（流式 TTS 期间设置）。
        # 在此初始化，使得 _vprint 在 run_conversation 之前即可引用。
        self._stream_callback = None
        # 延迟段落换行标志——工具迭代后设置，以便在
        # 下一个真正的文本增量前添加单个 "\n\n"。
        self._stream_needs_break = False
        # 通过实时 token 回调已传递的可见 assistant 文本，
        # 用于避免当提供商后续将其作为完整的中间 assistant 消息
        # 返回时重复发送相同的评论文本。
        
        self._current_streamed_assistant_text = ""

        # 当前轮次可选的用户消息覆盖——当面向 API 的用户消息
        # 有意与持久化会话记录不同时使用
        #（如 CLI 语音模式仅为实时通话添加临时前缀）。
        self._persist_user_message_idx = None
        self._persist_user_message_override = None

        # 按图片负载/URL 缓存 Anthropic 图片转文字的回退结果，
        # 避免同一工具循环在相同图片历史上反复运行辅助视觉识别。
        
        self._anthropic_image_fallback_cache: Dict[str, str] = {}

        # 通过集中式提供商路由器初始化 LLM 客户端。
        # 路由器处理认证解析、base URL、header 以及
        # 所有已知提供商的 Codex/Anthropic 封装。
        # raw_codex=True，因为主 Agent 需要直接访问 responses.stream()
        # 以实现 Codex Responses API 流式传输。
        self._anthropic_client = None
        self._is_anthropic_oauth = False

        # 提前统一解析每个提供商/每个模型的请求超时时间，
        # 使下方所有客户端构建路径（Anthropic 原生、OpenAI 协议、
        # 基于路由器的隐式认证）能够一致应用。Bedrock
        # Claude 使用独立的超时路径，不在此处理。
        _provider_timeout = get_provider_request_timeout(self.provider, self.model)

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            # Bedrock + Claude → use AnthropicBedrock SDK for full feature parity
            #（提示缓存、思考预算、自适应思考）。
            _is_bedrock_anthropic = self.provider == "bedrock"
            if _is_bedrock_anthropic:
                from agent.anthropic_adapter import build_anthropic_bedrock_client
                _region_match = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
                _br_region = _region_match.group(1) if _region_match else "us-east-1"
                self._bedrock_region = _br_region
                self._anthropic_client = build_anthropic_bedrock_client(_br_region)
                self._anthropic_api_key = "aws-sdk"
                self._anthropic_base_url = base_url
                self._is_anthropic_oauth = False
                self.api_key = "aws-sdk"
                self.client = None
                self._client_kwargs = {}
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model} (AWS Bedrock + AnthropicBedrock SDK, {_br_region})")
            else:
                # 仅当提供商确实是 Anthropic 时才回退到 ANTHROPIC_TOKEN。
                # 其他 anthropic_messages 提供商（MiniMax、Alibaba 等）必须使用各自的 API key。
                # 回退会将 Anthropic 凭据发送到第三方端点（修复 #1739、#minimax-401）。
                _is_native_anthropic = self.provider == "anthropic"
                effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = base_url
                # 仅当 token 确实属于原生 Anthropic 时才将会话标记为 OAuth 认证。
                # 第三方提供商
                #（MiniMax、Kimi、GLM、LiteLLM 代理）虽接受
                # Anthropic 协议，但绝不能触发 OAuth 代码路径——否则
                # 会注入 Claude-Code 身份 header 和 system prompt，
                # 导致其端点返回 401/403。守护 #1739 和
                # 第三方身份注入漏洞。
                from agent.anthropic_adapter import _is_oauth_token as _is_oat
                self._is_anthropic_oauth = _is_oat(effective_key) if _is_native_anthropic else False
                self._anthropic_client = build_anthropic_client(effective_key, base_url, timeout=_provider_timeout)
                # Anthropic 模式下不需要 OpenAI 客户端
                self.client = None
                self._client_kwargs = {}
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)")
                    if effective_key and len(effective_key) > 12:
                        print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        elif self.api_mode == "bedrock_converse":
            # AWS Bedrock — 直接使用 boto3，不需要 OpenAI 客户端。
            # 区域从 base_url 中提取，默认为 us-east-1。
            _region_match = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
            self._bedrock_region = _region_match.group(1) if _region_match else "us-east-1"
            # 护栏配置——在初始化时从 config.yaml 读取。
            self._bedrock_guardrail_config = None
            try:
                from hermes_cli.config import load_config as _load_br_cfg
                _gr = _load_br_cfg().get("bedrock", {}).get("guardrail", {})
                if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
                    self._bedrock_guardrail_config = {
                        "guardrailIdentifier": _gr["guardrail_identifier"],
                        "guardrailVersion": _gr["guardrail_version"],
                    }
                    if _gr.get("stream_processing_mode"):
                        self._bedrock_guardrail_config["streamProcessingMode"] = _gr["stream_processing_mode"]
                    if _gr.get("trace"):
                        self._bedrock_guardrail_config["trace"] = _gr["trace"]
            except Exception:
                pass
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                _gr_label = " + Guardrails" if self._bedrock_guardrail_config else ""
                print(f"🤖 AI Agent initialized with model: {self.model} (AWS Bedrock, {self._bedrock_region}{_gr_label})")
        else:
            if api_key and base_url:
                # 来自 CLI/网关的显式凭据——直接构建。
                # 运行时提供商解析器已为我们处理了认证。
                client_kwargs = {"api_key": api_key, "base_url": base_url}
                if _provider_timeout is not None:
                    client_kwargs["timeout"] = _provider_timeout
                if self.provider == "copilot-acp":
                    client_kwargs["command"] = self.acp_command
                    client_kwargs["args"] = self.acp_args
                effective_base = base_url
                if base_url_host_matches(effective_base, "openrouter.ai"):
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-OpenRouter-Title": "Hermes Agent",
                        "X-OpenRouter-Categories": "productivity,cli-agent",
                    }
                elif base_url_host_matches(effective_base, "api.githubcopilot.com"):
                    from hermes_cli.models import copilot_default_headers

                    client_kwargs["default_headers"] = copilot_default_headers()
                elif base_url_host_matches(effective_base, "api.kimi.com"):
                    client_kwargs["default_headers"] = {
                        "User-Agent": "claude-code/0.1.0",
                    }
                elif base_url_host_matches(effective_base, "portal.qwen.ai"):
                    client_kwargs["default_headers"] = _qwen_portal_headers()
                elif base_url_host_matches(effective_base, "chatgpt.com"):
                    from agent.auxiliary_client import _codex_cloudflare_headers
                    client_kwargs["default_headers"] = _codex_cloudflare_headers(api_key)
            else:
                # 无显式凭据——使用集中式提供商路由器
                from agent.auxiliary_client import resolve_provider_client
                _routed_client, _ = resolve_provider_client(
                    self.provider or "auto", model=self.model, raw_codex=True)
                if _routed_client is not None:
                    client_kwargs = {
                        "api_key": _routed_client.api_key,
                        "base_url": str(_routed_client.base_url),
                    }
                    if _provider_timeout is not None:
                        client_kwargs["timeout"] = _provider_timeout
                    # 保留路由器设置的 default_headers
                    if hasattr(_routed_client, '_default_headers') and _routed_client._default_headers:
                        client_kwargs["default_headers"] = dict(_routed_client._default_headers)
                else:
                    # 当用户明确选择了非 OpenRouter 提供商
                    # 但未找到凭据时，快速失败并给出清晰消息，
                    # 而非通过 OpenRouter 静默路由。
                    _explicit = (self.provider or "").strip().lower()
                    if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                        # 从提供商配置中查找实际的环境变量名称
                        # ——部分提供商使用非标准变量名
                        # (e.g. alibaba → DASHSCOPE_API_KEY, not ALIBABA_API_KEY).
                        _env_hint = f"{_explicit.upper()}_API_KEY"
                        try:
                            from hermes_cli.auth import PROVIDER_REGISTRY
                            _pcfg = PROVIDER_REGISTRY.get(_explicit)
                            if _pcfg and _pcfg.api_key_env_vars:
                                _env_hint = _pcfg.api_key_env_vars[0]
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_env_hint} environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )
                    # 未配置提供商——以清晰消息拒绝。
                    raise RuntimeError(
                        "No LLM provider configured. Run `hermes model` to "
                        "select a provider, or run `hermes setup` for first-time "
                        "configuration."
                    )
            
            self._client_kwargs = client_kwargs  # stored for rebuilding after interrupt

            # 为 OpenRouter 上的 Claude 启用细粒度工具流式传输。
            # 若不启用，Anthropic 会缓冲整个工具调用并在思考期间
            # 静默数分钟——OpenRouter 的上游代理在此期间因无数据
            # 而超时。beta header 使 Anthropic
            # 逐个 token 流式传输工具调用参数，保持
            # 连接活跃。
            _effective_base = str(client_kwargs.get("base_url", "")).lower()
            if base_url_host_matches(_effective_base, "openrouter.ai") and "claude" in (self.model or "").lower():
                headers = client_kwargs.get("default_headers") or {}
                existing_beta = headers.get("x-anthropic-beta", "")
                _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
                if _FINE_GRAINED not in existing_beta:
                    if existing_beta:
                        headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                    else:
                        headers["x-anthropic-beta"] = _FINE_GRAINED
                    client_kwargs["default_headers"] = headers

            self.api_key = client_kwargs.get("api_key", "")
            self.base_url = client_kwargs.get("base_url", self.base_url)
            try:
                self.client = self._create_openai_client(client_kwargs, reason="agent_init", shared=True)
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model}")
                    if base_url:
                        print(f"🔗 Using custom base URL: {base_url}")
                    # 始终显示 API key 信息（已遮盖）用于调试认证问题
                    key_used = client_kwargs.get("api_key", "none")
                    if key_used and key_used != "dummy-key" and len(key_used) > 12:
                        print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                    else:
                        print(f"⚠️  Warning: API key appears invalid or missing (got: '{key_used[:20] if key_used else 'none'}...')")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        # 提供商回退链——当主提供商耗尽时（限流、过载、
        # 连接故障）按序尝试的备用提供商列表。
        # 同时支持旧版单字典 ``fallback_model`` 和
        # 新版列表 ``fallback_providers`` 格式。
        if isinstance(fallback_model, list):
            self._fallback_chain = [
                f for f in fallback_model
                if isinstance(f, dict) and f.get("provider") and f.get("model")
            ]
        elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
            self._fallback_chain = [fallback_model]
        else:
            self._fallback_chain = []
        self._fallback_index = 0
        self._fallback_activated = False
        # 为向后兼容保留的旧版属性（测试、外部调用者）
        self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None
        if self._fallback_chain and not self.quiet_mode:
            if len(self._fallback_chain) == 1:
                fb = self._fallback_chain[0]
                print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
            else:
                print(f"🔄 Fallback chain ({len(self._fallback_chain)} providers): " +
                      " → ".join(f"{f['model']} ({f['provider']})" for f in self._fallback_chain))

        # 获取可用的过滤后工具
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=self.quiet_mode,
        )
        
        # 显示工具配置并存储有效工具名称用于验证
        self.valid_tool_names = set()
        if self.tools:
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
            if not self.quiet_mode:
                print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")
                
                # 如已应用过滤则显示过滤信息
                if enabled_toolsets:
                    print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
                if disabled_toolsets:
                    print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        elif not self.quiet_mode:
            print("🛠️  No tools loaded (all tools filtered out or unavailable)")
        
        # 检查工具依赖要求
        if self.tools and not self.quiet_mode:
            requirements = check_toolset_requirements()
            missing_reqs = [name for name, available in requirements.items() if not available]
            if missing_reqs:
                print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")
        
        # 显示轨迹保存状态
        if self.save_trajectories and not self.quiet_mode:
            print("📝 Trajectory saving enabled")
        
        # 显示临时 system prompt 状态
        if self.ephemeral_system_prompt and not self.quiet_mode:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")
        
        # 显示提示缓存状态
        if self._use_prompt_caching and not self.quiet_mode:
            if self._use_native_cache_layout and self.provider == "anthropic":
                source = "native Anthropic"
            elif self._use_native_cache_layout:
                source = "Anthropic-compatible endpoint"
            else:
                source = "Claude via OpenRouter"
            print(f"💾 Prompt caching: ENABLED ({source}, {self._cache_ttl} TTL)")
        
        # 会话日志设置——自动保存对话轨迹用于调试
        self.session_start = datetime.now()
        if session_id:
            # 使用提供的会话 ID（如来自 CLI）
            self.session_id = session_id
        else:
            # 生成新的会话 ID
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        
        # 会话日志存入 ~/.hermes/sessions/，与网关会话同目录
        hermes_home = get_hermes_home()
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
        
        # 追踪对话消息用于会话日志记录
        self._session_messages: List[Dict[str, Any]] = []
        
        # 缓存的 system prompt——每次会话构建一次，仅压缩时重建
        self._cached_system_prompt: Optional[str] = None
        
        # 文件系统检查点管理器（透明——非工具）
        from tools.checkpoint_manager import CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
        )
        
        # SQLite 会话存储（可选——由 CLI 或网关提供）
        self._session_db = session_db
        self._parent_session_id = parent_session_id
        self._last_flushed_db_idx = 0  # tracks DB-write cursor to prevent duplicate writes
        if self._session_db:
            try:
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_iterations,
                        "reasoning_config": reasoning_config,
                        "max_tokens": max_tokens,
                    },
                    user_id=None,
                    parent_session_id=self._parent_session_id,
                )
            except Exception as e:
                # 临时 SQLite 锁竞争（如 CLI 和网关并发写入）
                # 不得永久禁用此 Agent 的 session_search。
                # 保留 _session_db——一旦锁释放，后续的消息
                # 刷新和 session_search 调用仍然有效。
                # 本次运行的会话行可能在索引中缺失，
                # 但这是可恢复的（刷新时 upsert 行）。
                logger.warning(
                    "Session DB create_session failed (session_search still available): %s", e
                )
        
        # 内存中的 todo 列表，用于任务规划（每个 Agent/会话一个）
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()
        
        # 一次性加载配置，供记忆、技能和压缩模块使用
        try:
            from hermes_cli.config import load_config as _load_agent_config
            _agent_cfg = _load_agent_config()
        except Exception:
            _agent_cfg = {}
        # 仅缓存派生的辅助压缩上下文覆盖值，
        # 供后续启动可行性检查使用。避免在 Agent 实例上
        # 暴露广泛的伪公共配置对象。
        self._aux_compression_context_length_config = None

        # ═══════════════════════════════════════════════════════════════════════
        # [记忆系统] — 双层记忆架构：内置记忆 + 外部记忆提供商
        # ═══════════════════════════════════════════════════════════════════════
        #
        # Hermes Agent 的记忆系统分为两层，协同工作以确保跨会话的知识持久化：
        #
        # ┌─────────────────────────────────────────────────────────────────┐
        # │ 第一层：内置记忆（Built-in Memory）                               │
        # │   • 文件存储：MEMORY.md（Agent 知识）+ USER.md（用户画像）         │
        # │   • 条目分隔符：§（章节符号），支持多行条目                         │
        # │   • 字符限制：memory_char_limit（默认 2200）/ user_char_limit（1375）│
        # │   • 冻结快照模式：会话开始时将文件内容快照注入 system prompt          │
        # │   • 会话中途写入：立即更新磁盘文件，但不刷新 system prompt            │
        # │     （保护 Anthropic 前缀缓存，下次会话启动时自动加载新内容）         │
        # │   • 通过 memory 工具暴露给模型：add / replace / remove / read       │
        # │   • 安全扫描：注入 system prompt 前检测注入/外泄威胁                │
        # └─────────────────────────────────────────────────────────────────┘
        #                              │
        #                              ▼
        # ┌─────────────────────────────────────────────────────────────────┐
        # │ 第二层：外部记忆提供商（External Memory Provider）                  │
        # │   • 插件架构：plugins/memory/<name>/ 目录下                       │
        # │   • 一次只能激活一个外部提供商（防止工具冲突和 schema 膨胀）          │
        # │   • 提供商示例：Honcho（向量搜索）、Hindsight、Mem0 等              │
        # │   • 通过 MemoryManager 统一编排（agent/memory_manager.py）         │
        # │   • 生命周期：initialize → prefetch → sync → shutdown             │
        # │   • 工具 schema 自动注入到 Agent 工具表面                          │
        # │   • 桥接机制：内置 memory 写入自动转发到外部提供商                  │
        # │     （on_memory_write 钩子）                                      │
        # └─────────────────────────────────────────────────────────────────┘
        #
        # ┌─────────────────────────────────────────────────────────────────┐
        # │ 完整生命周期的记忆流转（run_conversation 中的 6 个阶段）             │
        # ├─────────────────────────────────────────────────────────────────┤
        # │                                                                  │
        # │  ① 会话启动 (__init__)                                           │
        # │     • 从磁盘加载 MEMORY.md / USER.md → self._memory_store        │
        # │     • 激活外部提供商 → MemoryManager.initialize_all()             │
        # │     • 记忆内容冻结为 system prompt 快照                           │
        # │                                                                  │
        # │  ② 每轮开始 (run_conversation 阶段四)                             │
        # │     • Nudge 检查：_turns_since_memory >= nudge_interval？         │
        # │       若是 → 本轮结束后触发后台记忆审查                            │
        # │     • MemoryManager.on_turn_start() — 通知提供商新轮次开始         │
        # │     • MemoryManager.prefetch_all() — 预取相关上下文                │
        # │       结果注入到 user message（包裹在 <memory-context> 标签中）     │
        # │                                                                  │
        # │  ③ 工具循环中 (Agent Loop)                                       │
        # │     • 模型调用 memory 工具 → _invoke_tool() 路由                   │
        # │     • 内置 memory 写入 → 磁盘立即更新 + on_memory_write 桥接       │
        # │     • 外部提供商工具 → MemoryManager.handle_tool_call()           │
        # │     • 写入成功后重置 _turns_since_memory = 0                     │
        # │                                                                  │
        # │  ④ 上下文压缩前 (_compress_context)                               │
        # │     • flush_memories(min_turns=0) — 强制保存记忆再丢弃上下文       │
        # │     • MemoryManager.on_pre_compress() — 通知提供商提取见解         │
        # │     • commit_memory_session() — 提交旧会话记忆                    │
        # │                                                                  │
        # │  ⑤ 每轮结束 (run_conversation 阶段六)                             │
        # │     • MemoryManager.sync_all() — 同步完成的轮次                  │
        # │     • MemoryManager.queue_prefetch_all() — 排队下次预取           │
        # │     • 若触发了 nudge → _spawn_background_review()                 │
        # │       （后台子 Agent 审查对话，自动调用 memory 工具保存发现）        │
        # │                                                                  │
        # │  ⑥ 会话结束 (close / /reset / 网关过期)                           │
        # │     • commit_memory_session() → MemoryManager.on_session_end()   │
        # │     • MemoryManager.shutdown_all() — 刷新队列，关闭连接            │
        # │     • 注意：不在每轮后调用 shutdown（多轮会话需要提供商持续运行）    │
        # │                                                                  │
        # └─────────────────────────────────────────────────────────────────┘
        #
        # 关键配置项（config.yaml memory 节）：
        #   memory_enabled:      是否启用 MEMORY.md
        #   user_profile_enabled: 是否启用 USER.md
        #   nudge_interval:      多少轮后提醒模型保存记忆（默认 10）
        #   flush_min_turns:     最少轮次后才允许压缩前刷新（默认 6）
        #   provider:            外部提供商名称（如 "honcho"、"hindsight"）
        #   memory_char_limit:   单条记忆字符上限（默认 2200）
        #   user_char_limit:     单条用户画像字符上限（默认 1375）
        #
        # 设计原则：
        #   1. 内存优先 — 记忆加载到 system prompt，利用前缀缓存减少成本
        #   2. 磁盘持久 — 写入立即落盘，崩溃不丢失
        #   3. 可选增强 — 外部提供商是附加层，内置记忆始终可用
        #   4. 隔离故障 — 任一提供商失败不影响其他提供商或主对话流程
        #   5. 后台审查 — nudge 触发后台子 Agent，不阻塞主对话
        # ═══════════════════════════════════════════════════════════════════════
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 10
        self._memory_flush_min_turns = 6
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        if not skip_memory:
            try:
                mem_config = _agent_cfg.get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                self._memory_flush_min_turns = int(mem_config.get("flush_min_turns", 6))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass  # Memory is optional -- don't break agent init
        


        # ── 外部记忆提供商插件 ──
        # MemoryManager 是双层记忆的编排中心（agent/memory_manager.py）：
        #   1. 内置提供商（BuiltinMemoryProvider）始终排在第一位，不可移除
        #   2. 外部提供商由 config.yaml 的 memory.provider 控制，一次最多一个
        #   3. 工具路由：MemoryManager 维护 tool_name → provider 映射，
        #      _invoke_tool() 通过 has_tool() / handle_tool_call() 分发调用
        #   4. 桥接写入：内置 memory 工具执行 add/replace 时，自动调用
        #      on_memory_write() 将内容镜像到外部提供商
        #   5. 生命周期钩子：on_session_end（会话级提取）、on_pre_compress
        #      （压缩前保存）、on_delegation（子Agent完成通知）
        self._memory_manager = None
        if not skip_memory:
            try:
                _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

                if _mem_provider_name:
                    from agent.memory_manager import MemoryManager as _MemoryManager
                    from plugins.memory import load_memory_provider as _load_mem
                    self._memory_manager = _MemoryManager()
                    _mp = _load_mem(_mem_provider_name)
                    if _mp and _mp.is_available():
                        self._memory_manager.add_provider(_mp)
                    if self._memory_manager.providers:
                        _init_kwargs = {
                            "session_id": self.session_id,
                            "platform": platform or "cli",
                            "hermes_home": str(get_hermes_home()),
                            "agent_context": "primary",
                        }
                        # 线程会话标题，用于记忆提供商的作用域限定
                        #（如 honcho 用此推导聊天作用域的会话 key）
                        if self._session_db:
                            try:
                                _st = self._session_db.get_session_title(self.session_id)
                                if _st:
                                    _init_kwargs["session_title"] = _st
                            except Exception:
                                pass
                        # 线程网关用户身份，用于按用户限定记忆作用域
                        if self._user_id:
                            _init_kwargs["user_id"] = self._user_id
                        if self._user_name:
                            _init_kwargs["user_name"] = self._user_name
                        if self._chat_id:
                            _init_kwargs["chat_id"] = self._chat_id
                        if self._chat_name:
                            _init_kwargs["chat_name"] = self._chat_name
                        if self._chat_type:
                            _init_kwargs["chat_type"] = self._chat_type
                        if self._thread_id:
                            _init_kwargs["thread_id"] = self._thread_id
                        # 线程网关会话 key，用于稳定的按聊天 Honcho 会话隔离
                        if self._gateway_session_key:
                            _init_kwargs["gateway_session_key"] = self._gateway_session_key
                        # 配置文件身份，用于按配置文件限定提供商作用域
                        try:
                            from hermes_cli.profiles import get_active_profile_name
                            _profile = get_active_profile_name()
                            _init_kwargs["agent_identity"] = _profile
                            _init_kwargs["agent_workspace"] = "hermes"
                        except Exception:
                            pass
                        self._memory_manager.initialize_all(**_init_kwargs)
                        logger.info("Memory provider '%s' activated", _mem_provider_name)
                    else:
                        logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                        self._memory_manager = None
            except Exception as _mpe:
                logger.warning("Memory provider plugin init failed: %s", _mpe)
                self._memory_manager = None

        # 将记忆提供商的工具 schema 注入工具表面。
        # 跳过名称已存在的工具（插件可能通过 ctx.register_tool()
        # 注册了相同工具，这些工具会出现在 self.tools 中
        # 通过 get_tool_definitions() 获取）。重复的函数名
        # 会在强制唯一名称的提供商上引发 400 错误
        #（如通过 Nous Portal 的 Xiaomi MiMo）。
        if self._memory_manager and self.tools is not None:
            _existing_tool_names = {
                t.get("function", {}).get("name")
                for t in self.tools
                if isinstance(t, dict)
            }
            for _schema in self._memory_manager.get_all_tool_schemas():
                _tname = _schema.get("name", "")
                if _tname and _tname in _existing_tool_names:
                    continue  # already registered via plugin path
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                if _tname:
                    self.valid_tool_names.add(_tname)
                    _existing_tool_names.add(_tname)

        # 技能配置：技能创建提醒的 nudge 间隔
        self._skill_nudge_interval = 10
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
        except Exception:
            pass

        # 工具使用强制配置："auto"（默认——匹配硬编码的
        # 模型列表）、true（始终）、false（从不）或子串模式列表。
        _agent_section = _agent_cfg.get("agent", {})
        if not isinstance(_agent_section, dict):
            _agent_section = {}
        self._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

        # 初始化上下文压缩器用于自动上下文管理
        # 在接近模型上下文限制时压缩对话
        # 通过 config.yaml 配置（compression 节）
        _compression_cfg = _agent_cfg.get("compression", {})
        if not isinstance(_compression_cfg, dict):
            _compression_cfg = {}
        compression_threshold = float(_compression_cfg.get("threshold", 0.50))
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in ("true", "1", "yes")
        compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))
        compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))

        # 读取可选的显式 context_length 覆盖值给辅助
        # 压缩模型。自定义端点通常无法通过 /models 报告此值，
        # 因此启动可行性检查需要这个配置提示。
        try:
            _aux_cfg = _agent_cfg.get("auxiliary", {}).get("compression", {})
        except Exception:
            _aux_cfg = {}
        if isinstance(_aux_cfg, dict):
            _aux_context_config = _aux_cfg.get("context_length")
        else:
            _aux_context_config = None
        if _aux_context_config is not None:
            try:
                _aux_context_config = int(_aux_context_config)
            except (TypeError, ValueError):
                _aux_context_config = None
        self._aux_compression_context_length_config = _aux_context_config

        # 从模型配置中读取显式的 context_length 覆盖值
        _model_cfg = _agent_cfg.get("model", {})
        if isinstance(_model_cfg, dict):
            _config_context_length = _model_cfg.get("context_length")
        else:
            _config_context_length = None
        if _config_context_length is not None:
            try:
                _config_context_length = int(_config_context_length)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid model.context_length in config.yaml: %r — "
                    "must be a plain integer (e.g. 256000, not '256K'). "
                    "Falling back to auto-detection.",
                    _config_context_length,
                )
                print(
                    f"\n⚠ Invalid model.context_length in config.yaml: {_config_context_length!r}\n"
                    f"  Must be a plain integer (e.g. 256000, not '256K').\n"
                    f"  Falling back to auto-detected context window.\n",
                    file=sys.stderr,
                )
                _config_context_length = None

        # 存储以在 switch_model 中复用（使配置覆盖在模型切换间持久）
        self._config_context_length = _config_context_length

        # 检查 custom_providers 中每个模型的 context_length
        if _config_context_length is None:
            try:
                from hermes_cli.config import get_compatible_custom_providers
                _custom_providers = get_compatible_custom_providers(_agent_cfg)
            except Exception:
                _custom_providers = _agent_cfg.get("custom_providers")
                if not isinstance(_custom_providers, list):
                    _custom_providers = []
            for _cp_entry in _custom_providers:
                if not isinstance(_cp_entry, dict):
                    continue
                _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                if _cp_url and _cp_url == self.base_url.rstrip("/"):
                    _cp_models = _cp_entry.get("models", {})
                    if isinstance(_cp_models, dict):
                        _cp_model_cfg = _cp_models.get(self.model, {})
                        if isinstance(_cp_model_cfg, dict):
                            _cp_ctx = _cp_model_cfg.get("context_length")
                            if _cp_ctx is not None:
                                try:
                                    _config_context_length = int(_cp_ctx)
                                except (TypeError, ValueError):
                                    logger.warning(
                                        "Invalid context_length for model %r in "
                                        "custom_providers: %r — must be a plain "
                                        "integer (e.g. 256000, not '256K'). "
                                        "Falling back to auto-detection.",
                                        self.model, _cp_ctx,
                                    )
                                    print(
                                        f"\n⚠ Invalid context_length for model {self.model!r} in custom_providers: {_cp_ctx!r}\n"
                                        f"  Must be a plain integer (e.g. 256000, not '256K').\n"
                                        f"  Falling back to auto-detected context window.\n",
                                        file=sys.stderr,
                                    )
                    break
        
        # 选择上下文引擎：配置驱动（类似记忆提供商）。
        # 1. 检查 config.yaml 中 context.engine 设置
        # 2. 检查 plugins/context_engine/<name>/ 目录（仓库自带）
        # 3. 检查通用插件系统（用户安装的插件）
        # 4. 回退到内置 ContextCompressor
        _selected_engine = None
        _engine_name = "compressor"  # default
        try:
            _ctx_cfg = _agent_cfg.get("context", {}) if isinstance(_agent_cfg, dict) else {}
            _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
        except Exception:
            pass

        if _engine_name != "compressor":
            # 尝试从 plugins/context_engine/<name>/ 加载
            try:
                from plugins.context_engine import load_context_engine
                _selected_engine = load_context_engine(_engine_name)
            except Exception as _ce_load_err:
                logger.debug("Context engine load from plugins/context_engine/: %s", _ce_load_err)

            # 尝试通用插件系统作为回退
            if _selected_engine is None:
                try:
                    from hermes_cli.plugins import get_plugin_context_engine
                    _candidate = get_plugin_context_engine()
                    if _candidate and _candidate.name == _engine_name:
                        _selected_engine = _candidate
                except Exception:
                    pass

            if _selected_engine is None:
                logger.warning(
                    "Context engine '%s' not found — falling back to built-in compressor",
                    _engine_name,
                )
        # 否则：配置指定 "compressor"——使用内置，不自动激活插件

        if _selected_engine is not None:
            self.context_compressor = _selected_engine
            # 为插件引擎解析 context_length——镜像 switch_model() 路径
            from agent.model_metadata import get_model_context_length
            _plugin_ctx_len = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
            )
            self.context_compressor.update_model(
                model=self.model,
                context_length=_plugin_ctx_len,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                provider=self.provider,
            )
            if not self.quiet_mode:
                logger.info("Using context engine: %s", _selected_engine.name)
        else:
            self.context_compressor = ContextCompressor(
                model=self.model,
                threshold_percent=compression_threshold,
                protect_first_n=3,
                protect_last_n=compression_protect_last,
                summary_target_ratio=compression_target_ratio,
                summary_model_override=None,
                quiet_mode=self.quiet_mode,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
                api_mode=self.api_mode,
            )
        self.compression_enabled = compression_enabled

        # 拒绝上下文窗口低于可靠工具调用工作流
        # 所需最小值的模型（64K tokens）。
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        _ctx = getattr(self.context_compressor, "context_length", 0)
        if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Model {self.model} has a context window of {_ctx:,} tokens, "
                f"which is below the minimum {MINIMUM_CONTEXT_LENGTH:,} required "
                f"by Hermes Agent.  Choose a model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context, or set "
                f"model.context_length in config.yaml to override."
            )

        # 注入上下文引擎工具 schema（如 lcm_grep、lcm_describe、lcm_expand）
        self._context_engine_tool_names: set = set()
        if hasattr(self, "context_compressor") and self.context_compressor and self.tools is not None:
            for _schema in self.context_compressor.get_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    self.valid_tool_names.add(_tname)
                    self._context_engine_tool_names.add(_tname)

        # 通知上下文引擎会话开始
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_start(
                    self.session_id,
                    hermes_home=str(get_hermes_home()),
                    platform=self.platform or "cli",
                    model=self.model,
                    context_length=getattr(self.context_compressor, "context_length", 0),
                )
            except Exception as _ce_err:
                logger.debug("Context engine on_session_start: %s", _ce_err)

        self._subdirectory_hints = SubdirectoryHintTracker(
            working_dir=os.getenv("TERMINAL_CWD") or None,
        )
        self._user_turn_count = 0

        # 会话的累计 token 使用量
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # ── Ollama num_ctx injection ──
        # Ollama 默认为 2048 上下文，无论模型能力如何。
        # 在连接 Ollama 服务器时，检测模型的最大上下文并
        # 在每次聊天请求中传递 num_ctx 以使用完整窗口。
        # 用户覆盖：在 config.yaml 中设置 model.ollama_num_ctx 来限制 VRAM 使用。
        self._ollama_num_ctx: int | None = None
        _ollama_num_ctx_override = None
        if isinstance(_model_cfg, dict):
            _ollama_num_ctx_override = _model_cfg.get("ollama_num_ctx")
        if _ollama_num_ctx_override is not None:
            try:
                self._ollama_num_ctx = int(_ollama_num_ctx_override)
            except (TypeError, ValueError):
                logger.debug("Invalid ollama_num_ctx config value: %r", _ollama_num_ctx_override)
        if self._ollama_num_ctx is None and self.base_url and is_local_endpoint(self.base_url):
            try:
                _detected = query_ollama_num_ctx(self.model, self.base_url, api_key=self.api_key or "")
                if _detected and _detected > 0:
                    self._ollama_num_ctx = _detected
            except Exception as exc:
                logger.debug("Ollama num_ctx detection failed: %s", exc)
        if self._ollama_num_ctx and not self.quiet_mode:
            logger.info(
                "Ollama num_ctx: will request %d tokens (model max from /api/show)",
                self._ollama_num_ctx,
            )

        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")

        # 立即检查以便 CLI 用户在启动时看到警告。
        # Gateway status_callback 尚未连接，因此任何警告
        # 存储在 _compression_warning 中并在首次 run_conversation() 时重放。
        self._compression_warning = None
        self._check_compression_model_feasibility()

        # 快照主运行时状态用于每轮恢复。当在一轮中激活回退时，
        # 下一轮恢复这些值，使首选模型每次都能获得新的尝试机会。
        # 使用单个字典存储，方便新增状态字段而无需 N 个独立属性。
        
        _cc = self.context_compressor
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "use_native_cache_layout": self._use_native_cache_layout,
            # _try_activate_fallback() 覆盖的上下文引擎状态。
            # 使用 getattr 访问 model/base_url/api_key/provider，因为
            # 插件引擎可能没有这些属性（它们是 ContextCompressor 专用的）。
            "compressor_model": getattr(_cc, "model", self.model),
            "compressor_base_url": getattr(_cc, "base_url", self.base_url),
            "compressor_api_key": getattr(_cc, "api_key", ""),
            "compressor_provider": getattr(_cc, "provider", self.provider),
            "compressor_context_length": _cc.context_length,
            "compressor_threshold_tokens": _cc.threshold_tokens,
        }
        if self.api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

    def reset_session_state(self):
        """重置所有会话级 token 计数器 — Reset all session-scoped token counters to 0.

        [中文] 封装所有会话级度量指标的清零逻辑:
        token 使用量 (输入/输出/总计/提示/完成)、缓存读写、API 调用次数、
        推理 token、成本估算、上下文压缩器内部计数器。使用 hasattr 安全处理可选属性。"""
        # Token 使用量计数器
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # 轮次计数器（在 reset_session_state 最初编写后添加——#2635）
        self._user_turn_count = 0

        # 上下文引擎重置（适用于内置压缩器和插件）
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.on_session_reset()
    
    def switch_model(self, new_model, new_provider, api_key='', base_url='', api_mode=''):
        """运行时切换模型/提供商 — Switch model/provider in-place for a live agent.

        [中文] 由 /model 命令调用 (CLI 和 gateway)。解析凭据、重建客户端、
        更新缓存标志、刷新上下文压缩器。实现镜像 _try_activate_fallback() 的
        客户端交换逻辑, 同时更新 _primary_runtime 使变更跨轮次持久化。"""
        from hermes_cli.providers import determine_api_mode

        # ── Determine api_mode if not provided ──
        if not api_mode:
            api_mode = determine_api_mode(new_provider, base_url)

        # 纵深防御：确保 OpenCode base_url 不携带尾部
        # /v1 进入 anthropic_messages 客户端，否则 SDK 会
        # 访问 /v1/v1/messages。`model_switch.switch_model()` 已剥离此内容，
        # 但在此处设防，防止任何直接调用者（未来代码路径、
        # 测试）重新引入双 /v1 的 404 bug。
        if (
            api_mode == "anthropic_messages"
            and new_provider in ("opencode-zen", "opencode-go")
            and isinstance(base_url, str)
            and base_url
        ):
            base_url = re.sub(r"/v1/?$", "", base_url)

        old_model = self.model
        old_provider = self.provider

        # ── Swap core runtime fields ──
        self.model = new_model
        self.provider = new_provider
        self.base_url = base_url or self.base_url
        self.api_mode = api_mode
        # 使传输缓存失效——新的 api_mode 可能需要不同的传输层
        if hasattr(self, "_transport_cache"):
            self._transport_cache.clear()
        if api_key:
            self.api_key = api_key

        # ── Build new client ──
        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            # 仅当提供商确实是 Anthropic 时才回退到 ANTHROPIC_TOKEN。
            # 其他 anthropic_messages 提供商（MiniMax、Alibaba 等）必须使用各自的
            # API key——回退会将 Anthropic 凭据发送到第三方端点。
            _is_native_anthropic = new_provider == "anthropic"
            effective_key = (api_key or self.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or self.api_key or "")
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url or getattr(self, "_anthropic_base_url", None)
            self._anthropic_client = build_anthropic_client(
                effective_key, self._anthropic_base_url,
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
            self._is_anthropic_oauth = _is_oauth_token(effective_key) if _is_native_anthropic else False
            self.client = None
            self._client_kwargs = {}
        else:
            effective_key = api_key or self.api_key
            effective_base = base_url or self.base_url
            self._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            _sm_timeout = get_provider_request_timeout(self.provider, self.model)
            if _sm_timeout is not None:
                self._client_kwargs["timeout"] = _sm_timeout
            self.client = self._create_openai_client(
                dict(self._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        # ── Re-evaluate prompt caching ──
        self._use_prompt_caching, self._use_native_cache_layout = (
            self._anthropic_prompt_cache_policy(
                provider=new_provider,
                base_url=self.base_url,
                api_mode=api_mode,
                model=new_model,
            )
        )

        # ── Update context compressor ──
        if hasattr(self, "context_compressor") and self.context_compressor:
            from agent.model_metadata import get_model_context_length
            new_context_length = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                provider=self.provider,
                config_context_length=getattr(self, "_config_context_length", None),
            )
            self.context_compressor.update_model(
                model=self.model,
                context_length=new_context_length,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                provider=self.provider,
                api_mode=self.api_mode,
            )

        # ── Invalidate cached system prompt so it rebuilds next turn ──
        self._cached_system_prompt = None

        # ── Update _primary_runtime so the change persists across turns ──
        _cc = self.context_compressor if hasattr(self, "context_compressor") and self.context_compressor else None
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "use_native_cache_layout": self._use_native_cache_layout,
            "compressor_model": getattr(_cc, "model", self.model) if _cc else self.model,
            "compressor_base_url": getattr(_cc, "base_url", self.base_url) if _cc else self.base_url,
            "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
            "compressor_provider": getattr(_cc, "provider", self.provider) if _cc else self.provider,
            "compressor_context_length": _cc.context_length if _cc else 0,
            "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
        }
        if api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

        # ── Reset fallback state ──
        self._fallback_activated = False
        self._fallback_index = 0

        # 当用户有意切换主提供商时（如 openrouter
        # → anthropic), drop any fallback entries that target the OLD primary
        # 而非旧的那个。回退链在 Agent 初始化时从配置种下，
        # 为原始提供商而设——若不修剪，新提供商上的失败轮次
        # 会静默重新激活用户刚刚拒绝的提供商，
        # 这正是 TUI v2 闪电测试期间报告的问题
        #（"切换到了 anthropic，TUI 仍尝试 openrouter"）。
        old_norm = (old_provider or "").strip().lower()
        new_norm = (new_provider or "").strip().lower()
        if old_norm and new_norm and old_norm != new_norm:
            self._fallback_chain = [
                entry for entry in self._fallback_chain
                if (entry.get("provider") or "").strip().lower() not in {old_norm, new_norm}
            ]
            self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None

        logging.info(
            "Model switched in-place: %s (%s) -> %s (%s)",
            old_model, old_provider, new_model, new_provider,
        )

    def _safe_print(self, *args, **kwargs):
        """安全打印 — 静默处理断管道/关闭的 stdout, 防止 headless 环境崩溃。

        Print that silently handles broken pipes / closed stdout.

        In headless environments (systemd, Docker, nohup) stdout may become
        unavailable mid-session.  A raw ``print()`` raises ``OSError`` which
        can crash cron jobs and lose completed work.

        Internally routes through ``self._print_fn`` (default: builtin
        ``print``) so callers such as the CLI can inject a renderer that
        handles ANSI escape sequences properly (e.g. prompt_toolkit's
        ``print_formatted_text(ANSI(...))``) without touching this method.
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """详细打印 — 流式输出时自动抑制, 工具执行时允许显示。Verbose print — suppressed when actively streaming tokens.

        Pass ``force=True`` for error/warning messages that should always be
        shown even during streaming playback (TTS or display).

        During tool execution (``_executing_tools`` is True), printing is
        allowed even with stream consumers registered because no tokens
        are being streamed at that point.

        After the main response has been delivered and the remaining tool
        calls are post-response housekeeping (``_mute_post_response``),
        all non-forced output is suppressed.

        ``suppress_status_output`` is a stricter CLI automation mode used by
        parseable single-query flows such as ``hermes chat -q``. In that mode,
        all status/diagnostic prints routed through ``_vprint`` are suppressed
        so stdout stays machine-readable.
        """
        if getattr(self, "suppress_status_output", False):
            return
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """判断静音模式下 spinner 输出是否有安全的接收端 — Check safe sink for quiet-mode spinner.

        In headless/stdio-protocol environments, a raw spinner with no custom
        ``_print_fn`` falls back to ``sys.stdout`` and can corrupt protocol
        streams such as ACP JSON-RPC. Allow quiet spinners only when either:
        - output is explicitly rerouted via ``_print_fn``; or
        - stdout is a real TTY.
        """
        if self._print_fn is not None:
            return True
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _should_emit_quiet_tool_messages(self) -> bool:
        """中文说明 — Return True when quiet-mode tool summaries should print directly.

        Quiet mode is used by both the interactive CLI and embedded/library
        callers. The CLI may still want compact progress hints when no callback
        owns rendering. Embedded/library callers, on the other hand, expect
        quiet mode to be truly silent.
        """
        return (
            self.quiet_mode
            and not self.tool_progress_callback
            and getattr(self, "platform", "") == "cli"
        )

    def _emit_status(self, message: str) -> None:
        """生命周期状态消息 — Emit a lifecycle status message to CLI and gateway.

        [中文] CLI 用户通过 _vprint(force=True) 始终可见, gateway 消费者通过 status_callback 接收。"""

        CLI users see the message via ``_vprint(force=True)`` so it is always
        visible regardless of verbose/quiet mode.  Gateway consumers receive
        it through ``status_callback("lifecycle", ...)``.

        This helper never raises — exceptions are swallowed so it cannot
        interrupt the retry/fallback logic.
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("lifecycle", message)
            except Exception:
                logger.debug("status_callback error in _emit_status", exc_info=True)

    def _current_main_runtime(self) -> Dict[str, str]:
        """中文说明 — Return the live main runtime for session-scoped auxiliary routing."""
        return {
            "model": getattr(self, "model", "") or "",
            "provider": getattr(self, "provider", "") or "",
            "base_url": getattr(self, "base_url", "") or "",
            "api_key": getattr(self, "api_key", "") or "",
            "api_mode": getattr(self, "api_mode", "") or "",
        }

    def _check_compression_model_feasibility(self) -> None:
        """中文说明 — Warn at session start if the auxiliary compression model's context
        window is smaller than the main model's compression threshold.

        When the auxiliary model cannot fit the content that needs summarising,
        compression will either fail outright (the LLM call errors) or produce
        a severely truncated summary.

        Called during ``__init__`` so CLI users see the warning immediately
        (via ``_vprint``).  The gateway sets ``status_callback`` *after*
        construction, so ``_replay_compression_warning()`` re-sends the
        stored warning through the callback on the first
        ``run_conversation()`` call.
        """
        if not self.compression_enabled:
            return
        try:
            from agent.auxiliary_client import get_text_auxiliary_client
            from agent.model_metadata import (
                MINIMUM_CONTEXT_LENGTH,
                get_model_context_length,
            )

            client, aux_model = get_text_auxiliary_client(
                "compression",
                main_runtime=self._current_main_runtime(),
            )
            if client is None or not aux_model:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "No auxiliary LLM provider for compression — "
                    "summaries will be unavailable."
                )
                return

            aux_base_url = str(getattr(client, "base_url", ""))
            aux_api_key = str(getattr(client, "api_key", ""))

            aux_context = get_model_context_length(
                aux_model,
                base_url=aux_base_url,
                api_key=aux_api_key,
                config_context_length=getattr(self, "_aux_compression_context_length_config", None),
            )

            # 硬性下限：辅助压缩模型至少需要
            # MINIMUM_CONTEXT_LENGTH (64K) tokens 的上下文。主模型
            # 已被要求满足此下限（在 __init__ 中已检查）；
            # __init__), so the compression model must too — otherwise it
            # 一个完整的阈值大小窗口的主模型内容。
            # 镜像了主模型的拒绝模式。
            if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
                raise ValueError(
                    f"Auxiliary compression model {aux_model} has a context "
                    f"window of {aux_context:,} tokens, which is below the "
                    f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                    f"Agent.  Choose a compression model with at least "
                    f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                    f"auxiliary.compression.model in config.yaml), or set "
                    f"auxiliary.compression.context_length to override the "
                    f"detected value if it is wrong."
                )

            threshold = self.context_compressor.threshold_tokens
            if aux_context < threshold:
                # 自动修正：降低当前会话阈值，使
                # 压缩在当前会话中确实可用。上述硬性下限
                # 保证了 aux_context >= MINIMUM_CONTEXT_LENGTH，
                # 因此新阈值始终 >= 64K。
                old_threshold = threshold
                new_threshold = aux_context
                self.context_compressor.threshold_tokens = new_threshold
                # 保持 threshold_percent 同步，以便将来主模型的
                # context_length 变更（update_model）能从一个合理的
                # 数值重新推导，而非使用最初过高的值。
                main_ctx = self.context_compressor.context_length
                if main_ctx:
                    self.context_compressor.threshold_percent = (
                        new_threshold / main_ctx
                    )
                safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
                msg = (
                    f"⚠ Compression model ({aux_model}) context is "
                    f"{aux_context:,} tokens, but the main model's "
                    f"compression threshold was {old_threshold:,} tokens. "
                    f"Auto-lowered this session's threshold to "
                    f"{new_threshold:,} tokens so compression can run.\n"
                    f"  To make this permanent, edit config.yaml — either:\n"
                    f"  1. Use a larger compression model:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{old_threshold:,}+-context>\n"
                    f"  2. Lower the compression threshold:\n"
                    f"       compression:\n"
                    f"         threshold: 0.{safe_pct:02d}"
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "Auxiliary compression model %s has %d token context, "
                    "below the main model's compression threshold of %d "
                    "tokens — auto-lowered session threshold to %d to "
                    "keep compression working.",
                    aux_model,
                    aux_context,
                    old_threshold,
                    new_threshold,
                )
        except ValueError:
            # 硬性拒绝（辅助模型低于最低上下文）必须传播
            # 以便会话拒绝启动。
            raise
        except Exception as exc:
            logger.debug(
                "Compression feasibility check failed (non-fatal): %s", exc
            )

    def _replay_compression_warning(self) -> None:
        """中文说明 — Re-send the compression warning through ``status_callback``.

        During ``__init__`` the gateway's ``status_callback`` is not yet
        wired, so ``_emit_status`` only reaches ``_vprint`` (CLI).  This
        method is called once at the start of the first
        ``run_conversation()`` — by then the gateway has set the callback,
        so every platform (Telegram, Discord, Slack, etc.) receives the
        warning.
        """
        msg = getattr(self, "_compression_warning", None)
        if msg and self.status_callback:
            try:
                self.status_callback("lifecycle", msg)
            except Exception:
                pass

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """中文说明 — Return True when a base URL targets OpenAI's native API."""
        if base_url is not None:
            hostname = base_url_hostname(base_url)
        else:
            hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
                getattr(self, "_base_url_lower", "")
            )
        return hostname == "api.openai.com"

    def _resolved_api_call_timeout(self) -> float:
        """中文说明 — Resolve the effective per-call request timeout in seconds.

        Priority:
          1. ``providers.<id>.models.<model>.timeout_seconds`` (per-model override)
          2. ``providers.<id>.request_timeout_seconds`` (provider-wide)
          3. ``HERMES_API_TIMEOUT`` env var (legacy escape hatch)
          4. 1800.0s default

        Used by OpenAI-wire chat completions (streaming and non-streaming) so
        the per-provider config knob wins over the 1800s default.  Without this
        helper, the hardcoded ``HERMES_API_TIMEOUT`` fallback would always be
        passed as a per-call ``timeout=`` kwarg, overriding the client-level
        timeout the AIAgent.__init__ path configured.
        """
        cfg = get_provider_request_timeout(self.provider, self.model)
        if cfg is not None:
            return cfg
        return float(os.getenv("HERMES_API_TIMEOUT", 1800.0))

    def _resolved_api_call_stale_timeout_base(self) -> tuple[float, bool]:
        """中文说明 — Resolve the base non-stream stale timeout and whether it is implicit.

        Priority:
          1. ``providers.<id>.models.<model>.stale_timeout_seconds``
          2. ``providers.<id>.stale_timeout_seconds``
          3. ``HERMES_API_CALL_STALE_TIMEOUT`` env var
          4. 300.0s default

        Returns ``(timeout_seconds, uses_implicit_default)`` so the caller can
        preserve legacy behaviors that only apply when the user has *not*
        explicitly configured a stale timeout, such as auto-disabling the
        detector for local endpoints.
        """
        cfg = get_provider_stale_timeout(self.provider, self.model)
        if cfg is not None:
            return cfg, False

        env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT")
        if env_timeout is not None:
            return float(env_timeout), False

        return 300.0, True

    def _compute_non_stream_stale_timeout(self, messages: list[dict[str, Any]]) -> float:
        """中文说明 — Compute the effective non-stream stale timeout for this request."""
        stale_base, uses_implicit_default = self._resolved_api_call_stale_timeout_base()
        base_url = getattr(self, "_base_url", None) or self.base_url or ""
        if uses_implicit_default and base_url and is_local_endpoint(base_url):
            return float("inf")

        est_tokens = sum(len(str(v)) for v in messages) // 4
        if est_tokens > 100_000:
            return max(stale_base, 600.0)
        if est_tokens > 50_000:
            return max(stale_base, 450.0)
        return stale_base

    def _is_openrouter_url(self) -> bool:
        """中文说明 — Return True when the base URL targets OpenRouter."""
        return base_url_host_matches(self._base_url_lower, "openrouter.ai")

    def _anthropic_prompt_cache_policy(
        self,
        *,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[bool, bool]:
        """中文说明 — Decide whether to apply Anthropic prompt caching and which layout to use.

        Returns ``(should_cache, use_native_layout)``:
          * ``should_cache`` — inject ``cache_control`` breakpoints for this
            request (applies to OpenRouter Claude, native Anthropic, and
            third-party gateways that speak the native Anthropic protocol).
          * ``use_native_layout`` — place markers on the *inner* content
            blocks (native Anthropic accepts and requires this layout);
            when False markers go on the message envelope (OpenRouter and
            OpenAI-wire proxies expect the looser layout).

        Third-party providers using the native Anthropic transport
        (``api_mode == 'anthropic_messages'`` + Claude-named model) get
        caching with the native layout so they benefit from the same
        cost reduction as direct Anthropic callers, provided their
        gateway implements the Anthropic cache_control contract
        (MiniMax, Zhipu GLM, LiteLLM's Anthropic proxy mode all do).

        Qwen / Alibaba-family models on OpenCode, OpenCode Go, and direct
        Alibaba (DashScope) also honour Anthropic-style ``cache_control``
        markers on OpenAI-wire chat completions. Upstream pi-mono #3392 /
        pi #3393 documented this for opencode-go Qwen. Without markers
        these providers serve zero cache hits, re-billing the full prompt
        on every turn.
        """
        eff_provider = (provider if provider is not None else self.provider) or ""
        eff_base_url = base_url if base_url is not None else (self.base_url or "")
        eff_api_mode = api_mode if api_mode is not None else (self.api_mode or "")
        eff_model = (model if model is not None else self.model) or ""

        base_lower = eff_base_url.lower()
        model_lower = eff_model.lower()
        provider_lower = eff_provider.lower()
        is_claude = "claude" in model_lower
        is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
        is_anthropic_wire = eff_api_mode == "anthropic_messages"
        is_native_anthropic = (
            is_anthropic_wire
            and (eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com")
        )

        if is_native_anthropic:
            return True, True
        if is_openrouter and is_claude:
            return True, False
        if is_anthropic_wire and is_claude:
            # 第三方兼容 Anthropic 的网关。
            return True, True

        # Qwen/Alibaba 在 OpenCode (Zen/Go) 和原生 DashScope 上：
        # OpenAI 协议传输，但接受 Anthropic 风格的 cache_control 标记并
        # 提供真正的缓存命中。若无此分支，
        # qwen3.6-plus 在 opencode-go 上报告 0% 缓存 token 并
        # 每轮都消耗订阅配额。
        model_is_qwen = "qwen" in model_lower
        provider_is_alibaba_family = provider_lower in {
            "opencode", "opencode-zen", "opencode-go", "alibaba",
        }
        if provider_is_alibaba_family and model_is_qwen:
            # 信封布局（native_anthropic=False）：标记在内部
            # 内容部分上，而非顶级工具消息。匹配
            # pi-mono 的 "alibaba" cacheControlFormat。
            return True, False

        return False, False

    @staticmethod
    def _model_requires_responses_api(model: str) -> bool:
        """中文说明 — Return True for models that require the Responses API path.

        GPT-5.x models are rejected on /v1/chat/completions by both
        OpenAI and OpenRouter (error: ``unsupported_api_for_model``).
        Detect these so the correct api_mode is set regardless of
        which provider is serving the model.
        """
        m = model.lower()
        # Strip vendor prefix (e.g. "openai/gpt-5.4" → "gpt-5.4")
        if "/" in m:
            m = m.rsplit("/", 1)[-1]
        return m.startswith("gpt-5")

    @staticmethod
    def _provider_model_requires_responses_api(
        model: str,
        *,
        provider: Optional[str] = None,
    ) -> bool:
        """中文说明 — Return True when this provider/model pair should use Responses API."""
        normalized_provider = (provider or "").strip().lower()
        if normalized_provider == "copilot":
            try:
                from hermes_cli.models import _should_use_copilot_responses_api
                return _should_use_copilot_responses_api(model)
            except Exception:
                # 如果 Copilot 专用逻辑因任何原因不可用，
                # 回退到通用的 GPT-5 规则。
                pass
        return AIAgent._model_requires_responses_api(model)

    def _max_tokens_param(self, value: int) -> dict:
        """中文说明 — Return the correct max tokens kwarg for the current provider.

        OpenAI's newer models (gpt-4o, o-series, gpt-5+) require
        'max_completion_tokens'. OpenRouter, local models, and older
        OpenAI models use 'max_tokens'.
        """
        if self._is_direct_openai_url():
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    def _has_content_after_think_block(self, content: str) -> bool:
        """
        中文说明: 检查思考块之后是否还有实际文本内容。

        Check if content has actual text after any reasoning/thinking blocks.

        This detects cases where the model only outputs reasoning but no actual
        response, which indicates an incomplete generation that should be retried.
        Must stay in sync with _strip_think_blocks() tag variants.

        Args:
            content: The assistant message content to check

        Returns:
            True if there's meaningful content after think blocks, False otherwise
        """
        if not content:
            return False

        # 移除所有推理标签变体（必须匹配 _strip_think_blocks）
        cleaned = self._strip_think_blocks(content)

        # 检查是否还有非空白内容残留
        return bool(cleaned.strip())
    
    def _strip_think_blocks(self, content: str) -> str:
        """中文说明 — Remove reasoning/thinking blocks from content, returning only visible text.

        Handles four cases:
          1. Closed tag pairs (``<think>…</think>``) — the common path when
             the provider emits complete reasoning blocks.
          2. Unterminated open tag at a block boundary (start of text or
             after a newline) — e.g. MiniMax M2.7 / NIM endpoints where the
             closing tag is dropped.  Everything from the open tag to end
             of string is stripped.  The block-boundary check mirrors
             ``gateway/stream_consumer.py``'s filter so models that mention
             ``<think>`` in prose aren't over-stripped.
          3. Stray orphan open/close tags that slip through.
          4. Tag variants: ``<think>``, ``<thinking>``, ``<reasoning>``,
             ``<REASONING_SCRATCHPAD>``, ``<thought>`` (Gemma 4), all
             case-insensitive.

        Additionally strips standalone tool-call XML blocks that some open
        models (notably Gemma variants on OpenRouter) emit inside assistant
        content instead of via the structured ``tool_calls`` field:
          * ``<tool_call>…</tool_call>``
          * ``<tool_calls>…</tool_calls>``
          * ``<tool_result>…</tool_result>``
          * ``<function_call>…</function_call>``
          * ``<function_calls>…</function_calls>``
          * ``<function name="…">…</function>`` (Gemma style)
        Ported from openclaw/openclaw#67318. The ``<function>`` variant is
        boundary-gated (only strips when the tag sits at start-of-line or
        after punctuation and carries a ``name="..."`` attribute) so prose
        mentions like "Use <function> in JavaScript" are preserved.
        """
        if not content:
            return ""
        # 1. Closed tag pairs — case-insensitive for all variants so
        #    混合大小写标签（<THINK>、<Thinking>）不会漏到
        #    未闭合标签处理中，避免带走尾部内容。
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # 1b. 工具调用 XML 块（openclaw/openclaw#67318）。处理
        #     generic tag names first — they have no attribute gating since
        #    正文中出现字面量 <tool_call> 本身已极为罕见。
        for _tc_name in ("tool_call", "tool_calls", "tool_result",
                          "function_call", "function_calls"):
            content = re.sub(
                rf'<{_tc_name}\b[^>]*>.*?</{_tc_name}>',
                '',
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )
        # 1c. <function name="...">...</function> — Gemma-style standalone
        #    工具调用。仅当标签位于块边界
        #    （文本开头、换行后或句子结束标点后）
        #    且带有 name="..." 属性时才移除。这使得
        #    "Use <function> to declare" 这类正文引用保持安全。
        content = re.sub(
            r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
            r'<function\b[^>]*\bname\s*=[^>]*>'
            r'(?:(?:(?!</function>).)*)</function>',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # 2. Unterminated reasoning block — open tag at a block boundary
        #    （文本开头或换行后）且无匹配闭合标签。
        #    从标签起截断到字符串末尾。修复 #8878 / #9568
        #    （MiniMax M2.7 将原始推理泄露到 assistant 内容中）。
        content = re.sub(
            r'(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # 3. 遗漏的孤立开/闭标签。
        content = re.sub(
            r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*',
            '',
            content,
            flags=re.IGNORECASE,
        )
        # 3b. 孤立的工具调用闭标签。（不移除裸 <function> 或
        #     未闭合的 <function name="...">，因为在流式传输中
        #     被截断的尾部可能仍对用户有价值；匹配
        #     OpenClaw 故意设计的不对称性。）
        content = re.sub(
            r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*',
            '',
            content,
            flags=re.IGNORECASE,
        )
        return content

    @staticmethod
    def _has_natural_response_ending(content: str) -> bool:
        """中文说明 — Heuristic: does visible assistant text look intentionally finished?"""
        if not content:
            return False
        stripped = content.rstrip()
        if not stripped:
            return False
        if stripped.endswith("```"):
            return True
        return stripped[-1] in '.!?:)"\']}。！？：）】」』》'

    def _is_ollama_glm_backend(self) -> bool:
        """中文说明 — Detect the narrow backend family affected by Ollama/GLM stop misreports."""
        model_lower = (self.model or "").lower()
        provider_lower = (self.provider or "").lower()
        if "glm" not in model_lower and provider_lower != "zai":
            return False
        if "ollama" in self._base_url_lower or ":11434" in self._base_url_lower:
            return True
        return bool(self.base_url and is_local_endpoint(self.base_url))

    def _should_treat_stop_as_truncated(
        self,
        finish_reason: str,
        assistant_message,
        messages: Optional[list] = None,
    ) -> bool:
        """中文说明 — Detect conservative stop->length misreports for Ollama-hosted GLM models."""
        if finish_reason != "stop" or self.api_mode != "chat_completions":
            return False
        if not self._is_ollama_glm_backend():
            return False
        if not any(
            isinstance(msg, dict) and msg.get("role") == "tool"
            for msg in (messages or [])
        ):
            return False
        if assistant_message is None or getattr(assistant_message, "tool_calls", None):
            return False

        content = getattr(assistant_message, "content", None)
        if not isinstance(content, str):
            return False

        visible_text = self._strip_think_blocks(content).strip()
        if not visible_text:
            return False
        if len(visible_text) < 20 or not re.search(r"\s", visible_text):
            return False

        return not self._has_natural_response_ending(visible_text)

    def _looks_like_codex_intermediate_ack(
        self,
        user_message: str,
        assistant_content: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """中文说明 — Detect a planning/ack message that should continue instead of ending the turn."""
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
            return False

        assistant_text = self._strip_think_blocks(assistant_content or "").strip().lower()
        if not assistant_text:
            return False
        if len(assistant_text) > 1200:
            return False

        has_future_ack = bool(
            re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
        )
        if not has_future_ack:
            return False

        action_markers = (
            "look into",
            "look at",
            "inspect",
            "scan",
            "check",
            "analyz",
            "review",
            "explore",
            "read",
            "open",
            "run",
            "test",
            "fix",
            "debug",
            "search",
            "find",
            "walkthrough",
            "report back",
            "summarize",
        )
        workspace_markers = (
            "directory",
            "current directory",
            "current dir",
            "cwd",
            "repo",
            "repository",
            "codebase",
            "project",
            "folder",
            "filesystem",
            "file tree",
            "files",
            "path",
        )

        user_text = (user_message or "").strip().lower()
        user_targets_workspace = (
            any(marker in user_text for marker in workspace_markers)
            or "~/" in user_text
            or "/" in user_text
        )
        assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
        assistant_targets_workspace = any(
            marker in assistant_text for marker in workspace_markers
        )
        return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action
    
    
    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        中文说明: 从助手消息中提取推理/思考内容。

        Extract reasoning/thinking content from an assistant message.
        
        OpenRouter and various providers can return reasoning in multiple formats:
        1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
        2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
        3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)
        
        Args:
            assistant_message: The assistant message object from the API response
            
        Returns:
            Combined reasoning text, or None if no reasoning found
        """
        reasoning_parts = []
        
        # 检查直接 reasoning 字段
        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)
        
        # 检查 reasoning_content 字段（部分提供商使用的替代名称）
        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            # 如果与 reasoning 相同则不重复
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)
        
        # 检查 reasoning_details 数组（OpenRouter 统一格式）
        # 格式：[{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # 从推理详情对象中提取 summary
                    summary = (
                        detail.get('summary')
                        or detail.get('thinking')
                        or detail.get('content')
                        or detail.get('text')
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # 部分提供商将推理直接嵌入 assistant 内容中
        # 而非返回结构化的推理字段。仅在没有结构化推理
        # 时才回退到内联提取。
        content = getattr(assistant_message, "content", None)
        if not reasoning_parts and isinstance(content, str) and content:
            inline_patterns = (
                r"<think>(.*?)</think>",
                r"<thinking>(.*?)</thinking>",
                r"<thought>(.*?)</thought>",
                r"<reasoning>(.*?)</reasoning>",
                r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
            )
            for pattern in inline_patterns:
                flags = re.DOTALL | re.IGNORECASE
                for block in re.findall(pattern, content, flags=flags):
                    cleaned = block.strip()
                    if cleaned and cleaned not in reasoning_parts:
                        reasoning_parts.append(cleaned)
        
        # 合并所有推理部分
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None

    def _cleanup_task_resources(self, task_id: str) -> None:
        """中文说明 — Clean up VM and browser resources for a given task.

        Skips ``cleanup_vm`` when the active terminal environment is marked
        persistent (``persistent_filesystem=True``) so that long-lived sandbox
        containers survive between turns. The idle reaper in
        ``terminal_tool._cleanup_inactive_envs`` still tears them down once
        ``terminal.lifetime_seconds`` is exceeded. Non-persistent backends are
        torn down per-turn as before to prevent resource leakage (the original
        intent of this hook for the Morph backend, see commit fbd3a2fd).
        """
        try:
            if is_persistent_env(task_id):
                if self.verbose_logging:
                    logging.debug(
                        f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                        f"idle reaper will handle it."
                    )
            else:
                cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # 后台记忆/技能审查
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "Review the conversation above and consider saving to memory if appropriate.\n\n"
        "Focus on:\n"
        "1. Has the user revealed things about themselves — their persona, desires, "
        "preferences, or personal details worth remembering?\n"
        "2. Has the user expressed expectations about how you should behave, their work "
        "style, or ways they want you to operate?\n\n"
        "If something stands out, save it using the memory tool. "
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _SKILL_REVIEW_PROMPT = (
        "Review the conversation above and consider saving or updating a skill if appropriate.\n\n"
        "Focus on: was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome?\n\n"
        "If a relevant skill already exists, update it with what you learned. "
        "Otherwise, create a new skill if the approach is reusable.\n"
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and consider two things:\n\n"
        "**Memory**: Has the user revealed things about themselves — their persona, "
        "desires, preferences, or personal details? Has the user expressed expectations "
        "about how you should behave, their work style, or ways they want you to operate? "
        "If so, save using the memory tool.\n\n"
        "**Skills**: Was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome? If a relevant skill "
        "already exists, update it. Otherwise, create a new one if the approach is reusable.\n\n"
        "Only act if there's something genuinely worth saving. "
        "If nothing stands out, just say 'Nothing to save.' and stop."
    )

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """中文说明 — Spawn a background thread to review the conversation for memory/skill saves.

        Creates a full AIAgent fork with the same model, tools, and context as the
        main session. The review prompt is appended as the next user turn in the
        forked conversation. Writes directly to the shared memory/skill stores.
        Never modifies the main conversation history or produces user-visible output.
        """
        import threading

        # 根据触发的触发器选择正确的提示词
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib
            review_agent = None
            try:
                with open(os.devnull, "w") as _devnull, \
                     contextlib.redirect_stdout(_devnull), \
                     contextlib.redirect_stderr(_devnull):
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=8,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                    )
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # 扫描审查 Agent 的消息中成功的工具操作
                # 并向用户展示简洁摘要。
                actions = []
                for msg in getattr(review_agent, "_session_messages", []):
                    if not isinstance(msg, dict) or msg.get("role") != "tool":
                        continue
                    try:
                        data = json.loads(msg.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not data.get("success"):
                        continue
                    message = data.get("message", "")
                    target = data.get("target", "")
                    if "created" in message.lower():
                        actions.append(message)
                    elif "updated" in message.lower():
                        actions.append(message)
                    elif "added" in message.lower() or (target and "add" in message.lower()):
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "Entry added" in message:
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "removed" in message.lower() or "replaced" in message.lower():
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(f"  💾 {summary}")
                    _bg_cb = self.background_review_callback
                    if _bg_cb:
                        try:
                            _bg_cb(f"💾 {summary}")
                        except Exception:
                            pass

            except Exception as e:
                logger.debug("Background memory/skill review failed: %s", e)
            finally:
                # 关闭所有资源（httpx 客户端、子进程等），
                # 防止 GC 在已死的 asyncio 事件循环上尝试清理，
                # 产生"Event loop is closed"错误。
                if review_agent is not None:
                    try:
                        review_agent.close()
                    except Exception:
                        pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """中文说明 — Rewrite the current-turn user message before persistence/return.

        Some call paths need an API-only user-message variant without letting
        that synthetic text leak into persisted transcripts or resumed session
        history. When an override is configured for the active turn, mutate the
        in-memory messages list in place so both persistence and returned
        history stay clean.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """中文说明 — Save session state to both JSON log and SQLite on any exit path.

        Ensures conversations are never lost, even on errors or early returns.
        Skipped when ``persist_session=False`` (ephemeral helper flows).
        """
        if not self.persist_session:
            return
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)

    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """中文说明 — Persist any un-flushed messages to the SQLite session store.

        Uses _last_flushed_db_idx to track which messages have already been
        written, so repeated calls (from multiple exit paths) only write
        truly new messages — preventing the duplicate-write bug (#860).
        """
        if not self._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # 如果 create_session() 在启动时失败（如临时锁），
            # 会话行可能尚不存在。ensure_session() 使用 INSERT OR
            # IGNORE，因此如果行已存在则为空操作。
            self._session_db.ensure_session(
                self.session_id,
                source=self.platform or "cli",
                model=self.model,
            )
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and isinstance(msg.tool_calls, list) and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_content=msg.get("reasoning_content") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        中文说明: 获取截至(但不包括)最后一个助手轮次的消息。

        Get messages up to (but not including) the last assistant turn.
        
        This is used when we need to "roll back" to the last successful point
        in the conversation, typically when the final assistant message is
        incomplete or malformed.
        
        Args:
            messages: Full message list
            
        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []
        
        # 查找最后一条 assistant 消息的索引
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # 未找到 assistant 消息，返回所有消息
            return messages.copy()
        
        # 返回到（不包括）最后一条 assistant 消息的所有内容
        return messages[:last_assistant_idx]
    
    def _format_tools_for_system_message(self) -> str:
        """
        中文说明: 格式化工具定义为系统消息中的轨迹格式。

        Format tool definitions for the system message in the trajectory format.
        
        Returns:
            str: JSON string representation of tool definitions
        """
        if not self.tools:
            return "[]"
        
        # 将工具定义转换为轨迹期望的格式
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools, ensure_ascii=False)
    
    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """
        中文说明: 将内部消息格式转换为轨迹格式以便保存。

        Convert internal message format to trajectory format for saving.
        
        Args:
            messages (List[Dict]): Internal message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
            
        Returns:
            List[Dict]: Messages in trajectory format
        """
        trajectory = []
        
        # 添加带有工具定义的 system 消息
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({
            "from": "system",
            "value": system_msg
        })
        
        # 将实际的用户提示（来自数据集）添加为第一条 human 消息
        trajectory.append({
            "from": "human",
            "value": user_query
        })
        
        # 跳过第一条消息（用户查询），因为我们在上面已添加。
        # Prefill 消息仅在 API 调用时注入（不在 messages
        # 列表中），因此此处无需偏移调整。
        i = 1
        
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                # 检查此消息是否包含工具调用
                if "tool_calls" in msg and msg["tool_calls"]:
                    # 格式化带有工具调用的 assistant 消息
                    # 为轨迹存储将推理包装在 <think> 标签中
                    content = ""
                    
                    # 如果有推理内容，放在 <think> 标签前缀中（原生思考 token）
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    if msg.get("content") and msg["content"].strip():
                        # 将 <REASONING_SCRATCHPAD> 标签转换为 <think> 标签
                        #（原生思考禁用且模型通过 XML 推理时使用）
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"
                    
                    # 添加用 XML 标签包裹的工具调用
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        # 解析参数——应该始终成功，因为我们在对话期间进行了验证
                        # 但保留 try-except 作为安全网
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            # 这不应发生，因为我们在对话期间验证和重试，
                            # 但如果发生，记录警告并使用空字典
                            logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
                    # 确保每个 gpt 轮次都有 <think> 块（无推理时为空）
                    # 使训练数据格式保持一致
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })
                    
                    # 收集所有后续的工具响应
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # 使用 XML 标签格式化工具响应
                        tool_response = "<tool_response>\n"
                        
                        # 如果看起来像 JSON，尝试解析工具内容
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # Keep as string if not valid JSON
                        
                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": tool_name,
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    # 将所有工具响应合并为单条消息
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1  # Skip the tool messages we just processed
                
                else:
                    # 无工具调用的常规 assistant 消息
                    # 为轨迹存储将推理包装在 <think> 标签中
                    content = ""
                    
                    # 如果有推理内容，放在 <think> 标签前缀中（原生思考 token）
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    # 将 <REASONING_SCRATCHPAD> 标签转换为 <think> 标签
                    #（原生思考禁用且模型通过 XML 推理时使用）
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)
                    
                    # 确保每个 gpt 轮次都有 <think> 块（无推理时为空）
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
                    })
            
            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })
            
            i += 1
        
        return trajectory
    
    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """
        中文说明: 将对话轨迹保存到 JSONL 文件。

        Save conversation trajectory to JSONL file.
        
        Args:
            messages (List[Dict]): Complete message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
        """
        if not self.save_trajectories:
            return
        
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)
    
    @staticmethod
    def _summarize_api_error(error: Exception) -> str:
        """中文说明 — Extract a human-readable one-liner from an API error.

        Handles Cloudflare HTML error pages (502, 503, etc.) by pulling the
        <title> tag instead of dumping raw HTML.  Falls back to a truncated
        str(error) for everything else.
        """
        raw = str(error)

        # Cloudflare / 代理 HTML 页面：提取 <title> 用于清晰摘要
        if "<!DOCTYPE" in raw or "<html" in raw:
            m = re.search(r"<title[^>]*>([^<]+)</title>", raw, re.IGNORECASE)
            title = m.group(1).strip() if m else "HTML error page (title not found)"
            # 同时提取 Cloudflare Ray ID（若存在）
            ray = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
            ray_id = ray.group(1).strip() if ray else None
            status_code = getattr(error, "status_code", None)
            parts = []
            if status_code:
                parts.append(f"HTTP {status_code}")
            parts.append(title)
            if ray_id:
                parts.append(f"Ray {ray_id}")
            return " — ".join(parts)

        # OpenAI/Anthropic SDK 的 JSON 正文错误
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                return f"{prefix}{msg[:300]}"

        # 回退：截断原始字符串，但留出 200 字符以上的空间
        status_code = getattr(error, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return f"{prefix}{raw[:500]}"

    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        """中文说明 — Mask API key for safe logging."""
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def _clean_error_message(self, error_msg: str) -> str:
        """
        中文说明: 清理错误消息以便显示给用户，移除 HTML 内容并截断。

        Clean up error messages for user display, removing HTML content and truncating.
        
        Args:
            error_msg: Raw error message from API or exception
            
        Returns:
            Clean, user-friendly error message
        """
        if not error_msg:
            return "Unknown error"
            
        # 移除 HTML 内容（CloudFlare 和网关错误页面中常见）
        if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
            return "Service temporarily unavailable (HTML error page returned)"
            
        # 移除换行和多余空白
        cleaned = ' '.join(error_msg.split())
        
        # 若过长则截断
        if len(cleaned) > 150:
            cleaned = cleaned[:150] + "..."
            
        return cleaned

    @staticmethod
    def _extract_api_error_context(error: Exception) -> Dict[str, Any]:
        """中文说明 — Extract structured rate-limit details from provider errors."""
        context: Dict[str, Any] = {}

        body = getattr(error, "body", None)
        payload = None
        if isinstance(body, dict):
            payload = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(payload, dict):
            reason = payload.get("code") or payload.get("error")
            if isinstance(reason, str) and reason.strip():
                context["reason"] = reason.strip()
            message = payload.get("message") or payload.get("error_description")
            if isinstance(message, str) and message.strip():
                context["message"] = message.strip()
            for key in ("resets_at", "reset_at"):
                value = payload.get(key)
                if value not in (None, ""):
                    context["reset_at"] = value
                    break
            retry_after = payload.get("retry_after")
            if retry_after not in (None, "") and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass
            ratelimit_reset = headers.get("x-ratelimit-reset")
            if ratelimit_reset and "reset_at" not in context:
                context["reset_at"] = ratelimit_reset

        if "message" not in context:
            raw_message = str(error).strip()
            if raw_message:
                context["message"] = raw_message[:500]

        if "reset_at" not in context:
            message = context.get("message") or ""
            if isinstance(message, str):
                delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
                if delay_match:
                    value = float(delay_match.group(1))
                    seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                    context["reset_at"] = time.time() + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

        return context

    def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[Dict[str, Any]]:
        """中文说明 — Token buckets for ``post_api_request`` plugins (no raw ``response`` object)."""
        if response is None:
            return None
        raw_usage = getattr(response, "usage", None)
        if not raw_usage:
            return None
        from dataclasses import asdict

        cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
        summary = asdict(cu)
        summary.pop("raw_usage", None)
        summary["prompt_tokens"] = cu.prompt_tokens
        summary["total_tokens"] = cu.total_tokens
        return summary

    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        中文说明: 为推理 API 生成调试友好的 HTTP 请求记录。

        Dump a debug-friendly HTTP request record for the active inference API.

        Captures the request body from api_kwargs (excluding transport-only keys
        like timeout). Intended for debugging provider-side 4xx failures where
        retries are not useful.
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(response_obj, "status_code", None)
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """中文说明 — Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        中文说明: 将完整的原始会话保存到 JSON 文件。

        Save the full raw session to a JSON file.

        Stores every message exactly as the agent sees it: user messages,
        assistant messages (with reasoning, finish_reason, tool_calls),
        tool responses (with tool_call_id, tool_name), and injected system
        messages (compression summaries, todo snapshots, etc.).

        REASONING_SCRATCHPAD tags are converted to <think> blocks for consistency.
        Overwritten after each turn so it always reflects the latest state.
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # 清理 assistant 内容用于会话日志
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            # 守卫：绝不用更少消息覆盖更大的会话日志。
            # 防止 --resume 加载的会话
            # messages weren't fully written to SQLite — the resumed agent starts
            # 使用部分历史覆盖完整 JSON 日志导致数据丢失。
            if self.session_log_file.exists():
                try:
                    existing = json.loads(self.session_log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # corrupted existing file — allow the overwrite

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
    
    def interrupt(self, message: str = None) -> None:
        """
        中文说明: 请求中断 Agent 当前的工具调用循环。

        Request the agent to interrupt its current tool-calling loop.
        
        Call this from another thread (e.g., input handler, message receiver)
        to gracefully stop the agent and process a new message.
        
        Also signals long-running tool executions (e.g. terminal commands)
        to terminate early, so the agent can respond immediately.
        
        Args:
            message: Optional new message that triggered the interrupt.
                     If provided, the agent will include this in its response context.
        
        Example (CLI):
            # 在独立的输入线程中：
            if user_typed_something:
                agent.interrupt(user_input)
        
        Example (Messaging):
            # 当活跃会话收到新消息时：
            if session_has_running_agent:
                running_agent.interrupt(new_message.text)
        """
        self._interrupt_requested = True
        self._interrupt_message = message
        # 通知所有工具立即中止进行中的操作。
        # 将中断范围限定为本 Agent 的执行线程，避免影响
        # 同一进程中运行的其他 Agent（网关）。
        if self._execution_thread_id is not None:
            _set_interrupt(True, self._execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            # 中断在 run_conversation() 完成
            # 将 Agent 绑定到执行线程之前到达。推迟工具级
            # 中断信号直到启动完成，而非错误地
            # 发送给调用者线程。
            self._interrupt_thread_signal_pending = True
        # 分发给并发工具工作线程。这些工作线程在自己的
        # tid 上运行工具（ThreadPoolExecutor worker），因此仅当
        # 其特定 tid 在 `_interrupted_threads` 集合中时，
        # 工具内的 `is_interrupted()` 才能感知到中断。若无此传播，
        # 已运行的并发工具（如卡在网络 I/O 上的终端命令）
        # 永远不会感知中断，只能等到自身超时。
        # 参见 `_run_tool` 了解匹配的进入/退出记录。
        # `getattr` 回退覆盖了通过 object.__new__ 构建 AIAgent
        # 并跳过 __init__ 的测试桩。
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(True, _wtid)
                except Exception:
                    pass
        # 将中断传播到所有运行中的子 Agent（子 Agent 委托）
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
    
    def clear_interrupt(self) -> None:
        """中文说明 — Clear any pending interrupt request and the per-thread tool interrupt signal."""
        self._interrupt_requested = False
        self._interrupt_message = None
        self._interrupt_thread_signal_pending = False
        if self._execution_thread_id is not None:
            _set_interrupt(False, self._execution_thread_id)
        # 同时清除并发工具工作线程的标记。被追踪的
        # 工作线程通常在退出时自行清除标记，但显式
        # 清除可以保证没有过期的中断能在轮次边界存活
        # 并在后续无关的工具调用（恰好调度到同一
        # 回收的 worker tid）上触发。
        # `getattr` 回退覆盖了通过 object.__new__ 构建 AIAgent
        # 并跳过 __init__ 的测试桩。
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(False, _wtid)
                except Exception:
                    pass
        # 硬中断优先于任何待处理的 /steer——steer 本意是给
        # Agent 的下一次工具调用迭代的，但该迭代不再发生。
        # 丢弃它，而非在中断后的轮次中意外地向用户
        # 注入延迟消息。
        _steer_lock = getattr(self, "_pending_steer_lock", None)
        if _steer_lock is not None:
            with _steer_lock:
                self._pending_steer = None

    def steer(self, text: str) -> bool:
        """
        中文说明: 在不中断当前操作的情况下将用户消息注入到下一个工具结果中。

        Inject a user message into the next tool result without interrupting.

        Unlike interrupt(), this does NOT stop the current tool call. The
        text is stashed and the agent loop appends it to the LAST tool
        result's content once the current tool batch finishes. The model
        sees the steer as part of the tool output on its next iteration.

        Thread-safe: callable from gateway/CLI/TUI threads. Multiple calls
        before the drain point concatenate with newlines.

        Args:
            text: The user text to inject. Empty strings are ignored.

        Returns:
            True if the steer was accepted, False if the text was empty.
        """
        if not text or not text.strip():
            return False
        cleaned = text.strip()
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            # 通过 object.__new__ 构建 AIAgent 的测试桩跳过了 __init__。
            # 回退到直接属性设置；这些桩中不涉及并发调用者。
            #
            existing = getattr(self, "_pending_steer", None)
            self._pending_steer = (existing + "\n" + cleaned) if existing else cleaned
            return True
        with _lock:
            if self._pending_steer:
                self._pending_steer = self._pending_steer + "\n" + cleaned
            else:
                self._pending_steer = cleaned
        return True

    def _drain_pending_steer(self) -> Optional[str]:
        """中文说明 — Return the pending steer text (if any) and clear the slot.

        Safe to call from the agent execution thread after appending tool
        results. Returns None when no steer is pending.
        """
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            text = getattr(self, "_pending_steer", None)
            self._pending_steer = None
            return text
        with _lock:
            text = self._pending_steer
            self._pending_steer = None
        return text

    def _apply_pending_steer_to_tool_results(self, messages: list, num_tool_msgs: int) -> None:
        """中文说明 — Append any pending /steer text to the last tool result in this turn.

        Called at the end of a tool-call batch, before the next API call.
        The steer is appended to the last ``role:"tool"`` message's content
        with a clear marker so the model understands it came from the user
        and NOT from the tool itself. Role alternation is preserved —
        nothing new is inserted, we only modify existing content.

        Args:
            messages: The running messages list.
            num_tool_msgs: Number of tool results appended in this batch;
                used to locate the tail slice safely.
        """
        if num_tool_msgs <= 0 or not messages:
            return
        steer_text = self._drain_pending_steer()
        if not steer_text:
            return
        # 在最近的尾部中查找最后一条 tool-role 消息。跳过
        # 非 tool 消息可防止未来代码在边界处追加
        # 其他内容。
        target_idx = None
        for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
            msg = messages[j]
            if isinstance(msg, dict) and msg.get("role") == "tool":
                target_idx = j
                break
        if target_idx is None:
            # 此批次中无工具结果（如全部被中断跳过）；
            # 放回 steer 以便调用者的回退路径能将其作为
            # 正常的下一轮用户消息传递。
            _lock = getattr(self, "_pending_steer_lock", None)
            if _lock is not None:
                with _lock:
                    if self._pending_steer:
                        self._pending_steer = self._pending_steer + "\n" + steer_text
                    else:
                        self._pending_steer = steer_text
            else:
                existing = getattr(self, "_pending_steer", None)
                self._pending_steer = (existing + "\n" + steer_text) if existing else steer_text
            return
        marker = f"\n\nUser guidance: {steer_text}"
        existing_content = messages[target_idx].get("content", "")
        if not isinstance(existing_content, str):
            # Anthropic multimodal content blocks — preserve them and append
            # 末尾的文本块。
            try:
                blocks = list(existing_content) if existing_content else []
                blocks.append({"type": "text", "text": marker.lstrip()})
                messages[target_idx]["content"] = blocks
            except Exception:
                # 若内容格式异常则回退到字符串替换。
                messages[target_idx]["content"] = f"{existing_content}{marker}"
        else:
            messages[target_idx]["content"] = existing_content + marker
        logger.info(
            "Delivered /steer to agent after tool batch (%d chars): %s",
            len(steer_text),
            steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
        )

    def _touch_activity(self, desc: str) -> None:
        """中文说明 — Update the last-activity timestamp and description (thread-safe)."""
        self._last_activity_ts = time.time()
        self._last_activity_desc = desc

    def _capture_rate_limits(self, http_response: Any) -> None:
        """中文说明 — Parse x-ratelimit-* headers from an HTTP response and cache the state.

        Called after each streaming API call.  The httpx Response object is
        available on the OpenAI SDK Stream via ``stream.response``.
        """
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        try:
            from agent.rate_limit_tracker import parse_rate_limit_headers
            state = parse_rate_limit_headers(headers, provider=self.provider)
            if state is not None:
                self._rate_limit_state = state
        except Exception:
            pass  # Never let header parsing break the agent loop

    def get_rate_limit_state(self):
        """中文说明 — Return the last captured RateLimitState, or None."""
        return self._rate_limit_state

    def get_activity_summary(self) -> dict:
        """中文说明 — Return a snapshot of the agent's current activity for diagnostics.

        Called by the gateway timeout handler to report what the agent was doing
        when it was killed, and by the periodic "still working" notifications.
        """
        elapsed = time.time() - self._last_activity_ts
        return {
            "last_activity_ts": self._last_activity_ts,
            "last_activity_desc": self._last_activity_desc,
            "seconds_since_activity": round(elapsed, 1),
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "budget_used": self.iteration_budget.used,
            "budget_max": self.iteration_budget.max_total,
        }

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """中文说明 — Shut down the memory provider and context engine — call at actual session boundaries.

        This calls on_session_end() then shutdown_all() on the memory
        manager, and on_session_end() on the context engine.
        NOT called per-turn — only at CLI exit, /reset, gateway
        session expiry, etc.
        """
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
            try:
                self._memory_manager.shutdown_all()
            except Exception:
                pass
        # 通知上下文引擎会话结束（刷新 DAG、关闭数据库等）
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass
    
    def commit_memory_session(self, messages: list = None) -> None:
        """中文说明 — Trigger end-of-session extraction without tearing providers down.
        Called when session_id rotates (e.g. /new, context compression);
        providers keep their state and continue running under the old
        session_id — they just flush pending extraction now."""
        if not self._memory_manager:
            return
        try:
            self._memory_manager.on_session_end(messages or [])
        except Exception:
            pass

    def release_clients(self) -> None:
        """中文说明 — Release LLM client resources WITHOUT tearing down session tool state.

        Used by the gateway when evicting this agent from _agent_cache for
        memory-management reasons (LRU cap or idle TTL) — the session may
        resume at any time with a freshly-built AIAgent that reuses the
        same task_id / session_id, so we must NOT kill:
          - process_registry entries for task_id (user's bg shells)
          - terminal sandbox for task_id (cwd, env, shell state)
          - browser daemon for task_id (open tabs, cookies)
          - memory provider (has its own lifecycle; keeps running)

        We DO close:
          - OpenAI/httpx client pool (big chunk of held memory + sockets;
            the rebuilt agent gets a fresh client anyway)
          - Active child subagents (per-turn artefacts; safe to drop)

        Safe to call multiple times.  Distinct from close() — which is the
        hard teardown for actual session boundaries (/new, /reset, session
        expiry).
        """
        # 关闭活跃的子 Agent（按轮次；无跨轮次持久化）。
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.release_clients()
                except Exception:
                    # 回退到子 Agent 的完全关闭；它们是按轮次的。
                    try:
                        child.close()
                    except Exception:
                        pass
        except Exception:
            pass

        # 关闭 OpenAI/httpx 客户端以立即释放 socket。
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="cache_evict", shared=True)
                self.client = None
        except Exception:
            pass

    def close(self) -> None:
        """释放 Agent 持有的全部资源 — 防止进程/连接泄漏.

        [中文] 资源清理按以下顺序执行 (每步独立保护, 一步失败不影响其余):
          1. 终止此任务的后台进程 (ProcessRegistry)
          2. 清理终端沙箱环境 (Docker/SSH/Daytona 等)
          3. 清理浏览器守护进程会话
          4. 关闭所有活跃的子 Agent (delegate_task 创建的)
          5. 关闭 OpenAI/httpx 客户端连接

        设计为幂等 — 可安全多次调用。"""
        task_id = getattr(self, "session_id", None) or ""

        # 1. 终止此任务的后台进程
        try:
            from tools.process_registry import process_registry
            process_registry.kill_all(task_id=task_id)
        except Exception:
            pass

        # 2. 清理终端沙箱环境
        try:
            cleanup_vm(task_id)
        except Exception:
            pass

        # 3. 清理浏览器守护进程会话
        try:
            cleanup_browser(task_id)
        except Exception:
            pass

        # 4. 关闭活跃的子 Agent
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.close()
                except Exception:
                    pass
        except Exception:
            pass

        # 5. 关闭 OpenAI/httpx 客户端
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="agent_close", shared=True)
                self.client = None
        except Exception:
            pass

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        中文说明: 从对话历史中恢复待办事项状态。

        Recover todo state from conversation history.
        
        The gateway creates a fresh AIAgent per message, so the in-memory
        TodoStore is empty. We scan the history for the most recent todo
        tool response and replay it to reconstruct the state.
        """
        # 从历史记录末尾向前查找最近的 todo 工具响应
        last_todo_response = None
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # 快速检查：todo 响应包含 "todos" key
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if last_todo_response:
            # 将条目重放到存储中（replace 模式）
            self._todo_store.write(last_todo_response, merge=False)
            if not self.quiet_mode:
                self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)
    
    @property
    def is_interrupted(self) -> bool:
        """中文说明 — Check if an interrupt has been requested."""
        return self._interrupt_requested










    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        组装完整系统提示词，按层级叠加。

        ═══════════════════════════════════════════════════════════════════════
        系统提示词构建架构
        ═══════════════════════════════════════════════════════════════════════

        [整体架构]

            ┌─────────────────────────────────────────────────────────┐
            │              _build_system_prompt() 构建                │
            │         (缓存到 _cached_system_prompt, 会话内稳定)       │
            │                                                         │
            │  Layer 1:  Agent 身份 (SOUL.md / DEFAULT_AGENT_IDENTITY)│
            │  Layer 2:  工具感知行为指导 (memory/skills/session)      │
            │  Layer 3:  Nous 订阅提示                                │
            │  Layer 4:  工具使用强制 (tool_use_enforcement)           │
            │  Layer 5:  用户/网关 system_message (如提供)             │
            │  Layer 6:  持久记忆快照 (MEMORY.md + USER.md)           │
            │  Layer 7:  外部记忆提供商 (Honcho/Mem0 等)              │
            │  Layer 8:  技能清单索引 (可用技能列表)                   │
            │  Layer 9:  上下文文件 (AGENTS.md, .cursorrules 等)      │
            │  Layer 10: 时间戳/元数据 (日期, session, model, provider)│
            │  Layer 11: 环境提示 (WSL, Termux 等)                    │
            │  Layer 12: 平台格式提示 (WhatsApp/Telegram 等)          │
            └─────────────────────────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────────────────────┐
            │          API 调用时拼接 (每次请求动态组合)               │
            │                                                         │
            │  effective_system = _cached_system_prompt                │
            │                  + "\n\n" + ephemeral_system_prompt     │
            │                                                         │
            │  → 注入为 messages[0] (role=system)                     │
            └─────────────────────────────────────────────────────────┘

        [ephemeral_system_prompt 设计]

            ephemeral_system_prompt 是"临时系统提示"，只在 API 调用时追加，
            不写入 _cached_system_prompt。这样做的目的是保护 Anthropic prompt
            caching —— Anthropic 的缓存基于 system prompt 前缀匹配，如果每次
            请求前缀不同，缓存就会失效。

            因此稳定的缓存前缀 (_cached_system_prompt) 和临时追加
            (ephemeral_system_prompt) 分离，确保缓存命中率最大化。

        [缓存策略]

            - _build_system_prompt() 只构建一次，结果缓存到 _cached_system_prompt
            - 上下文压缩 (context compression) 后才重建
            - 这确保系统提示在会话内稳定，最大化 Anthropic 前缀缓存命中率

        [层级说明]

          1. Agent 身份 — SOUL.md (自定义人格) 或 DEFAULT_AGENT_IDENTITY (硬编码默认)
          2. 工具感知行为指导 — MEMORY_GUIDANCE / SESSION_SEARCH_GUIDANCE / SKILLS_GUIDANCE
          3. 用户/网关系统提示 (如提供)
          4. 持久记忆快照 — MEMORY.md + USER.md (冻结在构建时)
          5. 外部记忆提供商区块 (Honcho, Mem0 等插件)
          6. 技能清单索引 — 可用技能列表供模型发现
          7. 上下文文件 — AGENTS.md, .cursorrules, HERMES.md (会扫描提示注入)
          8. 时间戳/元数据 — 日期、会话 ID、模型名、提供商
          9. 环境提示 — WSL, Termux 检测
         10. 平台提示 — WhatsApp(无markdown), Telegram(原生markdown) 等格式指导
        """
        # ── Layer 1: Agent 身份 ──────────────────────────────────────
        # 优先加载 SOUL.md 自定义人格；未找到时回退到 DEFAULT_AGENT_IDENTITY。
        # skip_context_files=True 时跳过 SOUL.md（用于批量模式等场景）。

        # 尝试 SOUL.md 作为主要身份 (除非跳过上下文文件)
        _soul_loaded = False
        if not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            # 回退到硬编码身份
            prompt_parts = [DEFAULT_AGENT_IDENTITY]

        # ── Layer 2: 工具感知行为指导 ─────────────────────────────────
        # 仅在对应工具已加载时注入对应指导，避免提示无关工具的行为规范。
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        # ── Layer 3: Nous 订阅提示 ────────────────────────────────────
        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            prompt_parts.append(nous_subscription_prompt)

        # ── Layer 4: 工具使用强制 ────────────────────────────────────
        # 告诉模型实际调用工具而不是
        # 描述意图操作。由 config.yaml 控制
        # agent.tool_use_enforcement：
        #   "auto" (默认) — 匹配 TOOL_USE_ENFORCEMENT_MODELS
        #   true  — 始终注入 (所有模型)
        #   false — 从不注入
        #   list  — 自定义模型名子串匹配
        if self.valid_tool_names:
            _enforce = self._tool_use_enforcement
            _inject = False
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in ("true", "always", "yes", "on")):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in ("false", "never", "no", "off")):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (self.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                # "auto" 或未识别的值 — 使用硬编码默认值
                model_lower = (self.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                prompt_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (self.model or "").lower()
                # Google 模型操作指导 (简洁性、绝对路径、
                # 并行工具调用、编辑前验证等)
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    prompt_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex 执行纪律 (工具持久性、
                # 前置检查、验证、反幻觉)。
                if "gpt" in _model_lower or "codex" in _model_lower:
                    prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        # 以便将用户引导到已有答案而非重新生成。

        # ── Layer 5: 用户/网关 system_message ────────────────────────
        # 注意: ephemeral_system_prompt 不在此包含。仅在 API 调用时
        # 注入，以便保持缓存/存储的系统提示不变。
        if system_message is not None:
            prompt_parts.append(system_message)

        # ── Layer 6: 持久记忆快照 ────────────────────────────────────
        # MEMORY.md (agent 记忆) + USER.md (用户画像)，构建时冻结。
        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # 启用时始终包含 USER.md。
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # ── Layer 7: 外部记忆提供商 ──────────────────────────────────
        # Honcho, Mem0 等插件的记忆区块，对内置 MEMORY.md 的补充。
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    prompt_parts.append(_ext_mem_block)
            except Exception:
                pass

        # ── Layer 8: 技能清单索引 ────────────────────────────────────
        # 可用技能列表供模型发现和调用。
        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in self.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        # ── Layer 9: 上下文文件 ──────────────────────────────────────
        # AGENTS.md, .cursorrules, HERMES.md 等项目级指导文件。
        # SOUL.md 已在 Layer 1 加载时在此排除，避免重复。
        if not self.skip_context_files:
            # 设置时使用 TERMINAL_CWD 进行上下文文件发现 (网关
            # 模式)。网关进程从 hermes-agent 安装目录运行，
            # 所以 os.getcwd() 会拾取 repo 的 AGENTS.md 和
            # 其他开发文件 — 无谓地增加约 10k token 消耗。
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        # ── Layer 10: 时间戳/元数据 ──────────────────────────────────
        # 构建时冻结的日期、会话 ID、模型名、提供商。
        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        # ── Layer 10.1: 模型身份修正 (Alibaba API bug workaround) ────
        # Alibaba Coding Plan API 始终返回 "glm-4.7" 作为模型名，不管
        # 请求的模型是什么。将明确的模型身份注入系统提示
        # 以便 agent 正确报告其模型 (绕过 API bug)。
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        # ── Layer 11: 环境提示 ───────────────────────────────────────
        # WSL, Termux 等特殊执行环境检测，告知 agent 路径转换和行为适配。
        _env_hints = build_environment_hints()
        if _env_hints:
            prompt_parts.append(_env_hints)

        # ── Layer 12: 平台格式提示 ──────────────────────────────────
        # WhatsApp(禁用 markdown)、Telegram(原生 markdown) 等格式适配。
        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(p.strip() for p in prompt_parts if p.strip())

    # =========================================================================
    # =========================================================================
    # Pre/post-call guardrails (inspired by PR #1321 — @alireza78a)

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """中文说明 — Extract call ID from a tool_call entry (dict or object)."""
        if isinstance(tc, dict):
            return tc.get("id", "") or ""
        return getattr(tc, "id", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """中文说明 — Fix orphaned tool_call / tool_result pairs before every LLM call.

        Runs unconditionally — not gated on whether the context compressor
        is present — so orphans from session loading or manual message
        manipulation are always caught.
        """
        # --- 角色白名单：丢弃 API 不会接受的角色消息 ---
        filtered = []
        for msg in messages:
            role = msg.get("role")
            if role not in AIAgent._VALID_API_ROLES:
                logger.debug(
                    "Pre-call sanitizer: dropping message with invalid role %r",
                    role,
                )
                continue
            filtered.append(msg)
        messages = filtered

        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = AIAgent._get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. 丢弃无匹配 assistant 调用的工具结果
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            logger.debug(
                "Pre-call sanitizer: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. 为结果被丢弃的调用注入桩结果
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = AIAgent._get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result unavailable — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            logger.debug(
                "Pre-call sanitizer: added %d stub tool result(s)",
                len(missing_results),
            )
        return messages

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """中文说明 — Truncate excess delegate_task calls to max_concurrent_children.

        The delegate_tool caps the task list inside a single call, but the
        model can emit multiple separate delegate_task tool_calls in one
        turn.  This truncates the excess, preserving all non-delegate calls.

        Returns the original list if no truncation was needed.
        """
        from tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < max_children:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "Truncated %d excess delegate_task call(s) to enforce "
            "max_concurrent_children=%d limit",
            delegate_count - max_children, max_children,
        )
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """中文说明 — Remove duplicate (tool_name, arguments) pairs within a single turn.

        Only the first occurrence of each unique pair is kept.
        Returns the original list if no duplicates were found.
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    def _repair_tool_call(self, tool_name: str) -> str | None:
        """中文说明 — Attempt to repair a mismatched tool name before aborting.

        1. Try lowercase
        2. Try normalized (lowercase + hyphens/spaces -> underscores)
        3. Try fuzzy match (difflib, cutoff=0.7)

        Returns the repaired name if found in valid_tool_names, else None.
        """
        from difflib import get_close_matches

        # 1. 转小写
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered

        # 2. 标准化
        normalized = lowered.replace("-", "_").replace(" ", "_")
        if normalized in self.valid_tool_names:
            return normalized

        # 3. 模糊匹配
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None

    def _invalidate_system_prompt(self):
        """使缓存的系统提示失效，强制下一轮重建。

        Invalidate the cached system prompt, forcing a rebuild on the next turn.
        
        Called after context compression events. Also reloads memory from disk
        so the rebuilt prompt captures any writes from this session.
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()

    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        """从工具调用内容生成确定性 call_id。

        Generate a deterministic call_id from tool call content.

        Used as a fallback when the API doesn't provide a call_id.
        Deterministic IDs prevent cache invalidation — random UUIDs would
        make every API call's prefix unique, breaking OpenAI's prompt cache.
        """
        return _codex_deterministic_call_id(fn_name, arguments, index)

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """将存储的工具 ID 拆分为 (call_id, response_item_id)。

        Split a stored tool id into (call_id, response_item_id).
        """
        return _codex_split_responses_tool_id(raw_id)

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """构建有效的 Responses `function_call.id`（必须以 `fc_` 开头）。

        Build a valid Responses `function_call.id` (must start with `fc_`).
        """
        return _codex_derive_responses_function_call_id(call_id, response_item_id)

    def _thread_identity(self) -> str:
        """返回当前线程标识字符串（线程名:线程ID）。 """

        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}

    def _client_log_context(self) -> str:
        """返回客户端日志上下文字符串（provider/base_url/model） """

        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        """返回线程安全的 OpenAI 客户端锁（RLock） """

        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """检查 OpenAI 客户端是否已关闭。

        Check if an OpenAI client is closed.

        Handles both property and method forms of is_closed:
        - httpx.Client.is_closed is a bool property
        - openai.OpenAI.is_closed is a method returning bool

        Prior bug: getattr(client, "is_closed", False) returned the bound method,
        which is always truthy, causing unnecessary client recreation on every call.
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # 处理方法（openai SDK）vs 属性（httpx）
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    @staticmethod
    def _build_keepalive_http_client() -> Any:
        """构建支持 TCP keepalive 的 httpx 客户端。
        """
        try:
            import httpx as _httpx
            import socket as _socket

            _sock_opts = [(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)]
            if hasattr(_socket, "TCP_KEEPIDLE"):
                _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE, 30))
                _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL, 10))
                _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, 3))
            elif hasattr(_socket, "TCP_KEEPALIVE"):
                _sock_opts.append((_socket.IPPROTO_TCP, _socket.TCP_KEEPALIVE, 30))
            # 提供自定义 transport 时，httpx 不会自动从环境变量读取
            # 代理设置（allow_env_proxies = trust_env 且 transport 为 None）。
            # 显式读取代理设置以确保 HTTP_PROXY/HTTPS_PROXY 生效。
            _proxy = _get_proxy_from_env()
            return _httpx.Client(
                transport=_httpx.HTTPTransport(socket_options=_sock_opts),
                proxy=_proxy,
            )
        except Exception:
            return None

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        """创建 OpenAI 客户端（或代码专用 copilot/gemini 客户端）。
        """
        from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
        # 将 client_kwargs 视为只读。调用者传入 self._client_kwargs（或其浅拷贝）；
        # 任何原地修改都会泄漏回存储的字典并在后续请求中
        # 被复用。#10933 因此问题：注入了 httpx.Client transport，
        # 该 transport 在首次请求后被拆除，导致下次请求
        # 包装已关闭的 transport 并在每次重试时抛出
        # "Cannot send a request, as the client has been closed"。回退解决了该路径；
        # 此拷贝锁定合约，防止未来 transport/keepalive 工作重新引入
        # 同类型的 bug。
        client_kwargs = dict(client_kwargs)
        _validate_proxy_env_urls()
        _validate_base_url(client_kwargs.get("base_url"))
        if self.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        if self.provider == "google-gemini-cli" or str(client_kwargs.get("base_url", "")).startswith("cloudcode-pa://"):
            from agent.gemini_cloudcode_adapter import GeminiCloudCodeClient

            # 剥离 Gemini 客户端不接受的 OpenAI 专用 kwarg
            safe_kwargs = {
                k: v for k, v in client_kwargs.items()
                if k in {"api_key", "base_url", "default_headers", "project_id", "timeout"}
            }
            client = GeminiCloudCodeClient(**safe_kwargs)
            logger.info(
                "Gemini Cloud Code Assist client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        if self.provider == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

            base_url = str(client_kwargs.get("base_url", "") or "")
            if is_native_gemini_base_url(base_url):
                safe_kwargs = {
                    k: v for k, v in client_kwargs.items()
                    if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
                }
                if "http_client" not in safe_kwargs:
                    keepalive_http = self._build_keepalive_http_client()
                    if keepalive_http is not None:
                        safe_kwargs["http_client"] = keepalive_http
                client = GeminiNativeClient(**safe_kwargs)
                logger.info(
                    "Gemini native client created (%s, shared=%s) %s",
                    reason,
                    shared,
                    self._client_log_context(),
                )
                return client
        #
        # 注入 TCP keepalive 使内核能检测到已死的提供商连接，
        # 而非让它们在 CLOSE-WAIT 中静默停滞（#10324）。若不这样，
        # 流中掉线的对端会使 socket 处于一种状态：
        # epoll_wait 永不触发，``httpx`` 读超时可能不触发，
        # Agent 一直挂起直到手动终止。空闲 30s 后探测，
        # every 10s, give up after 3 → dead peer detected within ~60s.
        # 针对 #10933 的安全措施：``client_kwargs = dict(client_kwargs)``
        # 确保此注入仅进入本地每次调用的副本，
        # 绝不回到 ``self._client_kwargs``。每次 ``_create_openai_client``
        # 调用因此获得自己的全新 ``httpx.Client``，其
        # 生命周期与传给的 OpenAI 客户端绑定。当
        # OpenAI 客户端关闭时（重建、拆除、凭据轮换），
        # 配对的 ``httpx.Client`` 随之关闭，下一次调用
        # constructs a fresh one — no stale closed transport can be reused.
        # ``tests/run_agent/test_create_openai_client_reuse.py`` 和
        # ``tests/run_agent/test_sequential_chats_live.py`` 中的测试锁定了此不变性。
        if "http_client" not in client_kwargs:
            keepalive_http = self._build_keepalive_http_client()
            if keepalive_http is not None:
                client_kwargs["http_client"] = keepalive_http
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """强制关闭底层 TCP socket 以防止 CLOSE-WAIT 堆积。

        Force-close underlying TCP sockets to prevent CLOSE-WAIT accumulation.

        When a provider drops a connection mid-stream, httpx's ``client.close()``
        performs a graceful shutdown which leaves sockets in CLOSE-WAIT until the
        OS times them out (often minutes).  This method walks the httpx transport
        pool and issues ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()`` to
        force an immediate TCP RST, freeing the file descriptors.

        Returns the number of sockets force-closed.
        """
        import socket as _socket

        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return 0
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return 0
            # httpx 使用 httpcore 连接池；连接存放在
            # _connections（列表）或 _pool（列表）中，取决于版本。
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            for conn in list(connections):
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        """关闭 OpenAI 客户端，先强制关闭 TCP socket。
        """
        if client is None:
            return
        # 先强制关闭 TCP socket 以防止 CLOSE-WAIT 累积，
        # 再做优雅的 SDK 级关闭。
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        """替换主 OpenAI 客户端（线程安全）。"""
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(self._client_kwargs, reason=reason, shared=True)
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        """确保主 OpenAI 客户端可用，必要时重建。"""
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _cleanup_dead_connections(self) -> bool:
        """检测并清理主客户端上的死 TCP 连接。

        Detect and clean up dead TCP connections on the primary client.

        Inspects the httpx connection pool for sockets in unhealthy states
        (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
        and rebuilds the primary client from scratch.

        Returns True if dead connections were found and cleaned up.
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # 检查空闲但 socket 已关闭的连接
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # 通过非阻塞 recv peek 探测 socket 健康状况
                import socket as _socket
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # No data available — socket is healthy
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    def _create_request_openai_client(self, *, reason: str) -> Any:
        """创建一次性请求专用 OpenAI 客户端。"""
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        """关闭一次性请求专用 OpenAI 客户端。"""
        self._close_openai_client(client, reason=reason, shared=False)

    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """执行一次流式 Responses API 请求并返回最终响应。

        Execute one streaming Responses API request and return the final response.
        """
        import httpx as _httpx

        active_client = client or self._ensure_primary_openai_client(reason="codex_stream_direct")
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        # 累积流式文本，以便在 get_final_response()
        # 返回空输出时可恢复（如 chatgpt.com backend-api 发送
        # response.incomplete 而非 response.completed）。
        self._codex_streamed_text_parts: list = []
        for attempt in range(max_stream_retries + 1):
            collected_output_items: list = []
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        self._touch_activity("receiving stream response")
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # 在文本内容增量时触发回调（工具调用期间抑制）
                        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", "")
                            if delta_text:
                                self._codex_streamed_text_parts.append(delta_text)
                            if delta_text and not has_tool_calls:
                                if not first_delta_fired:
                                    first_delta_fired = True
                                    if on_first_delta:
                                        try:
                                            on_first_delta()
                                        except Exception:
                                            pass
                                self._fire_stream_delta(delta_text)
                        # 追踪工具调用以抑制文本流输出
                        elif "function_call" in event_type:
                            has_tool_calls = True
                        # 触发推理回调
                        elif "reasoning" in event_type and "delta" in event_type:
                            reasoning_text = getattr(event, "delta", "")
                            if reasoning_text:
                                self._fire_reasoning_delta(reasoning_text)
                        # Collect completed output items — some backends
                        # (chatgpt.com/backend-api/codex) 流式传输有效条目，
                        # 通过 response.output_item.done，但 SDK 的
                        # get_final_response() 返回空输出列表。
                        elif event_type == "response.output_item.done":
                            done_item = getattr(event, "item", None)
                            if done_item is not None:
                                collected_output_items.append(done_item)
                        # 记录非完成的终端事件用于诊断
                        elif event_type in ("response.incomplete", "response.failed"):
                            resp_obj = getattr(event, "response", None)
                            status = getattr(resp_obj, "status", None) if resp_obj else None
                            incomplete_details = getattr(resp_obj, "incomplete_details", None) if resp_obj else None
                            logger.warning(
                                "Codex Responses stream received terminal event %s "
                                "(status=%s, incomplete_details=%s, streamed_chars=%d). %s",
                                event_type, status, incomplete_details,
                                sum(len(p) for p in self._codex_streamed_text_parts),
                                self._client_log_context(),
                            )
                    final_response = stream.get_final_response()
                    # 补丁：ChatGPT Codex 后端流式传输有效输出条目，
                    # 但 get_final_response() 可能返回空输出列表。
                    # 从收集的条目回填或从增量合成。
                    _out = getattr(final_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            final_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex stream: backfilled %d output items from stream events",
                                len(collected_output_items),
                            )
                        elif self._codex_streamed_text_parts and not has_tool_calls:
                            assembled = "".join(self._codex_streamed_text_parts)
                            final_response.output = [SimpleNamespace(
                                type="message",
                                role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex stream: synthesized output from %d text deltas (%d chars)",
                                len(self._codex_streamed_text_parts), len(assembled),
                            )
                    return final_response
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed (attempt %s/%s); retrying. %s error=%s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                        exc,
                    )
                    continue
                logger.debug(
                    "Codex Responses stream transport failed; falling back to create(stream=True). %s error=%s",
                    self._client_log_context(),
                    exc,
                )
                return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
            except RuntimeError as exc:
                err_text = str(exc)
                missing_completed = "response.completed" in err_text
                if missing_completed and attempt < max_stream_retries:
                    logger.debug(
                        "Responses stream closed before completion (attempt %s/%s); retrying. %s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                    )
                    continue
                if missing_completed:
                    logger.debug(
                        "Responses stream did not emit response.completed; falling back to create(stream=True). %s",
                        self._client_log_context(),
                    )
                    return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
                raise

    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """流完成边缘情况的回退路径（Codex 风格 Responses 后端）。

        Fallback path for stream completion edge cases on Codex-style Responses backends.
        """
        active_client = client or self._ensure_primary_openai_client(reason="codex_create_stream_fallback")
        fallback_kwargs = dict(api_kwargs)
        fallback_kwargs["stream"] = True
        fallback_kwargs = self._get_transport().preflight_kwargs(fallback_kwargs, allow_stream=True)
        stream_or_response = active_client.responses.create(**fallback_kwargs)

        # 兼容性填充层，用于仍返回具体响应的 mock 或提供商。
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        collected_output_items: list = []
        collected_text_deltas: list = []
        try:
            for event in stream_or_response:
                self._touch_activity("receiving stream response")
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")

                # 收集输出条目和文本增量用于回填
                if event_type == "response.output_item.done":
                    done_item = getattr(event, "item", None)
                    if done_item is None and isinstance(event, dict):
                        done_item = event.get("item")
                    if done_item is not None:
                        collected_output_items.append(done_item)
                elif event_type in ("response.output_text.delta",):
                    delta = getattr(event, "delta", "")
                    if not delta and isinstance(event, dict):
                        delta = event.get("delta", "")
                    if delta:
                        collected_text_deltas.append(delta)

                if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                    continue

                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
                    # 从收集的流事件回填空输出
                    _out = getattr(terminal_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            terminal_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex fallback stream: backfilled %d output items",
                                len(collected_output_items),
                            )
                        elif collected_text_deltas:
                            assembled = "".join(collected_text_deltas)
                            terminal_response.output = [SimpleNamespace(
                                type="message", role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex fallback stream: synthesized from %d deltas (%d chars)",
                                len(collected_text_deltas), len(assembled),
                            )
                    return terminal_response
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is not None:
            return terminal_response
        raise RuntimeError("Responses create(stream=True) fallback did not emit a terminal response.")

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        """尝试刷新 Codex 客户端凭据。"""
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        """尝试刷新 Nous 客户端凭据。"""
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous 请求不应继承 OpenRouter 专用归因 header。
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        """尝试刷新 Anthropic 客户端凭据。"""
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # 仅为原生 Anthropic 提供商刷新凭据。
        # 其他 anthropic_messages 提供商（MiniMax、Alibaba 等）使用各自的 key。
        if self.provider != "anthropic":
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(
                new_token,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # Update OAuth flag — token type may have changed (API key ↔ OAuth).
        # 仅在原生 Anthropic 上按 OAuth 处理；使用 Anthropic 协议的
        # 第三方端点不得触发 OAuth 路径（#1739 和第三方
        # 身份注入守护）。
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token) if self.provider == "anthropic" else False
        return True

    def _apply_client_headers_for_base_url(self, base_url: str) -> None:
        """根据 base_url 应用对应提供商的默认请求头。"""
        from agent.auxiliary_client import _AI_GATEWAY_HEADERS, _OR_HEADERS

        if base_url_host_matches(base_url, "openrouter.ai"):
            self._client_kwargs["default_headers"] = dict(_OR_HEADERS)
        elif base_url_host_matches(base_url, "ai-gateway.vercel.sh"):
            self._client_kwargs["default_headers"] = dict(_AI_GATEWAY_HEADERS)
        elif base_url_host_matches(base_url, "api.githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif base_url_host_matches(base_url, "api.kimi.com"):
            self._client_kwargs["default_headers"] = {"User-Agent": "claude-code/0.1.0"}
        elif base_url_host_matches(base_url, "portal.qwen.ai"):
            self._client_kwargs["default_headers"] = _qwen_portal_headers()
        elif base_url_host_matches(base_url, "chatgpt.com"):
            from agent.auxiliary_client import _codex_cloudflare_headers
            self._client_kwargs["default_headers"] = _codex_cloudflare_headers(
                self._client_kwargs.get("api_key", "")
            )
        else:
            self._client_kwargs.pop("default_headers", None)

    def _swap_credential(self, entry) -> None:
        """将当前凭据切换到凭证池中的下一个条目。"""
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base
            self._anthropic_client = build_anthropic_client(
                runtime_key, runtime_base,
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
            self._is_anthropic_oauth = _is_oauth_token(runtime_key) if self.provider == "anthropic" else False
            self.api_key = runtime_key
            self.base_url = runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(self.base_url)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        classified_reason: Optional[FailoverReason] = None,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """通过凭证池轮转尝试恢复凭据。

        Attempt credential recovery via pool rotation.

        Returns (recovered, has_retried_429).
        On rate limits: first occurrence retries same credential (sets flag True).
                        second consecutive failure rotates to next credential.
        On billing exhaustion: immediately rotates.
        On auth failures: attempts token refresh before rotating.

        `classified_reason` lets the recovery path honor the structured error
        classifier instead of relying only on raw HTTP codes. This matters for
        providers that surface billing/rate-limit/auth conditions under a
        different status code, such as Anthropic returning HTTP 400 for
        "out of extra usage".
        """
        pool = self._credential_pool
        if pool is None:
            return False, has_retried_429

        effective_reason = classified_reason
        if effective_reason is None:
            if status_code == 402:
                effective_reason = FailoverReason.billing
            elif status_code == 429:
                effective_reason = FailoverReason.rate_limit
            elif status_code == 401:
                effective_reason = FailoverReason.auth

        if effective_reason == FailoverReason.billing:
            rotate_status = status_code if status_code is not None else 402
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (billing) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, has_retried_429

        if effective_reason == FailoverReason.rate_limit:
            if not has_retried_429:
                return False, True
            rotate_status = status_code if status_code is not None else 429
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (rate limit) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, True

        if effective_reason == FailoverReason.auth:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                logger.info(f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}")
                self._swap_credential(refreshed)
                return True, has_retried_429
            # Refresh failed — rotate to next credential instead of giving up.
            # 失败条目已由 try_refresh_current() 标记为耗尽。
            rotate_status = status_code if status_code is not None else 401
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (auth refresh failed) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False

        return False, has_retried_429

    def _anthropic_messages_create(self, api_kwargs: dict):
        """使用 Anthropic Messages API 发送请求。"""
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        return self._anthropic_client.messages.create(**api_kwargs)

    def _interruptible_api_call(self, api_kwargs: dict):
        """可中断的非流式 API 调用 — 在后台线程执行, 主线程轮询中断.

        [中文] 核心 API 调用方法, 支持以下特性:

        **调用流程:**
          1. 在独立线程中发起 API 请求 (_call 闭包)
          2. 主线程每 0.3 秒检查一次 interrupt_requested 状态
          3. 收到中断时: 关闭 worker 专属的 OpenAI client, 中断 TCP 连接
          4. 连接过期检测: 无响应超时则杀死连接, 主重试循环可用 backoff/凭证轮转/fallback 重试

        **四种 API 模式路由:**
          - anthropic_messages → _anthropic_messages_create() / build_anthropic_client
          - codex_responses → _run_codex_stream() (内部处理流式)
          - bedrock_converse → boto3 converse() (非流式路径)
          - chat_completions → OpenAI client.chat.completions.create() (默认)

        **返回:** SimpleNamespace 对象, 模拟 OpenAI 响应结构 (choices[0].message 等)

        **错误处理:** 发生异常时 result["error"] 被设置, 主线程捕获后统一处理
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = self._create_request_openai_client(reason="codex_stream_request")
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                elif self.api_mode == "bedrock_converse":
                    # Bedrock uses boto3 directly — no OpenAI client needed.
                    # normalize_converse_response 生成 OpenAI 兼容的
                    # SimpleNamespace，使 Agent 循环的其余部分能将
                    # bedrock 响应当 chat_completions 响应一样处理。
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        normalize_converse_response,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    raw_response = client.converse(**api_kwargs)
                    result["response"] = normalize_converse_response(raw_response)
                else:
                    request_client_holder["client"] = self._create_request_openai_client(reason="chat_completion_request")
                    result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="request_complete")

        # ── Stale-call timeout (mirrors streaming stale detector) ────────
        # 非流式调用在完整响应就绪前不返回任何内容。
        # 无此机制时，挂起的提供商可能阻塞整个
        # httpx 超时（默认 1800s）而无任何反馈。过期
        # 检测器提前终止连接，使主重试循环能应用更丰富的
        # 恢复策略（凭据轮换、提供商回退）。
        _stale_timeout = self._compute_non_stream_stale_timeout(
            api_kwargs.get("messages", [])
        )

        _call_start = time.time()
        self._touch_activity("waiting for non-streaming API response")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        _poll_count = 0
        while t.is_alive():
            t.join(timeout=0.3)
            _poll_count += 1

            # 每约 30s 触碰活动追踪器，使网关的
            # 不活动监控器知道我们在等待响应时仍然活跃。
            if _poll_count % 100 == 0:  # 100 × 0.3s = 30s
                _elapsed = time.time() - _call_start
                self._touch_activity(
                    f"waiting for non-streaming response ({int(_elapsed)}s elapsed)"
                )

            # 过期调用检测器：若在配置的超时时间内
            # 无响应到达则终止连接。
            _elapsed = time.time() - _call_start
            if _elapsed > _stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Non-streaming API call stale for %.0fs (threshold %.0fs). "
                    "model=%s context=~%s tokens. Killing connection.",
                    _elapsed, _stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_elapsed)}s "
                    f"(non-streaming, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Aborting call."
                )
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                            timeout=get_provider_request_timeout(self.provider, self.model),
                        )
                    else:
                        rc = request_client_holder.get("client")
                        if rc is not None:
                            self._close_request_openai_client(rc, reason="stale_call_kill")
                except Exception:
                    pass
                self._touch_activity(
                    f"stale non-streaming call killed after {int(_elapsed)}s"
                )
                # 短暂等待线程注意到连接已关闭。
                t.join(timeout=2.0)
                if result["error"] is None and result["response"] is None:
                    result["error"] = TimeoutError(
                        f"Non-streaming API call timed out after {int(_elapsed)}s "
                        f"with no response (threshold: {int(_stale_timeout)}s)"
                    )
                break

            if self._interrupt_requested:
                # 强制关闭进行中的 worker 本地 HTTP 连接以停止
                # token 生成，而不污染用于后续重试的共享客户端。
                #
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                            timeout=get_provider_request_timeout(self.provider, self.model),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Unified streaming API call ─────────────────────────────────────────

    def _reset_stream_delivery_tracking(self) -> None:
        """重置当前模型响应的流式文本跟踪。

        Reset tracking for text delivered during the current model response.
        """
        self._current_streamed_assistant_text = ""

    def _record_streamed_assistant_text(self, text: str) -> None:
        """累积通过流回调发送的可见助手文本。

        Accumulate visible assistant text emitted through stream callbacks.
        """
        if isinstance(text, str) and text:
            self._current_streamed_assistant_text = (
                getattr(self, "_current_streamed_assistant_text", "") + text
            )

    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        """标准化中间可见文本，去除多余空白。"""
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _interim_content_was_streamed(self, content: str) -> bool:
        """检查指定内容是否已通过流方式发送给消费者。"""
        visible_content = self._normalize_interim_visible_text(
            self._strip_think_blocks(content or "")
        )
        if not visible_content:
            return False
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(getattr(self, "_current_streamed_assistant_text", "") or "")
        )
        return bool(streamed) and streamed == visible_content

    def _emit_interim_assistant_message(self, assistant_msg: Dict[str, Any]) -> None:
        """向 UI 层发送真实的轮中助手评论消息。

        Surface a real mid-turn assistant commentary message to the UI layer.
        """
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None or not isinstance(assistant_msg, dict):
            return
        content = assistant_msg.get("content")
        visible = self._strip_think_blocks(content or "").strip()
        if not visible or visible == "(empty)":
            return
        already_streamed = self._interim_content_was_streamed(visible)
        try:
            cb(visible, already_streamed=already_streamed)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _fire_stream_delta(self, text: str) -> None:
        """触发所有已注册的流 delta 回调（显示 + TTS）。

        Fire all registered stream delta callbacks (display + TTS).
        """
        # 若工具迭代设置了换行标志，在第一个真正的
        # 文本增量前添加单个段落换行。这在不累积
        # 多余空行（多工具迭代连续运行时）的前提下，
        # 解决了原始问题（跨工具边界的文本拼接）。
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
        callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
        delivered = False
        for cb in callbacks:
            try:
                cb(text)
                delivered = True
            except Exception:
                pass
        if delivered:
            self._record_streamed_assistant_text(text)

    def _fire_reasoning_delta(self, text: str) -> None:
        """触发推理内容回调（如已注册）。

        Fire reasoning callback if registered.
        """
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """通知显示层模型正在生成工具调用参数。

        Notify display layer that the model is generating tool call arguments.

        Fires once per tool name when the streaming response begins producing
        tool_call / tool_use tokens.  Gives the TUI a chance to show a spinner
        or status line so the user isn't staring at a frozen screen while a
        large tool payload (e.g. a 45 KB write_file) is being generated.
        """
        cb = self.tool_gen_callback
        if cb is not None:
            try:
                cb(tool_name)
            except Exception:
                pass

    def _has_stream_consumers(self) -> bool:
        """返回是否有已注册的流消费者。

        Return True if any streaming consumer is registered.
        """
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None
    ):
        """实时 token 交付的流式 API 调用变体。

        [中文] 可中断的流式 API 调用 — 运行在独立线程中，0.3s 轮询中断请求
        支持四种 API 模式:
          - chat_completions: stream=True, SSE chunk 迭代
          - anthropic_messages: client.messages.stream() 回调
          - codex_responses: _run_codex_stream 专用流
          - bedrock_converse: converse_stream() delta 回调
        每个 token 触发 stream_delta_callback 和 _stream_callback
        工具调用轮次抑制回调 — 只有纯文本最终回复才流式输出给消费者

        处理所有 API 模式:
        - chat_completions: OpenAI 兼容端点上 stream=True
        - anthropic_messages: 通过 Anthropic SDK 的 client.messages.stream()
        - codex_responses: 委托给 _run_codex_stream (已是流式)

        每个文本 token 触发 stream_delta_callback 和 _stream_callback。
        工具调用轮次抑制回调 — 仅纯文本最终响应
        才会流式输出给消费者。返回模拟非流式响应
        结构的 SimpleNamespace，使 agent 循环其余部分保持不变。

        当提供商错误表明不支持流式时，回退到 _interruptible_api_call。
        """
        if self.api_mode == "codex_responses":
            # Codex 通过 _run_codex_stream 内部流式处理。主分发
            # _interruptible_api_call 已经调用它；我们只需
            # 确保 on_first_delta 到达它。临时存储在实例上
            # 以便 _run_codex_stream 可以获取。
            self._codex_on_first_delta = on_first_delta
            try:
                return self._interruptible_api_call(api_kwargs)
            finally:
                self._codex_on_first_delta = None

        # Bedrock Converse 使用 boto3 的 converse_stream()，带实时 delta
        # 回调 — 与 Anthropic 和 chat_completions 流式体验相同。
        if self.api_mode == "bedrock_converse":
            result = {"response": None, "error": None}
            first_delta_fired = {"done": False}
            deltas_were_sent = {"yes": False}

            def _fire_first():
                if not first_delta_fired["done"] and on_first_delta:
                    first_delta_fired["done"] = True
                    try:
                        on_first_delta()
                    except Exception:
                        pass

            def _bedrock_call():
                try:
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        stream_converse_with_callbacks,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    raw_response = client.converse_stream(**api_kwargs)

                    def _on_text(text):
                        _fire_first()
                        self._fire_stream_delta(text)
                        deltas_were_sent["yes"] = True

                    def _on_tool(name):
                        _fire_first()
                        self._fire_tool_gen_started(name)

                    def _on_reasoning(text):
                        _fire_first()
                        self._fire_reasoning_delta(text)

                    result["response"] = stream_converse_with_callbacks(
                        raw_response,
                        on_text_delta=_on_text if self._has_stream_consumers() else None,
                        on_tool_start=_on_tool,
                        on_reasoning_delta=_on_reasoning if self.reasoning_callback or self.stream_delta_callback else None,
                        on_interrupt_check=lambda: self._interrupt_requested,
                    )
                except Exception as e:
                    result["error"] = e

            t = threading.Thread(target=_bedrock_call, daemon=True)
            t.start()
            while t.is_alive():
                t.join(timeout=0.3)
                if self._interrupt_requested:
                    raise InterruptedError("Agent interrupted during Bedrock API call")
            if result["error"] is not None:
                raise result["error"]
            return result["response"]

        result = {"response": None, "error": None, "partial_tool_names": []}
        request_client_holder = {"client": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # 跟踪是否触发了 delta (用于回退)
        # 最后一个真实流式 chunk 的壁钟时间戳。外层
        # 轮循环用此检测持续收到 SSE 保活 ping
        # 但无实际数据的过期连接。
        last_chunk_time = {"t": time.time()}

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """流式聊天补全响应。

            Stream a chat completions response.
            """
            import httpx as _httpx
            # 每个提供商/每个模型的 request_timeout_seconds (来自 config.yaml)
            # 如果用户设置了则优先于 HERMES_API_TIMEOUT 环境变量默认值。
            _provider_timeout_cfg = get_provider_request_timeout(self.provider, self.model)
            _base_timeout = (
                _provider_timeout_cfg
                if _provider_timeout_cfg is not None
                else float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            )
            # 读取超时: 配置同样优先。否则使用
            # HERMES_STREAM_READ_TIMEOUT (默认 120s) 用于云提供商。
            if _provider_timeout_cfg is not None:
                _stream_read_timeout = _provider_timeout_cfg
            else:
                _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
                # 本地提供商 (Ollama, llama.cpp, vLLM) 在大上下文预填时
                # 可能需要数分钟才能产生第一个 token。
                # 除非用户明确覆盖 HERMES_STREAM_READ_TIMEOUT，
                # 否则自动增加 httpx 读取超时。
                if _stream_read_timeout == 120.0 and self.base_url and is_local_endpoint(self.base_url):
                    _stream_read_timeout = _base_timeout
                    logger.debug(
                        "Local provider detected (%s) — stream read timeout raised to %.0fs",
                        self.base_url, _stream_read_timeout,
                    )
            stream_kwargs = {
                **api_kwargs,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": _httpx.Timeout(
                    connect=30.0,
                    read=_stream_read_timeout,
                    write=_base_timeout,
                    pool=30.0,
                ),
            }
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request"
            )
            # 重置过期流计时器，使检测器从此次尝试的开始
            # 时间测量，而非上次尝试的最后 chunk。
            last_chunk_time["t"] = time.time()
            self._touch_activity("waiting for provider response (streaming)")
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            # 从初始 HTTP 响应中捕获速率限制 header。
            # OpenAI SDK Stream 对象在任何 chunk 被消费前
            # 通过 .response 暴露底层 httpx 响应。
            self._capture_rate_limits(getattr(stream, "response", None))

            content_parts: list = []
            tool_calls_acc: dict = {}
            tool_gen_notified: set = set()
            # Ollama 兼容端点在并行批次中对每个工具调用重用
            # index 0，仅通过 id 区分。跟踪每个原始索引
            # 的最后见到的 id，以便检测在相同索引处
            # 开始的新工具调用并将其重定向到新槽位。
            _last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
            _active_slot_by_idx: dict = {}  # raw_index -> current slot in tool_calls_acc
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None
            for chunk in stream:
                last_chunk_time["t"] = time.time()
                self._touch_activity("receiving stream response")

                if self._interrupt_requested:
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # usage 信息在最后一帧中随空 choices 一起到达
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # 累积推理内容
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    _fire_first_delta()
                    self._fire_reasoning_delta(reasoning_text)

                # Accumulate text content — fire callback only when no tool calls
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True
                    else:
                        # 工具调用抑制常规内容流输出（避免
                        # 在工具调用旁显示啰嗦的"我将使用工具..."文本）。
                        # 但嵌入在被抑制内容中的推理标签——
                        # content should still reach the display — otherwise the
                        # 推理框仅作为响应后回退出现，
                        # 在已流出的响应之后呈现，造成困惑。
                        # 将被抑制的内容通过流
                        # 增量回调路由，使其标签提取能触发
                        # 推理显示。非推理文本则被无害地
                        # 由 CLI 的 _stream_delta 在流框已关闭时
                        # 抑制（工具边界刷新）。
                        if self.stream_delta_callback:
                            try:
                                self.stream_delta_callback(delta.content)
                                self._record_streamed_assistant_text(delta.content)
                            except Exception:
                                pass

                # Accumulate tool call deltas — notify display on first name
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        raw_idx = tc_delta.index if tc_delta.index is not None else 0
                        delta_id = tc_delta.id or ""

                        # Ollama 修复：检测新工具调用重用相同
                        # 原始索引（不同 id）并重定向到新槽位。
                        if raw_idx not in _active_slot_by_idx:
                            _active_slot_by_idx[raw_idx] = raw_idx
                        if (
                            delta_id
                            and raw_idx in _last_id_at_idx
                            and delta_id != _last_id_at_idx[raw_idx]
                        ):
                            new_slot = max(tool_calls_acc, default=-1) + 1
                            _active_slot_by_idx[raw_idx] = new_slot
                        if delta_id:
                            _last_id_at_idx[raw_idx] = delta_id
                        idx = _active_slot_by_idx[raw_idx]

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "extra_content": None,
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                # 使用赋值而非 +=。函数名是原子标识符，
                                # 在第一个帧中完整传递（OpenAI 规范）。
                                # 部分提供商（MiniMax M2.7 via NVIDIA NIM）
                                # 在每个帧中重发完整名称；拼接会产生
                                # "read_fileread_file"。赋值
                                # （匹配 OpenAI Node SDK / LiteLLM /
                                # Vercel AI 模式）对此免疫。
                                #
                                entry["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                        extra = getattr(tc_delta, "extra_content", None)
                        if extra is None and hasattr(tc_delta, "model_extra"):
                            extra = (tc_delta.model_extra or {}).get("extra_content")
                        if extra is not None:
                            if hasattr(extra, "model_dump"):
                                extra = extra.model_dump()
                            entry["extra_content"] = extra
                        # 每个工具在完整名称可用时触发一次
                        name = entry["function"]["name"]
                        if name and idx not in tool_gen_notified:
                            tool_gen_notified.add(idx)
                            _fire_first_delta()
                            self._fire_tool_gen_started(name)
                            # 记录部分工具调用名称，使外层
                            # 桩构建器能在流在此工具的参数完全
                            # 传递之前终止时，向用户显示可见警告。
                            # 无此机制时，工具调用 JSON 生成期间
                            # 的停滞会让行 ~6107 的桩
                            # 返回 `tool_calls=None`，静默
                            # 丢弃尝试的操作。
                            result["partial_tool_names"].append(name)

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # usage 在最后一帧中
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # 构建匹配非流式形态的模拟响应
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            has_truncated_tool_args = False
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    arguments = tc["function"]["arguments"]
                    if arguments and arguments.strip():
                        try:
                            json.loads(arguments)
                        except json.JSONDecodeError:
                            has_truncated_tool_args = True
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        extra_content=tc.get("extra_content"),
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=arguments,
                        ),
                    ))

            effective_finish_reason = finish_reason or "stop"
            if has_truncated_tool_args:
                effective_finish_reason = "length"

            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=mock_tool_calls,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=effective_finish_reason,
            )
            return SimpleNamespace(
                id="stream-" + str(uuid.uuid4()),
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
            )

        def _call_anthropic():
            """流式 Anthropic Messages API 响应。

            Stream an Anthropic Messages API response.

            Fires delta callbacks for real-time token delivery, but returns
            the native Anthropic Message object from get_final_message() so
            the rest of the agent loop (validation, tool extraction, etc.)
            works unchanged.
            """
            has_tool_use = False

            # 为本次尝试重置过期流计时器
            last_chunk_time["t"] = time.time()
            # 使用 Anthropic SDK 的流式上下文管理器
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    # 在每个事件上更新过期流计时器，使
                    # 外层轮询循环知道数据在流动。若无
                    # 此机制，检测器会在事件仍在活跃到达时
                    # 于 180s 后终止健康的长时运行 Opus 流
                    # （chat_completions 路径在其
                    # 数据块循环顶部已做了此操作）。
                    last_chunk_time["t"] = time.time()
                    self._touch_activity("receiving stream response")

                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True
                            tool_name = getattr(block, "name", None)
                            if tool_name:
                                _fire_first_delta()
                                self._fire_tool_gen_started(tool_name)

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                                    deltas_were_sent["yes"] = True
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    _fire_first_delta()
                                    self._fire_reasoning_delta(thinking_text)

                # 返回原生 Anthropic Message 供下游处理
                return stream.get_final_message()

        def _call():
            import httpx as _httpx

            _max_stream_retries = int(os.getenv("HERMES_STREAM_RETRIES", 2))

            try:
                for _stream_attempt in range(_max_stream_retries + 1):
                    try:
                        if self.api_mode == "anthropic_messages":
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        else:
                            result["response"] = _call_chat_completions()
                        return  # success
                    except Exception as e:
                        _is_timeout = isinstance(
                            e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                        )
                        _is_conn_err = isinstance(
                            e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                        )

                        # 若流在部分 token 已传递后终止：
                        # 通常我们不重试（用户已看到文本，
                        # 重试会重复输出）。但是：若流终止时
                        # 有工具调用正在进行中，静默中止会
                        # 完全丢弃工具调用。此时我们显示
                        # prefer to retry — the user sees a brief
                        # "reconnecting" 标记 + 重复的前言文本，
                        # 这比直接给出带有"手动重试"消息的失败
                        # 操作要好得多。仅限瞬时
                        # 连接错误（Clawdbot 风格窄门）：
                        # 此 API 调用中尚未执行任何工具，所以
                        # 静默重试在副作用方面是安全的。
                        if deltas_were_sent["yes"]:
                            _partial_tool_in_flight = bool(
                                result.get("partial_tool_names")
                            )
                            _is_sse_conn_err_preview = False
                            if not _is_timeout and not _is_conn_err:
                                from openai import APIError as _APIError
                                if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                    _err_lower_preview = str(e).lower()
                                    _SSE_PREVIEW_PHRASES = (
                                        "connection lost",
                                        "connection reset",
                                        "connection closed",
                                        "connection terminated",
                                        "network error",
                                        "network connection",
                                        "terminated",
                                        "peer closed",
                                        "broken pipe",
                                        "upstream connect error",
                                    )
                                    _is_sse_conn_err_preview = any(
                                        phrase in _err_lower_preview
                                        for phrase in _SSE_PREVIEW_PHRASES
                                    )
                            _is_transient = (
                                _is_timeout or _is_conn_err or _is_sse_conn_err_preview
                            )
                            _can_silent_retry = (
                                _partial_tool_in_flight
                                and _is_transient
                                and _stream_attempt < _max_stream_retries
                            )
                            if not _can_silent_retry:
                                # 没有工具调用在进行中（因此
                                # turn was a pure text response — current
                                # 桩+恢复文本的行为是
                                # 正确的），要么重试已耗尽，或
                                # 错误不是瞬时性的。落入
                                # 桩路径。
                                logger.warning(
                                    "Streaming failed after partial delivery, not retrying: %s", e
                                )
                                result["error"] = e
                                return
                            # 工具调用在进行中且错误是瞬时性的：
                            # 静默重试。清除每次尝试的状态，使
                            # 下一个流干净启动。触发"重新连接"
                            # 标记，让用户知道前言即将
                            # 重新流输出。
                            logger.info(
                                "Streaming attempt %s/%s died mid tool-call "
                                "(%s: %s) after user-visible text; retrying "
                                "silently to avoid losing the action. "
                                "Preamble will re-stream.",
                                _stream_attempt + 1,
                                _max_stream_retries + 1,
                                type(e).__name__,
                                e,
                            )
                            try:
                                self._fire_stream_delta(
                                    "\n\n⚠ Connection dropped mid tool-call; "
                                    "reconnecting…\n\n"
                                )
                            except Exception:
                                pass
                            # 重置流文本缓冲区，使重试的
                            # 新前言不会在 _current_streamed_assistant_text
                            # 中被双重记录（否则会
                            # 污染中间可见文本对比）。
                            try:
                                self._reset_stream_delivery_tracking()
                            except Exception:
                                pass
                            # 重置内存累加器，使下一个
                            # 尝试的帧不会拼接在已死
                            # 流的部分 JSON 上。
                            result["partial_tool_names"] = []
                            deltas_were_sent["yes"] = False
                            first_delta_fired["done"] = False
                            self._emit_status(
                                f"⚠️ Connection dropped mid tool-call "
                                f"({type(e).__name__}). Reconnecting… "
                                f"(attempt {_stream_attempt + 2}/{_max_stream_retries + 1})"
                            )
                            self._touch_activity(
                                f"stream retry {_stream_attempt + 2}/{_max_stream_retries + 1} "
                                f"mid tool-call after {type(e).__name__}"
                            )
                            stale = request_client_holder.get("client")
                            if stale is not None:
                                self._close_request_openai_client(
                                    stale, reason="stream_mid_tool_retry_cleanup"
                                )
                                request_client_holder["client"] = None
                            try:
                                self._replace_primary_openai_client(
                                    reason="stream_mid_tool_retry_pool_cleanup"
                                )
                            except Exception:
                                pass
                            self._emit_status("🔄 Reconnected — resuming…")
                            continue

                        # 代理的 SSE 错误事件（如 OpenRouter 发送
                        # {"error":{"message":"Network connection lost."}}）
                        # 被 OpenAI SDK 作为 APIError 抛出。这些
                        # semantically identical to httpx connection drops —
                        # the upstream stream died — and should be retried with
                        # 与 HTTP 错误区分：
                        # SSE 的 APIError 没有 status_code，而
                        # APIStatusError（4xx/5xx）总有 status_code。
                        _is_sse_conn_err = False
                        if not _is_timeout and not _is_conn_err:
                            from openai import APIError as _APIError
                            if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                _err_lower_sse = str(e).lower()
                                _SSE_CONN_PHRASES = (
                                    "connection lost",
                                    "connection reset",
                                    "connection closed",
                                    "connection terminated",
                                    "network error",
                                    "network connection",
                                    "terminated",
                                    "peer closed",
                                    "broken pipe",
                                    "upstream connect error",
                                )
                                _is_sse_conn_err = any(
                                    phrase in _err_lower_sse
                                    for phrase in _SSE_CONN_PHRASES
                                )

                        if _is_timeout or _is_conn_err or _is_sse_conn_err:
                            # 瞬时网络/超时错误。先用新鲜连接重试
                            # 流式请求。
                            if _stream_attempt < _max_stream_retries:
                                logger.info(
                                    "Streaming attempt %s/%s failed (%s: %s), "
                                    "retrying with fresh connection...",
                                    _stream_attempt + 1,
                                    _max_stream_retries + 1,
                                    type(e).__name__,
                                    e,
                                )
                                self._emit_status(
                                    f"⚠️ Connection to provider dropped "
                                    f"({type(e).__name__}). Reconnecting… "
                                    f"(attempt {_stream_attempt + 2}/{_max_stream_retries + 1})"
                                )
                                self._touch_activity(
                                    f"stream retry {_stream_attempt + 2}/{_max_stream_retries + 1} "
                                    f"after {type(e).__name__}"
                                )
                                # 重试前关闭过期的请求客户端
                                stale = request_client_holder.get("client")
                                if stale is not None:
                                    self._close_request_openai_client(
                                        stale, reason="stream_retry_cleanup"
                                    )
                                    request_client_holder["client"] = None
                                # 同时重建主客户端以清除
                                # 池中的任何已死连接。
                                try:
                                    self._replace_primary_openai_client(
                                        reason="stream_retry_pool_cleanup"
                                    )
                                except Exception:
                                    pass
                                self._emit_status("🔄 Reconnected — resuming…")
                                continue
                            self._emit_status(
                                "❌ Connection to provider failed after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                            logger.warning(
                                "Streaming exhausted %s retries on transient error: %s",
                                _max_stream_retries + 1,
                                e,
                            )
                        else:
                            _err_lower = str(e).lower()
                            _is_stream_unsupported = (
                                "stream" in _err_lower
                                and "not supported" in _err_lower
                            )
                            if _is_stream_unsupported:
                                self._disable_streaming = True
                                self._safe_print(
                                    "\n⚠  Streaming is not supported for this "
                                    "model/provider. Switching to non-streaming.\n"
                                    "   To avoid this delay, set display.streaming: false "
                                    "in config.yaml\n"
                                )
                            logger.info(
                                "Streaming failed before delivery: %s",
                                e,
                            )

                        # 将错误传播到主重试循环，而非
                        # 内联回退到非流式。主循环有
                        # 更丰富的恢复机制：凭据轮换、提供商回退、
                        # 退避策略，以及——对于"不支持流式"——将在下次尝试
                        # 通过 _disable_streaming 切换到非流式模式。
                        result["error"] = e
                        return
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
        # 本地提供商（Ollama、oMLX、llama-cpp）在大上下文上
        # 的预填充可能需要 300+ 秒。除非用户显式设置了
        # HERMES_STREAM_STALE_TIMEOUT，否则禁用过期检测器。
        if _stream_stale_timeout_base == 180.0 and self.base_url and is_local_endpoint(self.base_url):
            _stream_stale_timeout = float("inf")
            logger.debug("Local provider detected (%s) — stale stream timeout disabled", self.base_url)
        else:
            # 对大上下文缩放过期超时：慢模型（如 Opus）
            # 在上下文很大时可能合法地思考数分钟才产生第一个
            # token。无此机制时，过期检测器会在模型的思考阶段
            # 终止健康连接，产生虚假的
            # RemoteProtocolError（"peer closed connection"）。
            _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
            if _est_tokens > 100_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
            elif _est_tokens > 50_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
            else:
                _stream_stale_timeout = _stream_stale_timeout_base

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        _last_heartbeat = time.time()
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        while t.is_alive():
            t.join(timeout=0.3)

            # 定期心跳：触碰 Agent 的活动追踪器，使
            # 网关的不活动监控器知道我们在等待流数据块时
            # 仍然活跃。无此机制时，长时间思考暂停（如
            # 推理模型）或本地提供商的慢预填充（Ollama）
            # 会触发虚假的不活动超时。_call 线程在
            # 每个数据块上触碰活动，但 API 调用开始
            # and first chunk can exceed the gateway timeout — especially
            # 在过期流超时被禁用时使用（本地提供商）。
            _hb_now = time.time()
            if _hb_now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                _last_heartbeat = _hb_now
                _waiting_secs = int(_hb_now - last_chunk_time["t"])
                self._touch_activity(
                    f"waiting for stream response ({_waiting_secs}s, no chunks yet)"
                )

            # 检测过期流：由 SSE ping 保持活跃但
            # 不传递真实数据块的连接。终止客户端以便
            # 内层重试循环能启动新鲜连接。
            _stale_elapsed = time.time() - last_chunk_time["t"]
            if _stale_elapsed > _stream_stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                    "model=%s context=~%s tokens. Killing connection.",
                    _stale_elapsed, _stream_stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                    f"(model: {api_kwargs.get('model', 'unknown')}, "
                    f"context: ~{_est_ctx:,} tokens). "
                    f"Reconnecting..."
                )
                try:
                    rc = request_client_holder.get("client")
                    if rc is not None:
                        self._close_request_openai_client(rc, reason="stale_stream_kill")
                except Exception:
                    pass
                # Rebuild the primary client too — its connection pool
                # 可能持有同一提供商宕机的已死 socket。
                try:
                    self._replace_primary_openai_client(reason="stale_stream_pool_cleanup")
                except Exception:
                    pass
                # 重置计时器，避免在线程处理关闭时
                # 重复终止。
                last_chunk_time["t"] = time.time()
                self._touch_activity(
                    f"stale stream detected after {int(_stale_elapsed)}s, reconnecting"
                )

            if self._interrupt_requested:
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                            timeout=get_provider_request_timeout(self.provider, self.model),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during streaming API call")
        if result["error"] is not None:
            if deltas_were_sent["yes"]:
                # 流式在部分 token 已传递到平台后失败。
                # 重新抛出会让外层重试循环进行新的
                # API 调用，创建重复消息。返回部分
                # "stop" 响应，使外层循环将此轮视为
                # 已完成（不重试，不回退）。
                # 恢复已流式传输给用户的任何内容。
                # _current_streamed_assistant_text 累积通过
                # _fire_stream_delta 触发的文本，因此精确包含
                # 用户在连接断开前看到的内容。
                _partial_text = (
                    getattr(self, "_current_streamed_assistant_text", "") or ""
                ).strip() or None

                # 若流在模型发出工具调用时中断，
                # 下方的桩会静默设置 `tool_calls=None`，
                # agent loop will treat the turn as complete — the attempted
                # 操作丢失且无用户可见信号。追加
                # 人类可见的警告到桩内容中，使（a）用户
                # 知道发生了什么，且（b）下一轮的模型能从
                # 对话历史中看到尝试了什么并重试。
                _partial_names = list(result.get("partial_tool_names") or [])
                if _partial_names:
                    _name_str = ", ".join(_partial_names[:3])
                    if len(_partial_names) > 3:
                        _name_str += f", +{len(_partial_names) - 3} more"
                    _warn = (
                        f"\n\n⚠ Stream stalled mid tool-call "
                        f"({_name_str}); the action was not executed. "
                        f"Ask me to retry if you want to continue."
                    )
                    _partial_text = (_partial_text or "") + _warn
                    # 同时作为流增量触发，让用户立即看到，
                    # 而非仅在持久化的记录中。
                    try:
                        self._fire_stream_delta(_warn)
                    except Exception:
                        pass
                    logger.warning(
                        "Partial stream dropped tool call(s) %s after %s chars "
                        "of text; surfaced warning to user: %s",
                        _partial_names, len(_partial_text or ""), result["error"],
                    )
                else:
                    logger.warning(
                        "Partial stream delivered before error; returning stub "
                        "response with %s chars of recovered content to prevent "
                        "duplicate messages: %s",
                        len(_partial_text or ""),
                        result["error"],
                    )
                _stub_msg = SimpleNamespace(
                    role="assistant", content=_partial_text, tool_calls=None,
                    reasoning_content=None,
                )
                return SimpleNamespace(
                    id="partial-stream-stub",
                    model=getattr(self, "model", "unknown"),
                    choices=[SimpleNamespace(
                        index=0, message=_stub_msg, finish_reason="stop",
                    )],
                    usage=None,
                )
            raise result["error"]
        return result["response"]

    # ── Provider fallback ──────────────────────────────────────────────────

    def _try_activate_fallback(self) -> bool:
        """切换到 fallback 链中的下一个模型/提供商。

        Switch to the next fallback model/provider in the chain.

        Called when the current model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  Advances through the chain on
        each call; returns False when exhausted.

        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.
        """
        if self._fallback_index >= len(self._fallback_chain):
            return False

        fb = self._fallback_chain[self._fallback_index]
        self._fallback_index += 1
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return self._try_activate_fallback()  # skip invalid, try next

        # 使用集中式路由器构建客户端。
        # raw_codex=True，因为主 Agent 需要直接访问 responses.stream()
        # 用于 Codex 提供商。
        try:
            from agent.auxiliary_client import resolve_provider_client
            # 从回退配置传入 base_url 和 api_key，使自定义
            # 端点（如 Ollama Cloud）能正确解析，而非
            # 回退到 OpenRouter 默认值。
            fb_base_url_hint = (fb.get("base_url") or "").strip() or None
            fb_api_key_hint = (fb.get("api_key") or "").strip() or None
            if not fb_api_key_hint:
                fb_key_env = (fb.get("key_env") or "").strip()
                if fb_key_env:
                    fb_api_key_hint = os.getenv(fb_key_env, "").strip() or None
            # 对 Ollama Cloud 端点，从环境变量取 OLLAMA_API_KEY，
            # 当回退配置中无显式 key 时。主机匹配
            # (not substring) — see GHSA-76xc-57q6-vm5m.
            if fb_base_url_hint and base_url_host_matches(fb_base_url_hint, "ollama.com") and not fb_api_key_hint:
                fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
            fb_client, _resolved_fb_model = resolve_provider_client(
                fb_provider, model=fb_model, raw_codex=True,
                explicit_base_url=fb_base_url_hint,
                explicit_api_key=fb_api_key_hint)
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured",
                    fb_provider)
                return self._try_activate_fallback()  # try next in chain
            try:
                from hermes_cli.model_normalize import normalize_model_for_provider

                fb_model = normalize_model_for_provider(fb_model, fb_provider)
            except Exception:
                pass

            # 从提供商/base URL/模型确定 api_mode
            fb_api_mode = "chat_completions"
            fb_base_url = str(fb_client.base_url)
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith("/anthropic"):
                fb_api_mode = "anthropic_messages"
            elif self._is_direct_openai_url(fb_base_url):
                fb_api_mode = "codex_responses"
            elif self._provider_model_requires_responses_api(
                fb_model,
                provider=fb_provider,
            ):
                # GPT-5.x 模型通常需要 Responses API，但保留
                # 提供商特定例外，如 Copilot gpt-5-mini 使用
                # chat completions。
                fb_api_mode = "codex_responses"
            elif fb_provider == "bedrock" or (
                base_url_hostname(fb_base_url).startswith("bedrock-runtime.")
                and base_url_host_matches(fb_base_url, "amazonaws.com")
            ):
                fb_api_mode = "bedrock_converse"

            old_model = self.model
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            if hasattr(self, "_transport_cache"):
                self._transport_cache.clear()
            self._fallback_activated = True

            # 遵守每提供商/每模型的 request_timeout_seconds 用于
            # 回退目标（与主客户端使用的同一配置）。None = 用
            # SDK 默认值。
            _fb_timeout = get_provider_request_timeout(fb_provider, fb_model)

            if fb_api_mode == "anthropic_messages":
                # 构建原生 Anthropic 客户端而非使用 OpenAI 客户端
                from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token
                effective_key = (fb_client.api_key or resolve_anthropic_token() or "") if fb_provider == "anthropic" else (fb_client.api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = fb_base_url
                self._anthropic_client = build_anthropic_client(
                    effective_key, self._anthropic_base_url, timeout=_fb_timeout,
                )
                self._is_anthropic_oauth = _is_oauth_token(effective_key) if fb_provider == "anthropic" else False
                self.client = None
                self._client_kwargs = {}
            else:
                # 就地交换 OpenAI 客户端和配置
                self.api_key = fb_client.api_key
                self.client = fb_client
                # 保留提供商特定的 header，
                # resolve_provider_client() 可能已通过
                # default_headers kwarg 写入 fb_client。OpenAI
                # SDK 将这些存储在 _custom_headers 中。不处理的话，
                # 后续的请求客户端重建（通过
                # _create_request_openai_client）会丢失这些 header，
                # 导致需要 User-Agent 标记的提供商（如 Kimi Coding）
                # 返回 403。
                fb_headers = getattr(fb_client, "_custom_headers", None)
                if not fb_headers:
                    fb_headers = getattr(fb_client, "default_headers", None)
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                    **({"default_headers": dict(fb_headers)} if fb_headers else {}),
                }
                if _fb_timeout is not None:
                    self._client_kwargs["timeout"] = _fb_timeout
                    # 重建共享 OpenAI 客户端，使配置的
                    # 超时在下一个回退请求上立即生效，
                    # 而非等待后续凭据轮换重建。
                    self._replace_primary_openai_client(reason="fallback_timeout_apply")

            # 为新提供商/模型重新评估提示缓存
            self._use_prompt_caching, self._use_native_cache_layout = (
                self._anthropic_prompt_cache_policy(
                    provider=fb_provider,
                    base_url=fb_base_url,
                    api_mode=fb_api_mode,
                    model=fb_model,
                )
            )

            # 为回退模型更新上下文压缩器限制。
            # 不更新的话，压缩决策会使用主模型的
            # 上下文窗口（如 200K），而非回退模型的（如 32K），
            # 导致超大会话溢出回退模型。
            if hasattr(self, 'context_compressor') and self.context_compressor:
                from agent.model_metadata import get_model_context_length
                fb_context_length = get_model_context_length(
                    self.model, base_url=self.base_url,
                    api_key=self.api_key, provider=self.provider,
                )
                self.context_compressor.update_model(
                    model=self.model,
                    context_length=fb_context_length,
                    base_url=self.base_url,
                    api_key=getattr(self, "api_key", ""),
                    provider=self.provider,
                )

            self._emit_status(
                f"🔄 Primary model failed — switching to fallback: "
                f"{fb_model} via {fb_provider}"
            )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model, fb_model, fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback %s: %s", fb_model, e)
            return self._try_activate_fallback()  # try next in chain

    # ── Per-turn primary restoration ─────────────────────────────────────

    def _restore_primary_runtime(self) -> bool:
        """在新一轮开始时恢复主运行时配置。

        Restore the primary runtime at the start of a new turn.

        In long-lived CLI sessions a single AIAgent instance spans multiple
        turns.  Without restoration, one transient failure pins the session
        to the fallback provider for every subsequent turn.  Calling this at
        the top of ``run_conversation()`` makes fallback turn-scoped.

        The gateway caches agents across messages (``_agent_cache`` in
        ``gateway/run.py``), so this restoration IS needed there too.
        """
        if not self._fallback_activated:
            return False

        rt = self._primary_runtime
        try:
            # ── Core runtime state ──
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]           # setter updates _base_url_lower
            self.api_mode = rt["api_mode"]
            if hasattr(self, "_transport_cache"):
                self._transport_cache.clear()
            self.api_key = rt["api_key"]
            self._client_kwargs = dict(rt["client_kwargs"])
            self._use_prompt_caching = rt["use_prompt_caching"]
            # 当恢复的快照早于原生 vs 代理
            # 分离时默认为原生布局（此 PR 之前保存的旧会话）。
            self._use_native_cache_layout = rt.get(
                "use_native_cache_layout",
                self.api_mode == "anthropic_messages" and self.provider == "anthropic",
            )

            # ── Rebuild client for the primary provider ──
            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                    timeout=get_provider_request_timeout(self.provider, self.model),
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="restore_primary",
                    shared=True,
                )

            # ── Restore context engine state ──
            cc = self.context_compressor
            cc.update_model(
                model=rt["compressor_model"],
                context_length=rt["compressor_context_length"],
                base_url=rt["compressor_base_url"],
                api_key=rt["compressor_api_key"],
                provider=rt["compressor_provider"],
            )

            # ── Reset fallback chain for the new turn ──
            self._fallback_activated = False
            self._fallback_index = 0

            logging.info(
                "Primary runtime restored for new turn: %s (%s)",
                self.model, self.provider,
            )
            return True
        except Exception as e:
            logging.warning("Failed to restore primary runtime: %s", e)
            return False

    # 哪些错误类型表示值得用重建客户端/连接池
    # 再尝试一次的瞬时传输故障。
    _TRANSIENT_TRANSPORT_ERRORS = frozenset({
        "ReadTimeout", "ConnectTimeout", "PoolTimeout",
        "ConnectError", "RemoteProtocolError",
        "APIConnectionError", "APITimeoutError",
    })

    def _try_recover_primary_transport(
        self, api_error: Exception, *, retry_count: int, max_retries: int,
    ) -> bool:
        """对临时传输故障尝试一次额外的主提供商恢复。

        Attempt one extra primary-provider recovery cycle for transient transport failures.
        """

        """

        After ``max_retries`` exhaust, rebuild the primary client (clearing
        stale connection pools) and give it one more attempt before falling
        back.  This is most useful for direct endpoints (custom, Z.AI,
        Anthropic, OpenAI, local models) where a TCP-level hiccup does not
        mean the provider is down.

        Skipped for proxy/aggregator providers (OpenRouter, Nous) which
        already manage connection pools and retries server-side — if our
        retries through them are exhausted, one more rebuilt client won't help.
        """
        if self._fallback_activated:
            return False

        # 仅用于瞬时传输错误
        error_type = type(api_error).__name__
        if error_type not in self._TRANSIENT_TRANSPORT_ERRORS:
            return False

        # Skip for aggregator providers — they manage their own retry infra
        if self._is_openrouter_url():
            return False
        provider_lower = (self.provider or "").strip().lower()
        if provider_lower in ("nous", "nous-research"):
            return False

        try:
            # 关闭现有客户端以释放过期连接
            if getattr(self, "client", None) is not None:
                try:
                    self._close_openai_client(
                        self.client, reason="primary_recovery", shared=True,
                    )
                except Exception:
                    pass

            # 从主快照重建
            rt = self._primary_runtime
            self._client_kwargs = dict(rt["client_kwargs"])
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]
            self.api_mode = rt["api_mode"]
            if hasattr(self, "_transport_cache"):
                self._transport_cache.clear()
            self.api_key = rt["api_key"]

            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                    timeout=get_provider_request_timeout(self.provider, self.model),
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="primary_recovery",
                    shared=True,
                )

            wait_time = min(3 + retry_count, 8)
            self._vprint(
                f"{self.log_prefix}🔁 Transient {error_type} on {self.provider} — "
                f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
                force=True,
            )
            time.sleep(wait_time)
            return True
        except Exception as e:
            logging.warning("Primary transport recovery failed: %s", e)
            return False

    # ── End provider fallback ──────────────────────────────────────────────

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        """检查内容是否包含图片部分。"""
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        """将 data URL 图片物化为临时文件，返回路径。"""
        header, _, data = str(image_url or "").partition(",")
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        with tmp:
            tmp.write(base64.b64decode(data))
        path = Path(tmp.name)
        return str(path), path

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        """将图片转换为文字描述（Anthropic 回退路径）。"""
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, "user")
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, UI, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        vision_source = str(image_url or "")
        cleanup_path: Optional[Path] = None
        if vision_source.startswith("data:"):
            vision_source, cleanup_path = self._materialize_data_url_for_vision(vision_source)

        description = ""
        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(
                vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt)
            )
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"Image analysis failed: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        if not description:
            description = "Image analysis failed."

        note = f"[The {role_label} attached an image. Here's what it contains:\n{description}]"
        if vision_source and not str(image_url or "").startswith("data:"):
            note += (
                f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"
            )

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        """预处理 Anthropic 格式的图片内容，转为文本描述。"""
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
                if image_url:
                    image_notes.append(self._describe_image_for_anthropic_fallback(image_url, role))
                else:
                    image_notes.append("[An image was attached but no image source was available.]")
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return "[A multimodal message was converted to text for Anthropic compatibility.]"

    def _get_transport(self, api_mode: str = None):
        """返回给定（或当前）api_mode 的缓存 transport。

        Return the cached transport for the given (or current) api_mode.

        Lazy-initializes on first call per api_mode. Returns None if no
        transport is registered for the mode.
        """
        mode = api_mode or self.api_mode
        cache = getattr(self, "_transport_cache", None)
        if cache is None:
            cache = {}
            self._transport_cache = cache
        t = cache.get(mode)
        if t is None:
            from agent.transports import get_transport
            t = get_transport(mode)
            cache[mode] = t
        return t

    @staticmethod
    def _nr_to_assistant_message(nr):
        """将 NormalizedResponse 转换为下游期望的 SimpleNamespace 格式。

        Convert a NormalizedResponse to the SimpleNamespace shape downstream expects.

        This is the single back-compat shim between the transport layer
        (NormalizedResponse) and the agent loop (SimpleNamespace with
        .content, .tool_calls, .reasoning, .reasoning_content,
        .reasoning_details, .codex_reasoning_items, and per-tool-call
        .call_id / .response_item_id).

        TODO: Remove when downstream code reads NormalizedResponse directly.
        """
        tc_list = None
        if nr.tool_calls:
            tc_list = []
            for tc in nr.tool_calls:
                tc_ns = SimpleNamespace(
                    id=tc.id,
                    type="function",
                    function=SimpleNamespace(name=tc.name, arguments=tc.arguments),
                )
                if tc.provider_data:
                    for key in ("call_id", "response_item_id"):
                        if tc.provider_data.get(key):
                            setattr(tc_ns, key, tc.provider_data[key])
                tc_list.append(tc_ns)
        pd = nr.provider_data or {}
        return SimpleNamespace(
            content=nr.content,
            tool_calls=tc_list or None,
            reasoning=nr.reasoning,
            reasoning_content=pd.get("reasoning_content"),
            reasoning_details=pd.get("reasoning_details"),
            codex_reasoning_items=pd.get("codex_reasoning_items"),
        )

    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        """预处理 Anthropic 消息中的图片，转换为文字描述后发送。"""
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _anthropic_preserve_dots(self) -> bool:
        """当使用保留模型名中点的 anthropic 兼容端点时返回 True。

        True when using an anthropic-compatible endpoint that preserves dots in model names.
        Alibaba/DashScope keeps dots (e.g. qwen3.5-plus).
        MiniMax keeps dots (e.g. MiniMax-M2.7).
        OpenCode Go/Zen keeps dots for non-Claude models (e.g. minimax-m2.5-free).
        ZAI/Zhipu keeps dots (e.g. glm-4.7, glm-5.1).
        AWS Bedrock uses dotted inference-profile IDs
        (e.g. ``global.anthropic.claude-opus-4-7``,
        ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``) and rejects
        the hyphenated form with
        ``HTTP 400 The provided model identifier is invalid``.
        Regression for #11976; mirrors the opencode-go fix for #5211
        (commit f77be22c), which extended this same allowlist."""
        if (getattr(self, "provider", "") or "").lower() in {
            "alibaba", "minimax", "minimax-cn",
            "opencode-go", "opencode-zen",
            "zai", "bedrock",
        }:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return (
            "dashscope" in base
            or "aliyuncs" in base
            or "minimax" in base
            or "opencode.ai/zen/" in base
            or "bigmodel.cn" in base
            # AWS Bedrock runtime endpoints — defense-in-depth when
            # ``provider`` 未设置但 ``base_url`` 仍指向 Bedrock。
            or "bedrock-runtime." in base
        )

    def _is_qwen_portal(self) -> bool:
        """当 base URL 指向 Qwen Portal 时返回 True。

        Return True when the base URL targets Qwen Portal.
        """
        return base_url_host_matches(self._base_url_lower, "portal.qwen.ai")

    def _qwen_prepare_chat_messages(self, api_messages: list) -> list:
        """准备发送给 Qwen Portal 的聊天消息格式。"""
        prepared = copy.deepcopy(api_messages)
        if not prepared:
            return prepared

        for msg in prepared:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # 标准化：将裸字符串转为 text dict，dict 保持原样。
                # deepcopy 已创建独立副本，无需 dict()。
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        # 在 system 消息的最后部分注入 cache_control。
        for msg in prepared:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return prepared

    def _qwen_prepare_chat_messages_inplace(self, messages: list) -> None:
        """原地变体 — 修改已复制完毕的消息列表。

        In-place variant — mutates an already-copied message list.
        """
        if not messages:
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """为当前 API 模式构建关键字参数字典。

        [中文] 构建提供商特定的 API 请求参数 — 委托给对应 transport 的 build_kwargs()
        处理提供商特定关注点: OpenRouter provider 偏好、Qwen 会话元数据、
        温度覆盖、推理配置等。四种 API 模式各走各的分支。
        """
        if self.api_mode == "anthropic_messages":
            _transport = self._get_transport()
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            ctx_len = getattr(self, "context_compressor", None)
            ctx_len = ctx_len.context_length if ctx_len else None
            ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
            if ephemeral_out is not None:
                self._ephemeral_max_output_tokens = None  # consume immediately
            return _transport.build_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=ephemeral_out if ephemeral_out is not None else self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=self._is_anthropic_oauth,
                preserve_dots=self._anthropic_preserve_dots(),
                context_length=ctx_len,
                base_url=getattr(self, "_anthropic_base_url", None),
                fast_mode=(self.request_overrides or {}).get("speed") == "fast",
            )

        # AWS Bedrock 原生 Converse API — 完全绕过 OpenAI client。
        # adapter 直接处理消息/工具转换和 boto3 调用。
        if self.api_mode == "bedrock_converse":
            _bt = self._get_transport()
            region = getattr(self, "_bedrock_region", None) or "us-east-1"
            guardrail = getattr(self, "_bedrock_guardrail_config", None)
            return _bt.build_kwargs(
                model=self.model,
                messages=api_messages,
                tools=self.tools,
                max_tokens=self.max_tokens or 4096,
                region=region,
                guardrail_config=guardrail,
            )

        if self.api_mode == "codex_responses":
            _ct = self._get_transport()
            is_github_responses = (
                base_url_host_matches(self.base_url, "models.github.ai")
                or base_url_host_matches(self.base_url, "api.githubcopilot.com")
            )
            is_codex_backend = (
                self.provider == "openai-codex"
                or (
                    self._base_url_hostname == "chatgpt.com"
                    and "/backend-api/codex" in self._base_url_lower
                )
            )
            is_xai_responses = self.provider == "xai" or self._base_url_hostname == "api.x.ai"
            return _ct.build_kwargs(
                model=self.model,
                messages=api_messages,
                tools=self.tools,
                reasoning_config=self.reasoning_config,
                session_id=getattr(self, "session_id", None),
                max_tokens=self.max_tokens,
                request_overrides=self.request_overrides,
                is_github_responses=is_github_responses,
                is_codex_backend=is_codex_backend,
                is_xai_responses=is_xai_responses,
                github_reasoning_extra=self._github_models_reasoning_extra_body() if is_github_responses else None,
            )

        # ── chat_completions (默认) ─────────────────────────────────────
        _ct = self._get_transport()

        # 提供商检测标志
        _is_qwen = self._is_qwen_portal()
        _is_or = self._is_openrouter_url()
        _is_gh = (
            base_url_host_matches(self._base_url_lower, "models.github.ai")
            or base_url_host_matches(self._base_url_lower, "api.githubcopilot.com")
        )
        _is_nous = "nousresearch" in self._base_url_lower
        _is_nvidia = "integrate.api.nvidia.com" in self._base_url_lower
        _is_kimi = (
            base_url_host_matches(self.base_url, "api.kimi.com")
            or base_url_host_matches(self.base_url, "moonshot.ai")
            or base_url_host_matches(self.base_url, "moonshot.cn")
        )

        # Temperature: _fixed_temperature_for_model 可能返回 OMIT_TEMPERATURE
        # 哨关值 (完全省略 temperature)、数字覆盖值或 None。
        try:
            from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
            _ft = _fixed_temperature_for_model(self.model, self.base_url)
            _omit_temp = _ft is OMIT_TEMPERATURE
            _fixed_temp = _ft if not _omit_temp else None
        except Exception:
            _omit_temp = False
            _fixed_temp = None

        # 提供商偏好设置 (OpenRouter 特定)
        _prefs: Dict[str, Any] = {}
        if self.providers_allowed:
            _prefs["only"] = self.providers_allowed
        if self.providers_ignored:
            _prefs["ignore"] = self.providers_ignored
        if self.providers_order:
            _prefs["order"] = self.providers_order
        if self.provider_sort:
            _prefs["sort"] = self.provider_sort
        if self.provider_require_parameters:
            _prefs["require_parameters"] = True
        if self.provider_data_collection:
            _prefs["data_collection"] = self.provider_data_collection

        # OpenRouter/Nous 上 Claude 的 Anthropic 最大输出长度
        _ant_max = None
        if (_is_or or _is_nous) and "claude" in (self.model or "").lower():
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                _ant_max = _get_anthropic_max_output(self.model)
            except Exception:
                pass  # fail open — let the proxy pick its default

        # Qwen 会话元数据在此预计算 (promptId 每次调用随机生成)
        _qwen_meta = None
        if _is_qwen:
            _qwen_meta = {
                "sessionId": self.session_id or "hermes",
                "promptId": str(uuid.uuid4()),
            }

        # 临时最大输出覆盖 — 立即消耗，以免下一轮继承。
        _ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
        if _ephemeral_out is not None:
            self._ephemeral_max_output_tokens = None

        return _ct.build_kwargs(
            model=self.model,
            messages=api_messages,
            tools=self.tools,
            timeout=self._resolved_api_call_timeout(),
            max_tokens=self.max_tokens,
            ephemeral_max_output_tokens=_ephemeral_out,
            max_tokens_param_fn=self._max_tokens_param,
            reasoning_config=self.reasoning_config,
            request_overrides=self.request_overrides,
            session_id=getattr(self, "session_id", None),
            model_lower=(self.model or "").lower(),
            is_openrouter=_is_or,
            is_nous=_is_nous,
            is_qwen_portal=_is_qwen,
            is_github_models=_is_gh,
            is_nvidia_nim=_is_nvidia,
            is_kimi=_is_kimi,
            is_custom_provider=self.provider == "custom",
            ollama_num_ctx=self._ollama_num_ctx,
            provider_preferences=_prefs or None,
            qwen_prepare_fn=self._qwen_prepare_chat_messages if _is_qwen else None,
            qwen_prepare_inplace_fn=self._qwen_prepare_chat_messages_inplace if _is_qwen else None,
            qwen_session_metadata=_qwen_meta,
            fixed_temperature=_fixed_temp,
            omit_temperature=_omit_temp,
            supports_reasoning=self._supports_reasoning_extra_body(),
            github_reasoning_extra=self._github_models_reasoning_extra_body() if _is_gh else None,
            anthropic_max_output=_ant_max,
        )

    def _supports_reasoning_extra_body(self) -> bool:
        """当推理 extra_body 对此路由/模型安全时返回 True。

        Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body fields to upstream providers.
        Some providers/routes reject `reasoning` with 400s, so gate it to
        known reasoning-capable model families and direct Nous Portal.
        """
        if base_url_host_matches(self._base_url_lower, "nousresearch.com"):
            return True
        if base_url_host_matches(self._base_url_lower, "ai-gateway.vercel.sh"):
            return True
        if (
            base_url_host_matches(self._base_url_lower, "models.github.ai")
            or base_url_host_matches(self._base_url_lower, "api.githubcopilot.com")
        ):
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if "openrouter" not in self._base_url_lower:
            return False
        if "api.mistral.ai" in self._base_url_lower:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """格式化 GitHub Models/OpenAI 兼容路由的推理负载。

        Format reasoning payload for GitHub Models/OpenAI-compatible routes.
        """
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """从 API 响应消息构建标准化的助手消息字典。

        Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        reasoning_text = self._extract_reasoning(assistant_message)
        _from_structured = bool(reasoning_text)

        # 回退：当无结构化推理字段时从内容中提取
        # 内联 <think> 块（部分模型/提供商将思考直接
        # 嵌入内容而非返回独立的 API 字段）。
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

        if reasoning_text and self.reasoning_callback:
            # Skip callback when streaming is active — reasoning was already
            # 在流中通过以下两种路径之一显示：
            #   (a) _fire_reasoning_delta（结构化 reasoning_content 增量）
            #   (b) _stream_delta 标签提取（<think>/<REASONING_SCRATCHPAD>）
            # 当流未激活时，始终触发，使非流模式
            # （网关、批量、静默）仍能获得推理。
            # 流期间未显示的任何推理由 CLI 的
            # 响应后显示回退捕获（cli.py _reasoning_shown_this_turn）。
            if not self.stream_delta_callback and not self._stream_callback:
                try:
                    self.reasoning_callback(reasoning_text)
                except Exception:
                    pass

        # Sanitize surrogates from API response — some models (e.g. Kimi/GLM via Ollama)
        # 可能返回无效 surrogate 码点，持久化时崩溃 json.dumps()。
        _raw_content = assistant_message.content or ""
        _san_content = _sanitize_surrogates(_raw_content)
        if reasoning_text:
            reasoning_text = _sanitize_surrogates(reasoning_text)

        # Strip inline reasoning tags (<think>…</think> etc.) from the stored
        # assistant 内容。推理已被捕获到
        # 上方的 ``reasoning_text``（来自结构化字段或
        # 内联块回退），因此内容中的原始标签是冗余的。
        # 保留它们会导致推理泄露到消息平台
        # （#8878、#9568），增加后续轮次的上下文大小
        # （#9306 在实际 MiniMax 会话中观察到 16% 的内容缩减），
        # 并污染生成的会话标题。在存储边界处
        # 一次剥离，为所有下游消费者清洁内容：
        # API 重放、会话记录、网关传递、CLI 显示、
        # 压缩、标题生成。
        if isinstance(_san_content, str) and _san_content:
            _san_content = self._strip_think_blocks(_san_content).strip()

        msg = {
            "role": "assistant",
            "content": _san_content,
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        if hasattr(assistant_message, "reasoning_content"):
            raw_reasoning_content = getattr(assistant_message, "reasoning_content", None)
            if raw_reasoning_content is not None:
                msg["reasoning_content"] = _sanitize_surrogates(raw_reasoning_content)

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            # 原样传递 reasoning_details，使提供商（OpenRouter、
            # Anthropic、OpenAI）能在轮次间保持推理连续性。
            # 每个提供商可能包含不透明字段（signature、encrypted_content），
            # 必须精确保留。
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API：保留加密的推理条目用于
        # 多轮连续性。这些在下一轮作为输入重放。
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        _fn = getattr(tool_call, "function", None)
                        _fn_name = getattr(_fn, "name", "") if _fn else ""
                        _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                        call_id = self._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if not isinstance(response_item_id, str) or not response_item_id.strip():
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                }
                # 保留 extra_content（如 Gemini thought_signature），使其
                # 在后续 API 调用中发回。若无此，Gemini 3
                # 思考模型会以 400 错误拒绝请求。
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg

    def _copy_reasoning_content_for_api(self, source_msg: dict, api_msg: dict) -> None:
        """将提供商层面的推理字段复制到 API 重放消息上。

        Copy provider-facing reasoning fields onto an API replay message.
        """
        if source_msg.get("role") != "assistant":
            return

        explicit_reasoning = source_msg.get("reasoning_content")
        if isinstance(explicit_reasoning, str):
            api_msg["reasoning_content"] = explicit_reasoning
            return

        normalized_reasoning = source_msg.get("reasoning")
        if isinstance(normalized_reasoning, str) and normalized_reasoning:
            api_msg["reasoning_content"] = normalized_reasoning
            return

        kimi_requires_reasoning = (
            self.provider in {"kimi-coding", "kimi-coding-cn"}
            or base_url_host_matches(self.base_url, "api.kimi.com")
            or base_url_host_matches(self.base_url, "moonshot.ai")
            or base_url_host_matches(self.base_url, "moonshot.cn")
        )
        if kimi_requires_reasoning and source_msg.get("tool_calls"):
            api_msg["reasoning_content"] = ""

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """清理 tool_calls 中的 Codex Responses API 字段（用于严格提供商）。

        Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
        validate the Chat Completions schema and reject unknown fields (call_id,
        response_item_id) with 400 or 422 errors. These fields are preserved in
        the internal message history — this method only modifies the outgoing
        API copy.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).

        Fields stripped: call_id, response_item_id
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    def _should_sanitize_tool_calls(self) -> bool:
        """判断是否需要为严格 API 清理 tool_calls。

        Determine if tool_calls need sanitization for strict APIs.

        Codex Responses API uses fields like call_id and response_item_id
        that are not part of the standard Chat Completions schema. These
        fields must be stripped when calling any other API to avoid
        validation errors (400 Bad Request).

        Returns:
            bool: True if sanitization is needed (non-Codex API), False otherwise.
        """
        return self.api_mode != "codex_responses"

    def flush_memories(self, messages: list = None, min_turns: int = None):
        """让模型在一轮中持久化记忆，防止上下文丢失。

        Give the model one turn to persist memories before context is lost.

        Called before compression, session reset, or CLI exit. Injects a flush
        message, makes one API call, executes any memory tool calls, then
        strips all flush artifacts from the message list.

        Args:
            messages: The current conversation messages. If None, uses
                      self._session_messages (last run_conversation state).
            min_turns: Minimum user turns required to trigger the flush.
                       None = use config value (flush_min_turns).
                       0 = always flush (used for compression).
        """
        if self._memory_flush_min_turns == 0 and min_turns is None:
            return
        if "memory" not in self.valid_tool_names or not self._memory_store:
            return
        effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, '_session_messages', None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Save anything worth remembering — prioritize user preferences, "
            "corrections, and recurring patterns over task-specific details.]"
        )
        _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
        flush_msg = {"role": "user", "content": flush_content, "_flush_sentinel": _sentinel}
        messages.append(flush_msg)

        try:
            # 为刷新调用构建 API 消息
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                self._copy_reasoning_content_for_api(msg, api_msg)
                api_msg.pop("reasoning", None)
                api_msg.pop("finish_reason", None)
                api_msg.pop("_flush_sentinel", None)
                api_msg.pop("_thinking_prefill", None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            if self._cached_system_prompt:
                api_messages = [{"role": "system", "content": self._cached_system_prompt}] + api_messages

            # 仅使用记忆工具进行一次 API 调用
            memory_tool_def = None
            for t in (self.tools or []):
                if t.get("function", {}).get("name") == "memory":
                    memory_tool_def = t
                    break

            if not memory_tool_def:
                messages.pop()  # remove flush msg
                return

            # 可用时使用辅助客户端进行刷新调用——
            # 更便宜且避免 Codex Responses API 不兼容。
            from agent.auxiliary_client import (
                call_llm as _call_llm,
                _fixed_temperature_for_model,
                OMIT_TEMPERATURE,
            )
            _aux_available = True
            # Kimi models manage temperature server-side — omit it entirely.
            # 具有固定合约的其他模型使用该值；其余
            # 使用历史默认值 0.3。
            _fixed_temp = _fixed_temperature_for_model(self.model, self.base_url)
            _omit_temperature = _fixed_temp is OMIT_TEMPERATURE
            if _omit_temperature:
                _flush_temperature = None
            elif _fixed_temp is not None:
                _flush_temperature = _fixed_temp
            else:
                _flush_temperature = 0.3
            try:
                response = _call_llm(
                    task="flush_memories",
                    messages=api_messages,
                    tools=[memory_tool_def],
                    temperature=_flush_temperature,
                    max_tokens=5120,
                    # 超时从 auxiliary.flush_memories.timeout 配置解析
                )
            except RuntimeError:
                _aux_available = False
                response = None

            if not _aux_available and self.api_mode == "codex_responses":
                # 无辅助客户端——直接使用 Codex Responses 路径
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs["tools"] = self._get_transport().convert_tools([memory_tool_def])
                if _flush_temperature is not None:
                    codex_kwargs["temperature"] = _flush_temperature
                else:
                    codex_kwargs.pop("temperature", None)
                if "max_output_tokens" in codex_kwargs:
                    codex_kwargs["max_output_tokens"] = 5120
                response = self._run_codex_stream(codex_kwargs)
            elif not _aux_available and self.api_mode == "anthropic_messages":
                # Native Anthropic — use the transport for kwargs
                _tflush = self._get_transport()
                ant_kwargs = _tflush.build_kwargs(
                    model=self.model, messages=api_messages,
                    tools=[memory_tool_def], max_tokens=5120,
                    reasoning_config=None,
                    preserve_dots=self._anthropic_preserve_dots(),
                )
                response = self._anthropic_messages_create(ant_kwargs)
            elif not _aux_available:
                api_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": [memory_tool_def],
                    **self._max_tokens_param(5120),
                }
                if _flush_temperature is not None:
                    api_kwargs["temperature"] = _flush_temperature
                from agent.auxiliary_client import _get_task_timeout
                response = self._ensure_primary_openai_client(reason="flush_memories").chat.completions.create(
                    **api_kwargs, timeout=_get_task_timeout("flush_memories")
                )

            # 从响应提取工具调用，处理所有 API 格式
            tool_calls = []
            if self.api_mode == "codex_responses" and not _aux_available:
                _ct_flush = self._get_transport()
                _cnr_flush = _ct_flush.normalize_response(response)
                if _cnr_flush and _cnr_flush.tool_calls:
                    tool_calls = [
                        SimpleNamespace(
                            id=tc.id, type="function",
                            function=SimpleNamespace(name=tc.name, arguments=tc.arguments),
                        ) for tc in _cnr_flush.tool_calls
                    ]
            elif self.api_mode == "anthropic_messages" and not _aux_available:
                _tfn = self._get_transport()
                _flush_nr = _tfn.normalize_response(response, strip_tool_prefix=self._is_anthropic_oauth)
                if _flush_nr and _flush_nr.tool_calls:
                    tool_calls = [
                        SimpleNamespace(
                            id=tc.id, type="function",
                            function=SimpleNamespace(name=tc.name, arguments=tc.arguments),
                        ) for tc in _flush_nr.tool_calls
                    ]
            elif hasattr(response, "choices") and response.choices:
                # chat_completions / bedrock — normalize through transport
                _flush_cc_nr = self._get_transport().normalize_response(response)
                _flush_msg = self._nr_to_assistant_message(_flush_cc_nr)
                if _flush_msg.tool_calls:
                    tool_calls = _flush_msg.tool_calls

            for tc in tool_calls:
                if tc.function.name == "memory":
                    try:
                        args = json.loads(tc.function.arguments)
                        flush_target = args.get("target", "memory")
                        from tools.memory_tool import memory_tool as _memory_tool
                        _memory_tool(
                            action=args.get("action"),
                            target=flush_target,
                            content=args.get("content"),
                            old_text=args.get("old_text"),
                            store=self._memory_store,
                        )
                        if not self.quiet_mode:
                            print(f"  🧠 Memory flush: saved to {args.get('target', 'memory')}")
                    except Exception as e:
                        logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # 剥离刷新痕迹：从刷新消息起移除所有内容。
            # 使用标记哨兵而非身份检查以增强鲁棒性。
            while messages and messages[-1].get("_flush_sentinel") != _sentinel:
                messages.pop()
                if not messages:
                    break
            if messages and messages[-1].get("_flush_sentinel") == _sentinel:
                messages.pop()

    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default", focus_topic: str = None) -> tuple:
        """[上下文压缩入口] 压缩对话上下文 + 在 SQLite 中拆分会话。

        这是 ContextCompressor.compress() 的上层包装，额外负责：

        1. 压缩前：
           - flush_memories(): 强制刷新当前内存，让记忆管理器在上下文丢失前保存
           - memory_manager.on_pre_compress(): 通知外部记忆提供商（如 Honcho）
             即将有上下文被丢弃，让它们有机会提取最终见解

        2. 压缩中：
           - context_compressor.compress(): 执行实际的消息列表压缩

        3. 压缩后：
           - todo_snapshot: 将当前 TODO 列表注入到压缩后的上下文中
           - _invalidate_system_prompt(): 标记 system prompt 需重建
           - _build_system_prompt(): 用新的压缩后的消息上下文重建 system prompt

        4. 会话拆分（SQLite）:
           - 结束旧 session (end_reason = "compression")
           - 创建新 session (parent_session_id = 旧 session id)
           - 标题自动编号: "my session" → "my session #2" → "my session #3"
           - 在新的 session 中存储重建后的 system_prompt
           - 重置 _last_flushed_db_idx = 0，后续消息写入新 session

        5. 提示用户：
           - 压缩 ≥2 次时警告 "accuracy may degrade. Consider /new"

        会话链:
          压缩创建的 session 链通过 parent_session_id 连接，
          SessionDB.get_compression_tip() 可遍历链找到最新延续，
          确保用户在 /sessions 列表中看到的是一条逻辑会话。

        Args:
            messages: 当前完整的消息列表
            system_message: 原始 system prompt 文本
            approx_tokens: 当前估算的 token 数
            task_id: 任务标识
            focus_topic: 可选话题（/compress <topic> 引导压缩）

        Returns:
            (compressed_messages, new_system_prompt): 压缩后的消息列表和新的 system prompt
        """
        _pre_msg_count = len(messages)
        logger.info(
            "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
            self.session_id or "none", _pre_msg_count,
            f"{approx_tokens:,}" if approx_tokens else "unknown", self.model,
            focus_topic,
        )
        # 压缩前记忆刷新：让模型在记忆丢失前保存它们
        self.flush_memories(messages, min_turns=0)

        # 在压缩丢弃上下文前通知外部记忆提供商
        if self._memory_manager:
            try:
                self._memory_manager.on_pre_compress(messages)
            except Exception:
                pass

        compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic)

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # 将标题传播到新会话并自动编号
                old_title = self._session_db.get_session_title(self.session_id)
                # 在旧会话轮换前触发记忆提取。
                self.commit_memory_session(messages)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                # 更新 session_log_file 指向新会话的 JSON 文件
                self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    parent_session_id=old_session_id,
                )
                # 为延续会话自动编号标题
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(old_title)
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(self.session_id, new_system_prompt)
                # Reset flush cursor — new session starts with no messages written
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # 重复压缩时警告（每轮质量都会下降）
        _cc = self.context_compressor.compression_count
        if _cc >= 2:
            self._vprint(
                f"{self.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh.",
                force=True,
            )

        # 压缩后更新 token 估算，使压力计算
        # 使用压缩后计数而非过期的压缩前计数。
        _compressed_est = (
            estimate_tokens_rough(new_system_prompt)
            + estimate_messages_tokens_rough(compressed)
        )
        self.context_compressor.last_prompt_tokens = _compressed_est
        self.context_compressor.last_completion_tokens = 0

        # 清除文件读取去重缓存。压缩后原始
        # read content is summarised away — if the model re-reads the same
        # 文件，需要完整内容而非"文件未变"的桩。
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d tokens=~%s",
            self.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """执行 assistant 消息中的工具调用并将结果追加到 messages。

        [中文] 工具调用分发入口:
          - 只读工具始终可并发
          - 文件读写工具需检查路径不重叠才并发
          - 结果追加到 messages 列表作为 tool role 消息
          - 通过 _should_parallelize_tool_batch() 判断并发/顺序

        仅对看起来独立的批次调度并发执行: 只读工具始终可共享并行路径，
        文件读写工具仅在目标路径不重叠时可并发。
        """
        tool_calls = assistant_message.tool_calls

        # 允许工具执行期间 _vprint，即使有 stream consumers
        self._executing_tools = True
        try:
            if not _should_parallelize_tool_batch(tool_calls):
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            return self._execute_tool_calls_concurrent(
                assistant_message, messages, effective_task_id, api_call_count
            )
        finally:
            self._executing_tools = False

    def _dispatch_delegate_task(self, function_args: dict) -> str:
        """delegate_task 分发的单一调用点。

        Single call site for delegate_task dispatch.

        New DELEGATE_TASK_SCHEMA fields only need to be added here to reach all
        invocation paths (concurrent, sequential, inline).
        """
        from tools.delegate_tool import delegate_task as _delegate_task
        return _delegate_task(
            goal=function_args.get("goal"),
            context=function_args.get("context"),
            toolsets=function_args.get("toolsets"),
            tasks=function_args.get("tasks"),
            max_iterations=function_args.get("max_iterations"),
            acp_command=function_args.get("acp_command"),
            acp_args=function_args.get("acp_args"),
            role=function_args.get("role"),
            parent_agent=self,
        )

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None, messages: list = None) -> str:
        """调用单个工具并返回结果字符串。无显示逻辑。

        [中文] 单工具调用路由 — 根据工具名分发到不同处理路径:
          1. 插件钩子 pre_tool_call 可阻止执行 (返回 block_message)
          2. Agent 级工具 (todo/memory/session_search/delegate_task/clarify)
             → 直接在 AIAgent 内处理，因为需要 agent 级状态 (TodoStore/MemoryStore)
          3. 记忆提供商工具 → MemoryManager 路由
          4. 其他工具 → model_tools.handle_function_call() (通过 registry.dispatch)

        处理 agent 级工具 (todo, memory 等) 和 registry 分发的工具。
        用于并发执行路径；顺序路径保留自己的内联调用
        以保持向后兼容的显示处理。
        """
        # 执行前检查插件钩子是否有阻止指令。
        block_message: Optional[str] = None
        try:
            from hermes_cli.plugins import get_pre_tool_call_block_message
            block_message = get_pre_tool_call_block_message(
                function_name, function_args, task_id=effective_task_id or "",
            )
        except Exception:
            pass
        if block_message is not None:
            return json.dumps({"error": block_message}, ensure_ascii=False)

        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            if not self._session_db:
                return json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=self._session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # 桥接: 通知外部记忆提供商内置记忆写入
            if self._memory_manager and function_args.get("action") in ("add", "replace"):
                try:
                    self._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                    )
                except Exception:
                    pass
            return result
        elif self._memory_manager and self._memory_manager.has_tool(function_name):
            return self._memory_manager.handle_tool_call(function_name, function_args)
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            return self._dispatch_delegate_task(function_args)
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                tool_call_id=tool_call_id,
                session_id=self.session_id or "",
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
            )

    @staticmethod
    def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
        """自动换行工具详细输出以适应终端宽度。

        Word-wrap verbose tool output to fit the terminal width.

        Splits *text* on existing newlines and wraps each line individually,
        preserving intentional line breaks (e.g. pretty-printed JSON).
        Returns a ready-to-print string with *label* on the first line and
        continuation lines indented.
        """
        import shutil as _shutil
        import textwrap as _tw
        cols = _shutil.get_terminal_size((120, 24)).columns
        wrap_width = max(40, cols - len(indent))
        out_lines: list[str] = []
        for raw_line in text.split("\n"):
            if len(raw_line) <= wrap_width:
                out_lines.append(raw_line)
            else:
                wrapped = _tw.wrap(raw_line, width=wrap_width,
                                   break_long_words=True,
                                   break_on_hyphens=False)
                out_lines.extend(wrapped or [raw_line])
        body = ("\n" + indent).join(out_lines)
        return f"{indent}{label}{body}"

    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """使用线程池并发执行多个工具调用。

        Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages so the API sees them in the expected sequence.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── Pre-flight: interrupt check ──────────────────────────────────
        if self._interrupt_requested:
            print(f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                    "tool_call_id": tc.id,
                })
            return

        # ── Parse args + pre-execution bookkeeping ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # 重置 nudge 计数器
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # 为文件修改工具设置检查点
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # 破坏性终端命令前的检查点
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

            parsed_calls.append((tool_call, function_name, function_args))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(self._wrap_verbose("Args: ", json.dumps(args, indent=2, ensure_ascii=False)))
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for tc, name, args in parsed_calls:
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args in parsed_calls:
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # 每个槽存储（function_name、function_args、function_result、duration、error_flag）
        results = [None] * num_tools

        # 启动 worker 前触碰活动追踪器，使网关知道
        # 我们正在执行工具（非卡住）。
        self._current_tool = tool_names_str
        self._touch_activity(f"executing {num_tools} tools concurrently: {tool_names_str}")

        def _run_tool(index, tool_call, function_name, function_args):
            """在线程中执行的工作函数。

            Worker function executed in a thread.
            """
            # 注册此 worker tid，使 Agent 能将中断
            # to it — see AIAgent.interrupt().  Must happen first thing, and
            # 必须与 finally 块中的 discard + clear 配对。
            _worker_tid = threading.current_thread().ident
            with self._tool_worker_threads_lock:
                self._tool_worker_threads.add(_worker_tid)
            # 竞态：若 Agent 在分发（快照了空/更早的集合）
            # 和我们的注册之间被中断，现在将中断
            # 应用到我们自己的 tid，使工具内的
            # is_interrupted() 在下一次轮询时返回 True。
            if self._interrupt_requested:
                try:
                    _set_interrupt(True, _worker_tid)
                except Exception:
                    pass
            # 在此 worker 线程上设置活动回调，使
            # _wait_for_process（终端命令）能触发心跳。
            # 回调是线程本地的；主线程的回调
            # 对 worker 线程不可见。
            try:
                from tools.environments.base import set_activity_callback
                set_activity_callback(self._touch_activity)
            except Exception:
                pass
            start = time.time()
            try:
                result = self._invoke_tool(function_name, function_args, effective_task_id, tool_call.id, messages=messages)
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error)
            # 拆除 worker-tid 追踪。清除我们可能设置的
            # 中断位，使调度到此回收 tid 的下一个任务
            # 以干净状态启动。
            with self._tool_worker_threads_lock:
                self._tool_worker_threads.discard(_worker_tid)
            try:
                _set_interrupt(False, _worker_tid)
            except Exception:
                pass

        # 启动 CLI 模式 spinner（TUI 处理工具进度时跳过）
        spinner = None
        if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.get_waiting_faces())
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=self._print_fn)
            spinner.start()

        try:
            max_workers = min(num_tools, _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, (tc, name, args) in enumerate(parsed_calls):
                    f = executor.submit(_run_tool, i, tc, name, args)
                    futures.append(f)

                # 等待所有完成并定期发送心跳，使
                # 网关的不活动监控器在长时间并发工具批次
                # 期间不会杀死我们。同时检查用户中断，
                # 避免在用户发送 /stop 或新消息时无限期阻塞。
                #
                _conc_start = time.time()
                _interrupt_logged = False
                while True:
                    done, not_done = concurrent.futures.wait(
                        futures, timeout=5.0,
                    )
                    if not not_done:
                        break

                    # Check for interrupt — the per-thread interrupt signal
                    # 已导致个别工具（terminal、execute_code）
                    # 中止，但无中断检查的工具（web_search、
                    # read_file）会运行至完成。取消尚未启动的
                    # future，避免阻塞它们。
                    if self._interrupt_requested:
                        if not _interrupt_logged:
                            _interrupt_logged = True
                            self._vprint(
                                f"{self.log_prefix}⚡ Interrupt: cancelling "
                                f"{len(not_done)} pending concurrent tool(s)",
                                force=True,
                            )
                        for f in not_done:
                            f.cancel()
                        # 给已运行的工具一点时间来感知
                        # 每线程中断信号并优雅退出。
                        concurrent.futures.wait(not_done, timeout=3.0)
                        break

                    _conc_elapsed = int(time.time() - _conc_start)
                    # Heartbeat every ~30s (6 × 5s poll intervals)
                    if _conc_elapsed > 0 and _conc_elapsed % 30 < 6:
                        _still_running = [
                            parsed_calls[futures.index(f)][1]
                            for f in not_done
                            if f in futures
                        ]
                        self._touch_activity(
                            f"concurrent tools running ({_conc_elapsed}s, "
                            f"{len(not_done)} remaining: {', '.join(_still_running[:3])})"
                        )
        finally:
            if spinner:
                # 为 spinner 停止构建摘要消息
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args) in enumerate(parsed_calls):
            r = results[i]
            if r is None:
                # 工具被取消（中断）或线程未返回
                if self._interrupt_requested:
                    function_result = f"[Tool execution cancelled — {name} was skipped due to user interrupt]"
                else:
                    function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error = r

                if is_error:
                    result_preview = function_result[:200] if len(function_result) > 200 else function_result
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                if self.tool_progress_callback:
                    try:
                        self.tool_progress_callback(
                            "tool.completed", function_name, None, None,
                            duration=tool_duration, is_error=is_error,
                        )
                    except Exception as cb_err:
                        logging.debug(f"Tool progress callback error: {cb_err}")

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # 每个工具打印可爱消息
            if self._should_emit_quiet_tool_messages():
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(self._wrap_verbose("Result: ", function_result))
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=name,
                tool_use_id=tc.id,
                env=get_active_env(effective_task_id),
            )

            subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

            # ── Per-tool /steer drain ───────────────────────────────────
            # 顺序路径相同：在每个收集的
            # 结果间排空，使 steer 尽早落地。
            self._apply_pending_steer_to_tool_results(messages, 1)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools = len(parsed_calls)
        if num_tools > 0:
            turn_tool_msgs = messages[-num_tools:]
            enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id))

        # ── /steer injection ──────────────────────────────────────────────
        # 将待处理的用户 steer 文本附加到最后一条工具结果，使
        # Agent 在下一次迭代中看到它。在预算强制执行后运行，
        # 确保 steer 标记永不被截断。详见 steer()。
        if num_tools > 0:
            self._apply_pending_steer_to_tool_results(messages, num_tools)

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """顺序执行工具调用（原始行为）。用于单个调用或交互式工具。

        Execute tool calls sequentially (original behavior). Used for single calls or interactive tools.
        """
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # 安全检查：启动每个工具前检查中断。
            # 若用户在前一个工具执行期间发送了"stop"，
            # 不再启动任何工具——立即跳过所有。
            if self._interrupt_requested:
                remaining_calls = assistant_message.tool_calls[i-1:]
                if remaining_calls:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # 执行前检查插件钩子的阻止指令。
            _block_msg: Optional[str] = None
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                _block_msg = get_pre_tool_call_block_message(
                    function_name, function_args, task_id=effective_task_id or "",
                )
            except Exception:
                pass

            if _block_msg is not None:
                # Tool blocked by plugin policy — skip counter resets.
                # 执行在下方工具分发链中处理。
                pass
            else:
                # 当相关工具实际被使用时重置 nudge 计数器
                if function_name == "memory":
                    self._turns_since_memory = 0
                elif function_name == "skill_manage":
                    self._iters_since_skill = 0

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(self._wrap_verbose("Args: ", json.dumps(function_args, indent=2, ensure_ascii=False)))
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            if _block_msg is None:
                self._current_tool = function_name
                self._touch_activity(f"executing tool: {function_name}")

            # 为长时间运行的工具执行（终端命令等）
            # 设置活动回调，使网关的不活动监控器
            # 不会在命令运行期间杀死 Agent。
            if _block_msg is None:
                try:
                    from tools.environments.base import set_activity_callback
                    set_activity_callback(self._touch_activity)
                except Exception:
                    pass

            if _block_msg is None and self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback("tool.started", function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if _block_msg is None and self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # 检查点：文件修改工具前快照工作目录
            if _block_msg is None and function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(
                            work_dir, f"before {function_name}"
                        )
                except Exception:
                    pass  # never block tool execution

            # 破坏性终端命令前的检查点
            if _block_msg is None and function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass  # never block tool execution

            tool_start_time = time.time()

            if _block_msg is not None:
                # Tool blocked by plugin policy — return error without executing.
                function_result = json.dumps({"error": _block_msg}, ensure_ascii=False)
                tool_duration = 0.0
            elif function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                if not self._session_db:
                    function_result = json.dumps({"success": False, "error": "Session database not available."})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=self._session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            elif function_name == "memory":
                target = function_args.get("target", "memory")
                from tools.memory_tool import memory_tool as _memory_tool
                function_result = _memory_tool(
                    action=function_args.get("action"),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                # 桥接：通知外部记忆提供商内置记忆的写入
                if self._memory_manager and function_args.get("action") in ("add", "replace"):
                    try:
                        self._memory_manager.on_memory_write(
                            function_args.get("action", ""),
                            target,
                            function_args.get("content", ""),
                        )
                    except Exception:
                        pass
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = self._dispatch_delegate_task(function_args)
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._context_engine_tool_names and function_name in self._context_engine_tool_names:
                # 上下文引擎工具（lcm_grep、lcm_describe、lcm_expand 等）
                spinner = None
                if self._should_emit_quiet_tool_messages():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _ce_result = None
                try:
                    function_result = self.context_compressor.handle_tool_call(function_name, function_args, messages=messages)
                    _ce_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                    logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._memory_manager and self._memory_manager.has_tool(function_name):
                # 记忆提供商工具（hindsight_retain、honcho_search 等）
                # These are not in the tool registry — route through MemoryManager.
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _mem_result = None
                try:
                    function_result = self._memory_manager.handle_tool_call(function_name, function_args)
                    _mem_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                    logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode:
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.get_waiting_faces())
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        skip_pre_tool_call_hook=True,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                        skip_pre_tool_call_hook=True,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            result_preview = function_result if self.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )

            # 将工具错误记录到持久错误日志，使 UI 中的
            # [error] 标记在磁盘上总有相应的详细条目。
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, len(function_result))

            if self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=_is_error_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=function_name,
                tool_use_id=tool_call.id,
                env=get_active_env(effective_task_id),
            )

            # 从工具参数发现子目录上下文文件
            subdir_hints = self._subdirectory_hints.check_tool_call(function_name, function_args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            # ── Per-tool /steer drain ───────────────────────────────────
            # 在单个工具调用之间排空待处理的 steer，使
            # injection lands as soon as a tool finishes — not after the
            # 模型在下一次 API 迭代时看到它。
            self._apply_pending_steer_to_tool_results(messages, 1)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(self._wrap_verbose("Result: ", function_result))
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools_seq = len(assistant_message.tool_calls)
        if num_tools_seq > 0:
            enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id))

        # ── /steer injection ──────────────────────────────────────────────
        # 原理见 _execute_tool_calls_parallel。相同钩子，
        # 也应用于顺序执行。
        if num_tools_seq > 0:
            self._apply_pending_steer_to_tool_results(messages, num_tools_seq)



    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """达到最大迭代次数时请求摘要。返回最终响应文本。"""
        print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # 构建 API 消息，剥离仅内部使用的字段
            # (finish_reason, reasoning)，严格 API 如 Mistral 会以 422 拒绝
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                    api_msg.pop(internal_field, None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            summary_extra_body = {}
            try:
                from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE as _OMIT_TEMP
            except Exception:
                _fixed_temperature_for_model = None
                _OMIT_TEMP = None
            _raw_summary_temp = (
                _fixed_temperature_for_model(self.model, self.base_url)
                if _fixed_temperature_for_model is not None
                else None
            )
            _omit_summary_temperature = _raw_summary_temp is _OMIT_TEMP
            _summary_temperature = None if _omit_summary_temperature else _raw_summary_temp
            _is_nous = "nousresearch" in self._base_url_lower
            if self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                _ct_sum = self._get_transport()
                _cnr_sum = _ct_sum.normalize_response(summary_response)
                final_response = (_cnr_sum.content or "").strip()
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if _summary_temperature is not None:
                    summary_kwargs["temperature"] = _summary_temperature
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))

                # 包含提供商路由偏好设置
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences:
                    summary_extra_body["provider"] = provider_preferences

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    _tsum = self._get_transport()
                    _ant_kw = _tsum.build_kwargs(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=self._is_anthropic_oauth,
                                   preserve_dots=self._anthropic_preserve_dots())
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _sum_nr = _tsum.normalize_response(summary_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_sum_nr.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)
                    _sum_cc_nr = self._get_transport().normalize_response(summary_response)
                    final_response = (_sum_cc_nr.content or "").strip()

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                # 重试摘要生成
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    _ct_retry = self._get_transport()
                    _cnr_retry = _ct_retry.normalize_response(retry_response)
                    final_response = (_cnr_retry.content or "").strip()
                elif self.api_mode == "anthropic_messages":
                    _tretry = self._get_transport()
                    _ant_kw2 = _tretry.build_kwargs(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=self._is_anthropic_oauth,
                                    max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                    preserve_dots=self._anthropic_preserve_dots())
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_nr = _tretry.normalize_response(retry_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_retry_nr.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if _summary_temperature is not None:
                        summary_kwargs["temperature"] = _summary_temperature
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)
                    _retry_cc_nr = self._get_transport().normalize_response(summary_response)
                    final_response = (_retry_cc_nr.content or "").strip()

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    if final_response:
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling until completion.

        [中文] 运行一次完整的对话，包含工具调用循环，直到模型产生最终回复或迭代预算耗尽。
        这是 AIAgent 的核心公共方法，整个 Agent 的生命线。

        Args:
            user_message (str): The user's message/question
            system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
            conversation_history (List[Dict]): Previous conversation messages (optional)
            task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)
            stream_callback: Optional callback invoked with each text delta during streaming.
                Used by the TTS pipeline to start audio generation before the full response.
                When None (default), API calls use the standard non-streaming path.
            persist_user_message: Optional clean user message to store in
                transcripts/history when user_message contains API-only
                synthetic prefixes.
                    or queuing follow-up prefetch work.

        Returns:
            Dict: Complete conversation result with final response and message history.
                包含字段: final_response, messages, api_calls, completed, partial,
                interrupted, error, failed
        """
        # ═══════════════════════════════════════════════════════════════
        # [阶段一: 初始化] 初始化会话环境、清洗输入、重置重试计数器
        # ═══════════════════════════════════════════════════════════════

        # 防止断管道导致的 OSError (systemd/headless/daemon 场景)
        # 只安装一次，流正常时透明，防止写入时崩溃
        _install_safe_stdio()

        # 为当前线程的所有日志记录标记 session ID，
        # 以便 ``hermes logs --session <id>`` 过滤单次对话。
        from hermes_logging import set_session_context
        set_session_context(self.session_id)

        # 如果上一轮触发了 fallback，恢复主运行时，让本轮使用首选模型重新尝试。
        # 当 _fallback_activated 为 False 时无操作 (gateway、首轮等场景)。
        # [中文] 如果上一轮触发了 fallback，这里恢复到主模型，给本轮一个全新尝试
        self._restore_primary_runtime()

        # 清洗用户输入中的 surrogate 字符。富文本编辑器 (Google Docs、Word 等)
        # 的剪贴板粘贴可能注入无效的 lone surrogate，导致 OpenAI SDK 的 JSON 序列化崩溃。
        if isinstance(user_message, str):
            user_message = _sanitize_surrogates(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = _sanitize_surrogates(persist_user_message)

        # 剥离用户输入中泄漏的 <memory-context> 块。当 Honcho 的 saveMessages
        # 持久化了包含注入上下文的轮次时，该块可能通过消息历史重新出现在下一轮的用户消息中。
        # 在此剥离可防止过时的 memory 标签泄漏到对话中，避免被用户或模型视为用户文本。
        if isinstance(user_message, str):
            user_message = sanitize_context(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = sanitize_context(persist_user_message)

        # 保存 stream callback 供 _interruptible_api_call 使用
        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message
        # 未提供 task_id 时生成唯一标识，用于并发任务间的 VM 隔离
        effective_task_id = task_id or str(uuid.uuid4())
        # 暴露当前 task_id，使轮次内运行的工具 (如 delegate_tool.py 中的 delegate_task)
        # 能识别此 Agent 以用于跨 Agent 文件状态注册表。
        # 在任何工具调度之前设置，确保子 Agent 启动时快照能看到父 Agent 的真实 id 而非 None。
        self._current_task_id = effective_task_id
        
        # 每轮开始时重置重试计数器和迭代预算，
        # 防止上一轮子 Agent 的使用占用下一轮的预算。
        # [中文] 重置所有重试计数器，确保上一轮子 Agent 的使用不影响本轮预算
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._thinking_prefill_retries = 0
        self._post_tool_empty_retried = False
        self._last_content_with_tools = None
        self._last_content_tools_all_housekeeping = False
        self._mute_post_response = False
        self._unicode_sanitization_passes = 0

        # 轮次前连接健康检查：检测并清理提供商故障或流断开后残留的死 TCP 连接。
        # 防止下一次 API 调用在僵尸 socket 上挂起。
        if self.api_mode != "anthropic_messages":
            try:
                if self._cleanup_dead_connections():
                    self._emit_status(
                        "🔌 Detected stale connections from a previous provider "
                        "issue — cleaned up automatically. Proceeding with fresh "
                        "connection."
                    )
            except Exception:
                pass
        # 通过 status_callback 重放压缩警告 (gateway 平台场景，
        # 该回调在 __init__ 时尚未接入)。
        if self._compression_warning:
            self._replay_compression_warning()
            self._compression_warning = None  # 只发送一次

        # 注意: _turns_since_memory 和 _iters_since_skill 不在此处重置。
        # 它们在 __init__ 中初始化，必须在多次 run_conversation 调用间保持，以确保 CLI 模式下 nudge 逻辑正确累积。
        self.iteration_budget = IterationBudget(self.max_iterations)

        # 记录对话轮次开始日志，用于调试/可观测性
        _preview_text = _summarize_user_message_for_log(user_message)
        _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
        _msg_preview = _msg_preview.replace("\n", " ")
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            self.session_id or "none", self.model, self.provider or "unknown",
            self.platform or "unknown", len(conversation_history or []),
            _msg_preview,
        )

        # 初始化对话 (拷贝以避免修改调用方的列表)
        messages = list(conversation_history) if conversation_history else []

        # 从对话历史恢复 todo store (gateway 每条消息创建新的 AIAgent，
        # 内存中 store 为空 -- 需要从历史中最近的 todo 工具响应恢复状态)
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)
        
        # Prefill 消息 (few-shot priming) 仅在 API 调用时注入，不存入 messages 列表。
        # 这使其保持临时性：不会保存到 session DB、session 日志或 batch trajectories，
        # 但会在每次 API 调用时自动重新应用 (包括 session 续接)。
        
        # 跟踪用户轮次，用于 memory 刷新和周期性 nudge 逻辑
        self._user_turn_count += 1

        # 保留原始用户消息 (不含 nudge 注入)。
        original_user_message = persist_user_message if persist_user_message is not None else user_message

        # 跟踪 memory nudge 触发条件 (基于轮次，在此检查)。
        # Skill 触发在 Agent 循环完成后检查，基于本轮使用的工具迭代次数。
        _should_review_memory = False
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                _should_review_memory = True
                self._turns_since_memory = 0

        # 添加用户消息
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx
        
        if not self.quiet_mode:
            _print_preview = _summarize_user_message_for_log(user_message)
            self._safe_print(f"💬 Starting conversation: '{_print_preview[:60]}{'...' if len(_print_preview) > 60 else ''}'")
        
        # ═══════════════════════════════════════════════════════════════
        # [阶段二: 系统提示构建] 只构建一次，后续复用；压缩后才重建
        # ═══════════════════════════════════════════════════════════════
        # ── system prompt (按会话缓存，命中 prefix cache) ──
        # 首次调用时构建一次，后续复用；
        # 仅在 context compression 事件后重建（压缩会使缓存失效并重新加载 memory）
        # [中文] 系统提示按会话缓存，复用以最大化 Anthropic 前缀缓存命中率
        #
        # 对于持续会话（gateway 每条消息创建新的 AIAgent），
        # 从 session DB 加载已存储的 system prompt，而非重新构建。
        # 重新构建会引入模型已知的 memory 变更（模型自己写的！），
        # 导致 system prompt 不一致，破坏 Anthropic prefix cache。
        if self._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and self._session_db:
                try:
                    session_row = self._session_db.get_session(self.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass  # 回退到重新构建

            if stored_prompt:
                # 持续会话 — 复用上一轮的精确 system prompt，确保 Anthropic cache prefix 匹配
                self._cached_system_prompt = stored_prompt
            else:
                # 新会话首轮 — 从头构建
                self._cached_system_prompt = self._build_system_prompt(system_message)
                # 插件钩子：on_session_start
                # 仅在全新会话创建时触发一次（持续会话不触发）。
                # Plugin 可借此初始化会话级状态（如预热 memory cache）。
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_start",
                        session_id=self.session_id,
                        model=self.model,
                        platform=getattr(self, "platform", None) or "",
                    )
                except Exception as exc:
                    logger.warning("on_session_start hook failed: %s", exc)

                # 将 system prompt 快照存入 SQLite
                if self._session_db:
                    try:
                        self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                    except Exception as e:
                        logger.debug("Session DB update_system_prompt failed: %s", e)

        active_system_prompt = self._cached_system_prompt

        # ═══════════════════════════════════════════════════════════════
        # [阶段三: 预压缩] 进入主循环前，检查历史是否超出上下文阈值
        # ═══════════════════════════════════════════════════════════════
        # ── 预压缩 context compression ──
        # 进入主循环前，检查已加载的对话历史是否超出模型的 context 阈值。
        # 处理场景：用户在已有大 session 中切换到小 context window 的模型 ——
        # 主动压缩，而非等待 API 错误（可能被当作不可重试的 4xx 直接中止请求）。
        if (
            self.compression_enabled
            and len(messages) > self.context_compressor.protect_first_n
                                + self.context_compressor.protect_last_n + 1
        ):
            # 计入 tool schema token —— 工具多时可额外增加 20-30K+ token，
            # 旧的 sys+msg 估算完全遗漏了这部分。
            _preflight_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=self.tools or None,
            )

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                if not self.quiet_mode:
                    self._safe_print(
                        f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                        f">= {self.context_compressor.threshold_tokens:,} threshold"
                    )
                # 超大 session 配小 context window 可能需要多轮压缩
                # （每轮总结中间 N 轮对话）。
                for _pass in range(3):
                    _orig_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=_preflight_tokens,
                        task_id=effective_task_id,
                    )
                    if len(messages) >= _orig_len:
                        break  # 无法进一步压缩
                    # Compression 创建了新 session —— 清空 history 引用，
                    # 使 _flush_messages_to_session_db 将所有压缩后的消息写入
                    # 新 session 的 SQLite，避免因 conversation_history 仍是
                    # 压缩前长度而跳过写入。
                    conversation_history = None
                    # 修复：compression 后重置重试计数器，使模型在压缩后的
                    # context 上获得新的重试预算。否则压缩前的重试会延续，
                    # 模型在 compression 导致的 context 丢失后立即触发 "(empty)"。
                    self._empty_content_retries = 0
                    self._thinking_prefill_retries = 0
                    self._last_content_with_tools = None
                    self._last_content_tools_all_housekeeping = False
                    self._mute_post_response = False
                    # 压缩后重新估算 token 数
                    _preflight_tokens = estimate_request_tokens_rough(
                        messages,
                        system_prompt=active_system_prompt or "",
                        tools=self.tools or None,
                    )
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # 低于阈值

        # ═══════════════════════════════════════════════════════════════
        # [阶段四: 插件钩子 + 记忆预取]
        # ═══════════════════════════════════════════════════════════════
        # 插件钩子：pre_llm_call
        # 每轮 tool-calling 循环前触发一次。Plugin 可返回含 ``context`` 键的
        # dict（或纯字符串），其值会追加到当前轮次的 user message 末尾。
        #
        # Context 始终注入到 user message 中，从不注入 system prompt。
        # 这样可保持 prompt cache prefix 不变 —— system prompt 在各轮次间
        # 保持一致，缓存的 token 才能被复用。system prompt 由 Hermes 控制；
        # plugin 通过 user input 旁注入 context。
        #
        # 所有注入的 context 均为临时性（不持久化到 session DB）。
        _plugin_user_context = ""
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _pre_results = _invoke_hook(
                "pre_llm_call",
                session_id=self.session_id,
                user_message=original_user_message,
                conversation_history=list(messages),
                is_first_turn=(not bool(conversation_history)),
                model=self.model,
                platform=getattr(self, "platform", None) or "",
                sender_id=getattr(self, "_user_id", None) or "",
            )
            _ctx_parts: list[str] = []
            for r in _pre_results:
                if isinstance(r, dict) and r.get("context"):
                    _ctx_parts.append(str(r["context"]))
                elif isinstance(r, str) and r.strip():
                    _ctx_parts.append(r)
            if _ctx_parts:
                _plugin_user_context = "\n\n".join(_ctx_parts)
        except Exception as exc:
            logger.warning("pre_llm_call hook failed: %s", exc)

        # ═══════════════════════════════════════════════════════════════
        # [阶段五: 主循环] 核心 Agent 循环 — API调用 → 响应处理 → 工具执行
        # 条件: api_call_count < max_iterations 且 iteration_budget.remaining > 0
        # 每轮迭代: 检查中断 → 消耗预算 → 构建API消息 → 调用LLM → 处理响应/工具
        # ═══════════════════════════════════════════════════════════════
        # 主对话循环
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_tool_call_retries = 0
        truncated_response_prefix = ""
        compression_attempts = 0
        _turn_exit_reason = "unknown"  # 诊断：循环结束原因
        
        # 记录执行线程, 使 interrupt()/clear_interrupt() 能将
        # 工具级中断信号限定在本 agent 的线程内.
        # 必须在线程级中断同步之前设置.
        self._execution_thread_id = threading.current_thread().ident

        # 始终清除上一轮遗留的线程状态. 若中断在启动完成前到达,
        # 则保留并绑定到当前执行线程, 而非丢弃.
        _set_interrupt(False, self._execution_thread_id)
        if self._interrupt_requested:
            _set_interrupt(True, self._execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            self._interrupt_message = None
            self._interrupt_thread_signal_pending = False

        # 通知 memory provider 新轮次开始, 以便节奏跟踪生效.
        # 必须在 prefetch_all() 之前执行, 这样 provider 才知道当前轮次,
        # 才能通过 contextCadence/dialecticCadence 控制刷新频率.
        if self._memory_manager:
            try:
                _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
                self._memory_manager.on_turn_start(self._user_turn_count, _turn_msg)
            except Exception:
                pass

        # 外部 memory provider: 在工具循环前预取一次.
        # 每次迭代复用缓存结果, 避免重复调用 prefetch_all()
        # (10 次工具调用 = 10 倍延迟 + 成本).
        # 使用 original_user_message (干净输入) — user_message 可能包含
        # 注入的 skill 内容, 会导致 provider 查询膨胀或失败.
        _ext_prefetch_cache = ""
        if self._memory_manager:
            try:
                _query = original_user_message if isinstance(original_user_message, str) else ""
                _ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""
            except Exception:
                pass

        while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) or self._budget_grace_call:
            # 重置每轮 checkpoint 去重, 使每次迭代可拍一次快照
            self._checkpoint_mgr.new_turn()

            # 检查中断请求 (如用户发送了新消息)
            if self._interrupt_requested:
                interrupted = True
                _turn_exit_reason = "interrupted_by_user"
                if not self.quiet_mode:
                    self._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1
            self._api_call_count = api_call_count
            self._touch_activity(f"starting API call #{api_call_count}")

            # 宽限调用: 预算已耗尽, 但给模型最后一次机会.
            # 消耗宽限标志, 使循环在本次迭代后无论结果如何都退出.
            if self._budget_grace_call:
                self._budget_grace_call = False
            elif not self.iteration_budget.consume():
                _turn_exit_reason = "budget_exhausted"
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
                break

            # 触发 step_callback 供 gateway hooks 使用 (agent:step 事件)
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _idx, _m in enumerate(reversed(messages)):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            _fwd_start = len(messages) - _idx
                            _results_by_id = {}
                            for _tm in messages[_fwd_start:]:
                                if _tm.get("role") != "tool":
                                    break
                                _tcid = _tm.get("tool_call_id")
                                if _tcid:
                                    _results_by_id[_tcid] = _tm.get("content", "")
                            prev_tools = [
                                {
                                    "name": tc["function"]["name"],
                                    "result": _results_by_id.get(tc.get("id")),
                                    "arguments": tc["function"].get("arguments"),
                                }
                                for tc in _m["tool_calls"]
                                if isinstance(tc, dict)
                            ]
                            break
                    self.step_callback(api_call_count, prev_tools)
                except Exception as _step_err:
                    logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

            # 跟踪工具调用迭代次数, 用于 skill nudge.
            # 计数器在 skill_manage 实际使用时重置.
            if (self._skill_nudge_interval > 0
                    and "skill_manage" in self.valid_tool_names):
                self._iters_since_skill += 1
            
            # ── API 调用前 /steer 排空 ──────────────────────────────────
            # 若上一次 API 调用期间收到 /steer (模型正在思考时),
            # 则在构建 api_messages 之前立即排空, 使模型在本轮看到 steer 文本.
            # 否则, API 调用期间发送的 steer 只能在下一批工具结果后才生效,
            # 而如果模型返回了最终响应, 那批工具结果可能永远不会到来.
            #
            # 反向扫描 messages 列表中最后一条 tool 角色消息.
            # 找到则追加 steer; 未找到 (首轮, 尚无工具) 则保持 pending,
            # 等待下一批工具结果 — 注入 user message 会破坏角色交替,
            # 且没有 tool 输出可搭载.
            _pre_api_steer = self._drain_pending_steer()
            if _pre_api_steer:
                _injected = False
                for _si in range(len(messages) - 1, -1, -1):
                    _sm = messages[_si]
                    if isinstance(_sm, dict) and _sm.get("role") == "tool":
                        marker = f"\n\nUser guidance: {_pre_api_steer}"
                        existing = _sm.get("content", "")
                        if isinstance(existing, str):
                            _sm["content"] = existing + marker
                        else:
                            # 多模态内容块 — 追加文本块
                            try:
                                blocks = list(existing) if existing else []
                                blocks.append({"type": "text", "text": marker})
                                _sm["content"] = blocks
                            except Exception:
                                pass
                        _injected = True
                        logger.debug(
                            "Pre-API-call steer drain: injected into tool msg at index %d",
                            _si,
                        )
                        break
                if not _injected:
                    # 没有可注入的 tool 消息 — 放回队列,
                    # 等工具执行后的排空阶段再处理.
                    _lock = getattr(self, "_pending_steer_lock", None)
                    if _lock is not None:
                        with _lock:
                            if self._pending_steer:
                                self._pending_steer = self._pending_steer + "\n" + _pre_api_steer
                            else:
                                self._pending_steer = _pre_api_steer
                    else:
                        existing = getattr(self, "_pending_steer", None)
                        self._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

            # 准备 API 调用消息
            # 如有临时 system prompt, 将其前置到消息列表
            # 注: reasoning 通过 <think> 标签嵌入 content 用于轨迹存储.
            # [中文] 构建 API 请求消息 — 从内部 messages 复制并做以下处理:
            #   1. 注入临时记忆/插件上下文到 user message (仅 API 调用时，不持久化)
            #   2. 复制 reasoning_content 给 API (保留多轮推理上下文)
            #   3. 移除内部字段 (reasoning, finish_reason, _thinking_prefill)
            #   4. 清理 Codex 特有字段 (call_id, response_item_id) 供严格 API 使用
            #   5. 后续: 添加 system prompt + prefill + prompt caching + 归一化
            # 但 Moonshot AI 等 provider 要求 assistant 消息的 tool_calls
            # 附带单独的 'reasoning_content' 字段. 此处两种情况均处理.
            api_messages = []
            for idx, msg in enumerate(messages):
                api_msg = msg.copy()

                # 向当前轮 user message 注入临时上下文.
                # 来源: memory manager prefetch + plugin pre_llm_call hooks
                # (target="user_message", 默认). 两者仅在 API 调用时生效,
                # `messages` 中的原始消息不会被修改, 不会泄漏到会话持久化中.
                if idx == current_turn_user_idx and msg.get("role") == "user":
                    _injections = []
                    if _ext_prefetch_cache:
                        _fenced = build_memory_context_block(_ext_prefetch_cache)
                        if _fenced:
                            _injections.append(_fenced)
                    if _plugin_user_context:
                        _injections.append(_plugin_user_context)
                    if _injections:
                        _base = api_msg.get("content", "")
                        if isinstance(_base, str):
                            api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

                # 对所有 assistant 消息, 将 reasoning 传回 API
                # 确保多轮推理上下文被保留
                self._copy_reasoning_content_for_api(msg, api_msg)

                # 移除 'reasoning' 字段 — 仅用于轨迹存储
                # 已在上方复制到 'reasoning_content' 供 API 使用
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # 移除 finish_reason — 严格 API (如 Mistral) 不接受此字段
                if "finish_reason" in api_msg:
                    api_msg.pop("finish_reason")
                # 移除内部 thinking-prefill 标记
                api_msg.pop("_thinking_prefill", None)
                # 移除 Codex Responses API 字段 (call_id, response_item_id),
                # 供 Mistral, Fireworks 等拒绝未知字段的严格 provider 使用.
                # 使用新 dict 以保留内部 messages 列表中的字段,
                # 维持 Codex Responses 兼容性.
                if self._should_sanitize_tool_calls():
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                # 保留 'reasoning_details' — OpenRouter 用它维持多轮推理上下文
                # 签名字段有助于保持推理连续性
                api_messages.append(api_msg)

            # 构建最终 system message: 缓存 prompt + 临时 system prompt.
            # 临时添加仅在 API 调用时生效 (不持久化到会话 DB).
            # 外部 recall 上下文注入到 user message 而非 system prompt,
            # 以保持稳定的缓存前缀不变.
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            # 注意: pre_llm_call hooks 的 plugin 上下文注入到 user message
            # (见上方注入逻辑), 而非 system prompt.
            # 这是有意为之 — 修改 system prompt 会破坏 prompt cache 前缀.
            # system prompt 仅用于 Hermes 内部逻辑.
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages

            # 在 system prompt 之后、对话历史之前注入临时 prefill 消息.
            # 同样仅在 API 调用时生效.
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # 为 Claude 模型启用 Anthropic prompt caching, 支持原生 Anthropic,
            # OpenRouter 和第三方 Anthropic 兼容 gateway.
            # 自动检测: 若设置了 ``_use_prompt_caching``, 则注入
            # cache_control 断点 (system + 最后3条消息),
            # 可在多轮对话中降低约 75% 的 input token 成本.
            # 布局由 ``_anthropic_prompt_cache_policy`` 按 endpoint 选择.
            if self._use_prompt_caching:
                api_messages = apply_anthropic_cache_control(
                    api_messages,
                    cache_ttl=self._cache_ttl,
                    native_anthropic=self._use_native_cache_layout,
                )

            # 安全网: 发送 API 前移除孤立的 tool 结果 / 为缺失结果添加存根.
            # 无条件执行 — 不受 context_compressor 限制 —
            # 确保会话加载或手动消息操作产生的孤立结果始终被捕获.
            api_messages = self._sanitize_api_messages(api_messages)

            # 规范化消息空白和 tool-call JSON, 确保一致的前缀匹配.
            # 跨轮次 bit-perfect 前缀使本地推理服务器
            # (llama.cpp, vLLM, Ollama) 可复用 KV cache,
            # 并提升云 provider 的 cache 命中率.
            # 操作 api_messages (API 副本), 原始 `messages` 不受影响.
            for am in api_messages:
                if isinstance(am.get("content"), str):
                    am["content"] = am["content"].strip()
            for am in api_messages:
                tcs = am.get("tool_calls")
                if not tcs:
                    continue
                new_tcs = []
                for tc in tcs:
                    if isinstance(tc, dict) and "function" in tc:
                        try:
                            args_obj = json.loads(tc["function"]["arguments"])
                            tc = {**tc, "function": {
                                **tc["function"],
                                "arguments": json.dumps(
                                    args_obj, separators=(",", ":"),
                                    sort_keys=True,
                                ),
                            }}
                        except Exception:
                            tc["function"]["arguments"] = _repair_tool_call_arguments(
                                tc["function"]["arguments"],
                                tc["function"].get("name", "?"),
                            )
                    new_tcs.append(tc)
                am["tool_calls"] = new_tcs

            # API 调用前主动清除 surrogate 字符.
            # 通过 Ollama 服务的模型 (Kimi K2.5, GLM-5, Qwen) 可能返回
            # 孤立 surrogate (U+D800-U+DFFF), 导致 OpenAI SDK 内部
            # json.dumps() 崩溃. 此处清理可避免 3 次重试循环.
            _sanitize_messages_surrogates(api_messages)

            # 计算请求近似大小用于日志
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = estimate_messages_tokens_rough(api_messages)
            
            # quiet 模式下的思考动画 spinner (API 调用期间显示)
            thinking_spinner = None
            
            if not self.quiet_mode:
                self._vprint(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
                self._vprint(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
                self._vprint(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            else:
                # quiet 模式下显示动画思考 spinner
                face = random.choice(KawaiiSpinner.get_thinking_faces())
                verb = random.choice(KawaiiSpinner.get_thinking_verbs())
                if self.thinking_callback:
                    # CLI TUI 模式: 使用 prompt_toolkit 组件代替原始 spinner
                    # (streaming 和非 streaming 模式均可用)
                    self.thinking_callback(f"{face} {verb}...")
                elif not self._has_stream_consumers() and self._should_start_quiet_spinner():
                    # 仅在无 streaming 消费者且 spinner 输出有安全接收端时
                    # 才使用原始 KawaiiSpinner。
                    spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                    thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=self._print_fn)
                    thinking_spinner.start()
            
            # verbose 模式下记录请求详情
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            primary_recovery_attempted = False
            max_compression_attempts = 3
            codex_auth_retry_attempted=False
            anthropic_auth_retry_attempted=False
            nous_auth_retry_attempted=False
            thinking_sig_retry_attempted = False
            has_retried_429 = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # 防止所有重试失败时出现 UnboundLocalError
            api_kwargs = None  # 防止 except 处理器中出现 UnboundLocalError

            while retry_count < max_retries:
                # [中文] API 调用内层重试循环 (最多3次)
                # 错误恢复链: 速率限制 → 凭证池轮换 → 401刷新 → 上下文压缩 → fallback
                # ── Nous Portal rate limit 防护 ──────────────────────
                # 如果其他 session 已记录 Nous 触发了 rate limit，
                # 则完全跳过 API 调用。每次尝试 (包括 SDK 层重试)
                # 都会消耗 RPH 并加深 rate limit 限制。
                if self.provider == "nous":
                    try:
                        from agent.nous_rate_guard import (
                            nous_rate_limit_remaining,
                            format_remaining as _fmt_nous_remaining,
                        )
                        _nous_remaining = nous_rate_limit_remaining()
                        if _nous_remaining is not None and _nous_remaining > 0:
                            _nous_msg = (
                                f"Nous Portal rate limit active — "
                                f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                            )
                            self._vprint(
                                f"{self.log_prefix}⏳ {_nous_msg} Trying fallback...",
                                force=True,
                            )
                            self._emit_status(f"⏳ {_nous_msg}")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue
                            # 无可用 fallback — 返回明确提示
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": (
                                    f"⏳ {_nous_msg}\n\n"
                                    "No fallback provider available. "
                                    "Try again after the reset, or add a "
                                    "fallback provider in config.yaml."
                                ),
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": _nous_msg,
                            }
                    except ImportError:
                        pass
                    except Exception:
                        pass  # 绝不因 rate guard 异常中断 agent 循环

                try:
                    self._reset_stream_delivery_tracking()
                    api_kwargs = self._build_api_kwargs(api_messages)
                    if self._force_ascii_payload:
                        _sanitize_structure_non_ascii(api_kwargs)
                    if self.api_mode == "codex_responses":
                        api_kwargs = self._get_transport().preflight_kwargs(api_kwargs, allow_stream=False)

                    try:
                        from hermes_cli.plugins import invoke_hook as _invoke_hook
                        _invoke_hook(
                            "pre_api_request",
                            task_id=effective_task_id,
                            session_id=self.session_id or "",
                            platform=self.platform or "",
                            model=self.model,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_mode=self.api_mode,
                            api_call_count=api_call_count,
                            message_count=len(api_messages),
                            tool_count=len(self.tools or []),
                            approx_input_tokens=approx_tokens,
                            request_char_count=total_chars,
                            max_tokens=self.max_tokens,
                        )
                    except Exception:
                        pass

                    if env_var_enabled("HERMES_DUMP_REQUESTS"):
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    # [中文] 流式优先策略 — 即使没有显示消费者也用流式
                    # 原因: 流式有细粒度健康检查 (90s 陈旧流检测, 60s 读取超时)
                    # 非流式路径可能导致子 Agent 无限挂起
                    # (当 provider 仅以 SSE ping 保持连接但从不发送响应时)
                    # 无消费者时 streaming 路径对 callback 为空操作，
                    # 且若 provider 不支持 streaming 会自动 fallback 到非流式。
                    def _stop_spinner():
                        nonlocal thinking_spinner
                        if thinking_spinner:
                            thinking_spinner.stop("")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")

                    _use_streaming = True
                    # provider 在之前的尝试中返回了 "stream not supported"
                    # 则在本 session 剩余时间内切换到非 streaming 模式，
                    # 避免每次重试都失败。
                    if getattr(self, "_disable_streaming", False):
                        _use_streaming = False
                    elif not self._has_stream_consumers():
                        # 无显示/TTS 消费者。仍优先使用 streaming 以获取
                        # 健康检查能力，但对测试中的 Mock 客户端跳过
                        # (Mock 返回 SimpleNamespace 而非 streaming 迭代器)。
                        from unittest.mock import Mock
                        if isinstance(getattr(self, "client", None), Mock):
                            _use_streaming = False

                    if _use_streaming:
                        response = self._interruptible_streaming_api_call(
                            api_kwargs, on_first_delta=_stop_spinner
                        )
                    else:
                        response = self._interruptible_api_call(api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    
                    # 静默停止思考 spinner — 后续的响应框或工具执行消息
                    # 更具信息量。
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")

                    if self.verbose_logging:
                        # 记录响应日志 (含 provider 信息)
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    # 校验响应结构是否合法
                    response_invalid = False
                    error_details = []
                    if self.api_mode == "codex_responses":
                        _ct_v = self._get_transport()
                        if not _ct_v.validate_response(response):
                            if response is None:
                                response_invalid = True
                                error_details.append("response is None")
                            else:
                                # output_text fallback: stream backfill 可能失败
                                # 但 normalize 仍可从 output_text 恢复
                                _out_text = getattr(response, "output_text", None)
                                _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                                if _out_text_stripped:
                                    logger.debug(
                                        "Codex response.output is empty but output_text is present "
                                        "(%d chars); deferring to normalization.",
                                        len(_out_text_stripped),
                                    )
                                else:
                                    _resp_status = getattr(response, "status", None)
                                    _resp_incomplete = getattr(response, "incomplete_details", None)
                                    logger.warning(
                                        "Codex response.output is empty after stream backfill "
                                        "(status=%s, incomplete_details=%s, model=%s). %s",
                                        _resp_status, _resp_incomplete,
                                        getattr(response, "model", None),
                                        f"api_mode={self.api_mode} provider={self.provider}",
                                    )
                                    response_invalid = True
                                    error_details.append("response.output is empty")
                    elif self.api_mode == "anthropic_messages":
                        _tv = self._get_transport()
                        if not _tv.validate_response(response):
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            else:
                                error_details.append("response.content invalid (not a non-empty list)")
                    elif self.api_mode == "bedrock_converse":
                        _btv = self._get_transport()
                        if not _btv.validate_response(response):
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            else:
                                error_details.append("Bedrock response invalid (no output or choices)")
                    else:
                        _ctv = self._get_transport()
                        if not _ctv.validate_response(response):
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            elif not hasattr(response, 'choices'):
                                error_details.append("response has no 'choices' attribute")
                            elif response.choices is None:
                                error_details.append("response.choices is None")
                            else:
                                error_details.append("response.choices is empty")

                    if response_invalid:
                        # 打印错误信息前先停止 spinner
                        if thinking_spinner:
                            thinking_spinner.stop("(´;ω;`) oops, retrying...")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")
                        
                        # 无效响应 — 可能是 rate limiting、provider 超时、
                        # 上游服务器错误或响应格式异常。
                        retry_count += 1

                        # 立即 fallback: 空响应/格式异常通常是 rate limit 的
                        # 表现。直接切换 fallback，而非延长 backoff 重试。
                        if self._fallback_index < len(self._fallback_chain):
                            self._emit_status("⚠️ Empty/malformed response — switching to fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                        # 检查响应中的 error 字段 (部分 provider 会包含)
                        error_msg = "Unknown"
                        provider_name = "Unknown"
                        if response and hasattr(response, 'error') and response.error:
                            error_msg = str(response.error)
                            # 尝试从 error 元数据中提取 provider 信息
                            if hasattr(response.error, 'metadata') and response.error.metadata:
                                provider_name = response.error.metadata.get('provider_name', 'Unknown')
                        elif response and hasattr(response, 'message') and response.message:
                            error_msg = str(response.message)
                        
                        # 尝试从 model 字段获取 provider (OpenRouter 通常返回实际使用的模型)
                        if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                            provider_name = f"model={response.model}"
                        
                        # 检查 x-openrouter-provider 或类似元数据
                        if provider_name == "Unknown" and response:
                            # 记录所有响应属性用于调试
                            resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                            if self.verbose_logging:
                                logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                        
                        # 从响应中提取错误码用于上下文诊断
                        _resp_error_code = None
                        if response and hasattr(response, 'error') and response.error:
                            _code_raw = getattr(response.error, 'code', None)
                            if _code_raw is None and isinstance(response.error, dict):
                                _code_raw = response.error.get('code')
                            if _code_raw is not None:
                                try:
                                    _resp_error_code = int(_code_raw)
                                except (TypeError, ValueError):
                                    pass

                        # 根据错误码和响应时间生成可读的失败提示，
                        # 而非默认假设 rate limiting。
                        if _resp_error_code == 524:
                            _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                        elif _resp_error_code == 504:
                            _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                        elif _resp_error_code == 429:
                            _failure_hint = f"rate limited by upstream provider (429)"
                        elif _resp_error_code in (500, 502):
                            _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                        elif _resp_error_code in (503, 529):
                            _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                        elif _resp_error_code is not None:
                            _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                        elif api_duration < 10:
                            _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                        elif api_duration > 60:
                            _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                        else:
                            _failure_hint = f"response time {api_duration:.1f}s"

                        self._vprint(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                        self._vprint(f"{self.log_prefix}   🏢 Provider: {provider_name}", force=True)
                        cleaned_provider_error = self._clean_error_message(error_msg)
                        self._vprint(f"{self.log_prefix}   📝 Provider message: {cleaned_provider_error}", force=True)
                        self._vprint(f"{self.log_prefix}   ⏱️  {_failure_hint}", force=True)
                        
                        if retry_count >= max_retries:
                            # 放弃前先尝试 fallback
                            self._emit_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue
                            self._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                            logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                                "failed": True  # 标记为失败以便过滤
                            }
                        
                        # 重试前 backoff — 抖动指数退避: 5s 基础, 120s 上限
                        wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                        self._vprint(f"{self.log_prefix}⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...", force=True)
                        logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                        
                        # 分段小睡以保持对中断的响应能力
                        sleep_end = time.time() + wait_time
                        _backoff_touch_counter = 0
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                                self._persist_session(messages, conversation_history)
                                self.clear_interrupt()
                                return {
                                    "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
                            # 每 ~30s 发送活跃信号，让 gateway 的不活跃检测
                            # 知道我们在 backoff 等待期间仍存活。
                            _backoff_touch_counter += 1
                            if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                                self._touch_activity(
                                    f"retry backoff ({retry_count}/{max_retries}), "
                                    f"{int(sleep_end - time.time())}s remaining"
                                )
                        continue  # 重试 API 调用

                    # 检查 finish_reason 后继续处理
                    if self.api_mode == "codex_responses":
                        status = getattr(response, "status", None)
                        incomplete_details = getattr(response, "incomplete_details", None)
                        incomplete_reason = None
                        if isinstance(incomplete_details, dict):
                            incomplete_reason = incomplete_details.get("reason")
                        else:
                            incomplete_reason = getattr(incomplete_details, "reason", None)
                        if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"
                    elif self.api_mode == "anthropic_messages":
                        _tfr = self._get_transport()
                        finish_reason = _tfr.map_finish_reason(response.stop_reason)
                    elif self.api_mode == "bedrock_converse":
                        # Bedrock 响应已在 dispatch 时归一化 — 使用 transport
                        _bt_fr = self._get_transport()
                        _bt_fr_nr = _bt_fr.normalize_response(response)
                        finish_reason = _bt_fr_nr.finish_reason
                    else:
                        _cc_fr = self._get_transport()
                        _cc_fr_nr = _cc_fr.normalize_response(response)
                        finish_reason = _cc_fr_nr.finish_reason
                        assistant_message = self._nr_to_assistant_message(_cc_fr_nr)
                        if self._should_treat_stop_as_truncated(
                            finish_reason,
                            assistant_message,
                            messages,
                        ):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                                force=True,
                            )
                            finish_reason = "length"

                    if finish_reason == "length":
                        # [中文] 截断处理 — 模型输出达到 max_output_tokens 限制
                        # 三种子情况: ① 思考预算耗尽 ② 纯文本截断(续接3次) ③ 工具调用截断(重试1次)
                        self._vprint(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

                        # 将截断响应归一化为统一的 OpenAI 风格消息格式，
                        # 使 text-continuation 和 tool-call retry 在
                        # chat_completions、bedrock_converse 和 anthropic_messages
                        # 之间统一工作。对 Anthropic 使用 agent 循环已依赖的
                        # adapter，使重建的临时 assistant 消息与非截断路径
                        # 中追加的消息字节一致。
                        _trunc_msg = None
                        _trunc_transport = self._get_transport()
                        if self.api_mode == "anthropic_messages":
                            _trunc_nr = _trunc_transport.normalize_response(
                                response, strip_tool_prefix=self._is_anthropic_oauth
                            )
                        else:
                            _trunc_nr = _trunc_transport.normalize_response(response)
                        _trunc_msg = self._nr_to_assistant_message(_trunc_nr)

                        _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                        _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

                        # ── 检测思考预算耗尽 ──────────────
                        # 当模型将所有 output tokens 用于推理，
                        # 没有剩余用于响应时，续接重试毫无意义。
                        # 尽早检测并给出针对性错误，避免浪费 3 次 API 调用。
                        # 仅当模型实际产出了推理块但之后无可见文本时，
                        # 才判定为 "思考耗尽"。不使用 <think> 标签的模型
                        # (如 NVIDIA Build 上的 GLM-4.7、minimax) 可能因
                        # 其他原因返回 content=None 或空字符串 —— 这些应
                        # 视为正常截断并允许续接重试，而非思考预算耗尽。
                        _has_think_tags = bool(
                            _trunc_content and re.search(
                                r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                                _trunc_content,
                                re.IGNORECASE,
                            )
                        )
                        _thinking_exhausted = (
                            not _trunc_has_tool_calls
                            and _has_think_tags
                            and (
                                (_trunc_content is not None and not self._has_content_after_think_block(_trunc_content))
                                or _trunc_content is None
                            )
                        )

                        if _thinking_exhausted:
                            _exhaust_error = (
                                "Model used all output tokens on reasoning with none left "
                                "for the response. Try lowering reasoning effort or "
                                "increasing max_tokens."
                            )
                            self._vprint(
                                f"{self.log_prefix}💭 Reasoning exhausted the output token budget — "
                                f"no visible response was produced.",
                                force=True,
                            )
                            # 返回用户友好的消息作为响应，
                            # 使 CLI (响应框) 和 gateway (聊天消息) 都能
                            # 自然展示，而非显示被抑制的错误。
                            _exhaust_response = (
                                "⚠️ **Thinking Budget Exhausted**\n\n"
                                "The model used all its output tokens on reasoning "
                                "and had none left for the actual response.\n\n"
                                "To fix this:\n"
                                "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                                "→ Or switch to a larger/non-reasoning model with `/model`"
                            )
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": _exhaust_response,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": _exhaust_error,
                            }

                        if self.api_mode in ("chat_completions", "bedrock_converse", "anthropic_messages"):
                            assistant_message = _trunc_msg
                            if assistant_message is not None and not _trunc_has_tool_calls:
                                length_continue_retries += 1
                                interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                                messages.append(interim_msg)
                                if assistant_message.content:
                                    truncated_response_prefix += assistant_message.content

                                if length_continue_retries < 3:
                                    self._vprint(
                                        f"{self.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )
                                    continue_msg = {
                                        "role": "user",
                                        "content": (
                                            "[System: Your previous response was truncated by the output "
                                            "length limit. Continue exactly where you left off. Do not "
                                            "restart or repeat prior text. Finish the answer directly.]"
                                        ),
                                    }
                                    messages.append(continue_msg)
                                    self._session_messages = messages
                                    self._save_session_log(messages)
                                    restart_with_length_continuation = True
                                    break

                                partial_response = self._strip_think_blocks(truncated_response_prefix).strip()
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": partial_response or None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response remained truncated after 3 continuation attempts",
                                }

                        if self.api_mode in ("chat_completions", "bedrock_converse", "anthropic_messages"):
                            assistant_message = _trunc_msg
                            if assistant_message is not None and _trunc_has_tool_calls:
                                if truncated_tool_call_retries < 1:
                                    truncated_tool_call_retries += 1
                                    self._vprint(
                                        f"{self.log_prefix}⚠️  Truncated tool call detected — retrying API call...",
                                        force=True,
                                    )
                                    # 不将残缺的响应追加到消息列表；
                                    # 直接从当前消息状态重新运行相同的 API 调用，
                                    # 给模型另一次机会。
                                    continue
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                    force=True,
                                )
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response truncated due to output length limit",
                                }

                        # 如有历史消息，回滚到上一个完整状态
                        if len(messages) > 1:
                            self._vprint(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)

                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)

                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit"
                            }
                        else:
                            # 首条消息即被截断 - 标记为失败
                            self._vprint(f"{self.log_prefix}❌ First response truncated - cannot recover", force=True)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": "First response truncated due to output length limit"
                            }
                    
                    # 跟踪响应中的实际 token 用量，用于上下文管理
                    # [中文] Token 用量跟踪 — 用于上下文压缩决策、成本估算、会话统计
                    if hasattr(response, 'usage') and response.usage:
                        canonical_usage = normalize_usage(
                            response.usage,
                            provider=self.provider,
                            api_mode=self.api_mode,
                        )
                        prompt_tokens = canonical_usage.prompt_tokens
                        completion_tokens = canonical_usage.output_tokens
                        total_tokens = canonical_usage.total_tokens
                        usage_dict = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                        self.context_compressor.update_from_response(usage_dict)

                        # 成功调用后缓存发现的上下文长度。
                        # 仅持久化 provider 确认的限制 (从错误消息解析)，
                        # 而非猜测的探测等级。
                        if getattr(self.context_compressor, "_context_probed", False):
                            ctx = self.context_compressor.context_length
                            if getattr(self.context_compressor, "_context_probe_persistable", False):
                                save_context_length(self.model, self.base_url, ctx)
                                self._safe_print(f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}")
                            self.context_compressor._context_probed = False
                            self.context_compressor._context_probe_persistable = False

                        self.session_prompt_tokens += prompt_tokens
                        self.session_completion_tokens += completion_tokens
                        self.session_total_tokens += total_tokens
                        self.session_api_calls += 1
                        self.session_input_tokens += canonical_usage.input_tokens
                        self.session_output_tokens += canonical_usage.output_tokens
                        self.session_cache_read_tokens += canonical_usage.cache_read_tokens
                        self.session_cache_write_tokens += canonical_usage.cache_write_tokens
                        self.session_reasoning_tokens += canonical_usage.reasoning_tokens

                        # 记录 API 调用详情用于调试/可观测性
                        _cache_pct = ""
                        if canonical_usage.cache_read_tokens and prompt_tokens:
                            _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                        logger.info(
                            "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                            self.session_api_calls, self.model, self.provider or "unknown",
                            prompt_tokens, completion_tokens, total_tokens,
                            api_duration, _cache_pct,
                        )

                        cost_result = estimate_usage_cost(
                            self.model,
                            canonical_usage,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                        )
                        if cost_result.amount_usd is not None:
                            self.session_estimated_cost_usd += float(cost_result.amount_usd)
                        self.session_cost_status = cost_result.status
                        self.session_cost_source = cost_result.source

                        # 将 token 计数持久化到 session DB 用于 /insights。
                        # 对所有带 session_id 的平台执行此操作，确保非 CLI
                        # session (gateway、cron、委托运行) 在高层持久化路径
                        # 被跳过或失败时不丢失 token/计费数据。
                        # Gateway/session-store 写入使用绝对总量，
                        # 安全覆盖这些每次调用的增量而不会重复计数。
                        if self._session_db and self.session_id:
                            try:
                                self._session_db.update_token_counts(
                                    self.session_id,
                                    input_tokens=canonical_usage.input_tokens,
                                    output_tokens=canonical_usage.output_tokens,
                                    cache_read_tokens=canonical_usage.cache_read_tokens,
                                    cache_write_tokens=canonical_usage.cache_write_tokens,
                                    reasoning_tokens=canonical_usage.reasoning_tokens,
                                    estimated_cost_usd=float(cost_result.amount_usd)
                                    if cost_result.amount_usd is not None else None,
                                    cost_status=cost_result.status,
                                    cost_source=cost_result.source,
                                    billing_provider=self.provider,
                                    billing_base_url=self.base_url,
                                    billing_mode="subscription_included"
                                    if cost_result.status == "included" else None,
                                    model=self.model,
                                    api_call_count=1,
                                )
                            except Exception:
                                pass  # 不阻塞 agent 主循环
                        
                        if self.verbose_logging:
                            logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                        
                        # 展示所有报告缓存命中的 provider 的统计信息，
                        # 不仅限于注入 cache_control 标记的 provider。
                        # OpenAI/Kimi/DeepSeek/Qwen 均执行自动服务端前缀缓存，
                        # 返回 prompt_tokens_details.cached_tokens；
                        # 用户此前无法看到缓存命中率，因为该行仅在
                        # _use_prompt_caching 为 True (Anthropic 风格标记注入) 时
                        # 才显示。canonical_usage 已从三种 API 归一化
                        # (Anthropic / Codex / OpenAI-chat)，可直接依赖其值。
                        cached = canonical_usage.cache_read_tokens
                        written = canonical_usage.cache_write_tokens
                        prompt = usage_dict["prompt_tokens"]
                        if (cached or written) and not self.quiet_mode:
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            self._vprint(
                                f"{self.log_prefix}   💾 Cache: "
                                f"{cached:,}/{prompt:,} tokens "
                                f"({hit_pct:.0f}% hit, {written:,} written)"
                            )
                    
                    has_retried_429 = False  # 成功后重置
                    # 成功请求后清除 Nous rate limit 状态 ——
                    # 证明限制已重置，其他 session 可恢复访问 Nous。
                    if self.provider == "nous":
                        try:
                            from agent.nous_rate_guard import clear_nous_rate_limit
                            clear_nous_rate_limit()
                        except Exception:
                            pass
                    self._touch_activity(f"API call #{api_call_count} completed")
                    break  # Success, exit retry loop

                except InterruptedError:
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    api_elapsed = time.time() - api_start_time
                    self._vprint(f"{self.log_prefix}⚡ Interrupted during API call.", force=True)
                    self._persist_session(messages, conversation_history)
                    interrupted = True
                    final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                    break

                except Exception as api_error:
                    # 打印错误信息前先停止 spinner
                    if thinking_spinner:
                        thinking_spinner.stop("(╥_╥) error, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    # -----------------------------------------------------------
                    # UnicodeEncodeError 恢复。两种常见原因:
                    #   1. 来自剪贴板粘贴 (Google Docs、富文本编辑器) 的孤立
                    #      surrogate (U+D800..U+DFFF) — 清洗后重试。
                    #   2. LANG=C 或非 UTF-8 locale 系统 (如 Chromebook) 上的
                    #      ASCII codec — 任何非 ASCII 字符都会失败。
                    #      通过错误消息中的 'ascii' codec 关键字检测。
                    # 就地清洗消息并最多重试两次:
                    # 先清除 surrogate，若需要再做纯 ASCII locale 清洗。
                    # -----------------------------------------------------------
                    if isinstance(api_error, UnicodeEncodeError) and getattr(self, '_unicode_sanitization_passes', 0) < 2:
                        _err_str = str(api_error).lower()
                        _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                        # 检测 surrogate 错误 — utf-8 codec 拒绝编码
                        # U+D800..U+DFFF。错误文本为:
                        #   （错误消息示例）"'utf-8' codec 无法编码位置
                        #    N-M 的字符：不允许 surrogate 字符"
                        _is_surrogate_error = (
                            "surrogate" in _err_str
                            or ("'utf-8'" in _err_str and not _is_ascii_codec)
                        )
                        # 从 canonical messages 列表和 api_messages (API 副本，
                        # 可能携带从 reasoning 转换的 reasoning_content/
                        # reasoning_details 等 canonical 列表没有的字段)
                        # 中清除 surrogate。同时清洗已构建的 api_kwargs
                        # 和 prefill_messages (如存在)。与下方 ASCII
                        # 编解码器恢复如下。
                        _surrogates_found = _sanitize_messages_surrogates(messages)
                        if isinstance(api_messages, list):
                            if _sanitize_messages_surrogates(api_messages):
                                _surrogates_found = True
                        if isinstance(api_kwargs, dict):
                            if _sanitize_structure_surrogates(api_kwargs):
                                _surrogates_found = True
                        if isinstance(getattr(self, "prefill_messages", None), list):
                            if _sanitize_messages_surrogates(self.prefill_messages):
                                _surrogates_found = True
                        # 根据错误类型而非是否找到异常来决定重试 ——
                        # _force_ascii_payload / 上方扩展的 surrogate 遍历
                        # 已覆盖所有已知路径，但新的转换字段仍可能遗漏。
                        # 如果错误是 surrogate 编码失败，始终允许重试；
                        # 主动清洗器会在下次迭代再次运行。
                        # 受 _unicode_sanitization_passes < 2 约束。
                        if _surrogates_found or _is_surrogate_error:
                            self._unicode_sanitization_passes += 1
                            if _surrogates_found:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Stripped invalid surrogate characters from messages. Retrying...",
                                    force=True,
                                )
                            else:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Surrogate encoding error — retrying after full-payload sanitization...",
                                    force=True,
                                )
                            continue
                        if _is_ascii_codec:
                            self._force_ascii_payload = True
                            # ASCII codec: 系统编码完全无法处理非 ASCII 字符。
                            # 清洗 messages/tool schemas 中的所有非 ASCII
                            # 内容并重试。同时清洗 canonical messages 列表
                            # 和 api_messages (重试循环前构建的 API 副本，
                            # 可能包含 messages 中没有的 reasoning_content
                            # 等额外字段)。
                            _messages_sanitized = _sanitize_messages_non_ascii(messages)
                            if isinstance(api_messages, list):
                                _sanitize_messages_non_ascii(api_messages)
                            # 同时清洗已构建的 api_kwargs，避免转换字段中残留的
                            # 非 ASCII 值 (如 extra_body、reasoning_content)
                            # 通过 _build_api_kwargs 缓存路径存活到下次尝试。
                            if isinstance(api_kwargs, dict):
                                _sanitize_structure_non_ascii(api_kwargs)
                            _prefill_sanitized = False
                            if isinstance(getattr(self, "prefill_messages", None), list):
                                _prefill_sanitized = _sanitize_messages_non_ascii(self.prefill_messages)

                            _tools_sanitized = False
                            if isinstance(getattr(self, "tools", None), list):
                                _tools_sanitized = _sanitize_tools_non_ascii(self.tools)

                            _system_sanitized = False
                            if isinstance(active_system_prompt, str):
                                _sanitized_system = _strip_non_ascii(active_system_prompt)
                                if _sanitized_system != active_system_prompt:
                                    active_system_prompt = _sanitized_system
                                    self._cached_system_prompt = _sanitized_system
                                    _system_sanitized = True
                            if isinstance(getattr(self, "ephemeral_system_prompt", None), str):
                                _sanitized_ephemeral = _strip_non_ascii(self.ephemeral_system_prompt)
                                if _sanitized_ephemeral != self.ephemeral_system_prompt:
                                    self.ephemeral_system_prompt = _sanitized_ephemeral
                                    _system_sanitized = True

                            _headers_sanitized = False
                            _default_headers = (
                                self._client_kwargs.get("default_headers")
                                if isinstance(getattr(self, "_client_kwargs", None), dict)
                                else None
                            )
                            if isinstance(_default_headers, dict):
                                _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                            # 清洗 API key — 凭据中的非 ASCII 字符
                            # (如错误粘贴导致的 ʋ 替代 v) 会使 httpx
                            # 在将 Authorization header 编码为 ASCII 时失败。
                            # 这是消息/工具清洗后仍持续出现
                            # UnicodeEncodeError 的最常见原因 (#6843)。
                            _credential_sanitized = False
                            _raw_key = getattr(self, "api_key", None) or ""
                            if _raw_key:
                                _clean_key = _strip_non_ascii(_raw_key)
                                if _clean_key != _raw_key:
                                    self.api_key = _clean_key
                                    if isinstance(getattr(self, "_client_kwargs", None), dict):
                                        self._client_kwargs["api_key"] = _clean_key
                                    # 同时更新活跃的 client — 它持有自己的
                                    # api_key 副本，auth_headers 每次请求时动态读取。
                                    if getattr(self, "client", None) is not None and hasattr(self.client, "api_key"):
                                        self.client.api_key = _clean_key
                                    _credential_sanitized = True
                                    self._vprint(
                                        f"{self.log_prefix}⚠️  API key contained non-ASCII characters "
                                        f"(bad copy-paste?) — stripped them. If auth fails, "
                                        f"re-copy the key from your provider's dashboard.",
                                        force=True,
                                    )

                            # 检测到 ASCII codec 始终重试 ——
                            # _force_ascii_payload 保证下次迭代时
                            # 完整 api_kwargs 负载被清洗。
                            # 即使上方的逐组件检查未发现异常
                            # (如非 ASCII 仅在 api_messages 的
                            # reasoning_content 中)，该标志也会捕获。
                            # 受 _unicode_sanitization_passes < 2 约束。
                            self._unicode_sanitization_passes += 1
                            _any_sanitized = (
                                _messages_sanitized
                                or _prefill_sanitized
                                or _tools_sanitized
                                or _system_sanitized
                                or _headers_sanitized
                                or _credential_sanitized
                            )
                            if _any_sanitized:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                                    force=True,
                                )
                            else:
                                self._vprint(
                                    f"{self.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                                    force=True,
                                )
                            continue

                    status_code = getattr(api_error, "status_code", None)
                    error_context = self._extract_api_error_context(api_error)

                    # ── Classify the error for structured recovery decisions ──
                    # [中文] 错误分类系统 — 将 API 错误分为结构化类别，决定恢复策略:
                    #   FailoverReason.rate_limit → 等待/fallback
                    #   FailoverReason.payload_too_large → 压缩重试
                    #   FailoverReason.context_overflow → 压缩或降低 max_tokens
                    #   FailoverReason.billing → fallback
                    #   FailoverReason.thinking_signature → 剥离 reasoning_details
                    #   FailoverReason.long_context_tier → 降级到 200K 上下文
                    _compressor = getattr(self, "context_compressor", None)
                    _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                    classified = classify_api_error(
                        api_error,
                        provider=getattr(self, "provider", "") or "",
                        model=getattr(self, "model", "") or "",
                        approx_tokens=approx_tokens,
                        context_length=_ctx_len,
                        num_messages=len(api_messages) if api_messages else 0,
                    )
                    logger.debug(
                        "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                        classified.reason.value, classified.status_code,
                        classified.retryable, classified.should_compress,
                        classified.should_rotate_credential, classified.should_fallback,
                    )

                    recovered_with_pool, has_retried_429 = self._recover_with_credential_pool(
                        status_code=status_code,
                        has_retried_429=has_retried_429,
                        classified_reason=classified.reason,
                        error_context=error_context,
                    )
                    if recovered_with_pool:
                        continue
                    if (
                        self.api_mode == "codex_responses"
                        and self.provider == "openai-codex"
                        and status_code == 401
                        and not codex_auth_retry_attempted
                    ):
                        codex_auth_retry_attempted = True
                        if self._try_refresh_codex_client_credentials(force=True):
                            self._vprint(f"{self.log_prefix}🔐 Codex auth refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "chat_completions"
                        and self.provider == "nous"
                        and status_code == 401
                        and not nous_auth_retry_attempted
                    ):
                        nous_auth_retry_attempted = True
                        if self._try_refresh_nous_client_credentials(force=True):
                            print(f"{self.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                            continue
                        # 凭据刷新无效 —— 显示诊断信息。
                        # 最常见原因：Portal OAuth 过期/撤销、
                        # 账户额度用尽、或 agent key 被封禁。
                        from hermes_constants import display_hermes_home as _dhh_fn
                        _dhh = _dhh_fn()
                        _body_text = ""
                        try:
                            _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
                            if _body is not None:
                                _body_text = str(_body)[:200]
                        except Exception:
                            pass
                        print(f"{self.log_prefix}🔐 Nous 401 — Portal authentication failed.")
                        if _body_text:
                            print(f"{self.log_prefix}   Response: {_body_text}")
                        print(f"{self.log_prefix}   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
                        print(f"{self.log_prefix}   Troubleshooting:")
                        print(f"{self.log_prefix}     • Re-authenticate: hermes login --provider nous")
                        print(f"{self.log_prefix}     • Check credits / billing: https://portal.nousresearch.com")
                        print(f"{self.log_prefix}     • Verify stored credentials: {_dhh}/auth.json")
                        print(f"{self.log_prefix}     • Switch providers temporarily: /model <model> --provider openrouter")
                    if (
                        self.api_mode == "anthropic_messages"
                        and status_code == 401
                        and hasattr(self, '_anthropic_api_key')
                        and not anthropic_auth_retry_attempted
                    ):
                        anthropic_auth_retry_attempted = True
                        from agent.anthropic_adapter import _is_oauth_token
                        if self._try_refresh_anthropic_client_credentials():
                            print(f"{self.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                            continue
                        # 凭据刷新无效 —— 显示诊断信息
                        key = self._anthropic_api_key
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                        print(f"{self.log_prefix}🔐 Anthropic 401 — authentication failed.")
                        print(f"{self.log_prefix}   Auth method: {auth_method}")
                        print(f"{self.log_prefix}   Token prefix: {key[:12]}..." if key and len(key) > 12 else f"{self.log_prefix}   Token: (empty or short)")
                        print(f"{self.log_prefix}   Troubleshooting:")
                        from hermes_constants import display_hermes_home as _dhh_fn
                        _dhh = _dhh_fn()
                        print(f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                        print(f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                        print(f"{self.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                        print(f"{self.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                        print(f"{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                        print(f"{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                    # ── Thinking block signature recovery ─────────────────
                    # Anthropic 对完整轮次内容签名思考块。
                    # 任何上游修改（上下文压缩、
                    # 会话截断、消息合并）会使签名失效，
                    # signature → HTTP 400.  Recovery: strip reasoning_details
                    # 从所有消息中移除，使下次重试不发送思考内容。
                    # blocks at all.  One-shot — don't retry infinitely.
                    if (
                        classified.reason == FailoverReason.thinking_signature
                        and not thinking_sig_retry_attempted
                    ):
                        thinking_sig_retry_attempted = True
                        for _m in messages:
                            if isinstance(_m, dict):
                                _m.pop("reasoning_details", None)
                        self._vprint(
                            f"{self.log_prefix}⚠️  Thinking block signature invalid — "
                            f"stripped all thinking blocks, retrying...",
                            force=True,
                        )
                        logging.warning(
                            "%sThinking block signature recovery: stripped "
                            "reasoning_details from %d messages",
                            self.log_prefix, len(messages),
                        )
                        continue

                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    self._touch_activity(
                        f"API error recovery (attempt {retry_count}/{max_retries})"
                    )
                    
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    _error_summary = self._summarize_api_error(api_error)
                    logger.warning(
                        "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                        retry_count,
                        max_retries,
                        error_type,
                        self._client_log_context(),
                        _error_summary,
                    )

                    _provider = getattr(self, "provider", "unknown")
                    _base = getattr(self, "base_url", "unknown")
                    _model = getattr(self, "model", "unknown")
                    _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                    self._vprint(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}", force=True)
                    self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    self._vprint(f"{self.log_prefix}   📝 Error: {_error_summary}", force=True)
                    if status_code and status_code < 500:
                        _err_body = getattr(api_error, "body", None)
                        _err_body_str = str(_err_body)[:300] if _err_body else None
                        if _err_body_str:
                            self._vprint(f"{self.log_prefix}   📋 Details: {_err_body_str}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                    # OpenRouter "no tool endpoints" 错误的可操作提示。
                    # This fires regardless of whether fallback succeeds — the
                    # 用户需要知道为什么他们的模型失败了，
                    # 以便修复提供商路由，而非仅静默回退。
                    if (
                        self._is_openrouter_url()
                        and "support tool use" in error_msg
                    ):
                        self._vprint(
                            f"{self.log_prefix}   💡 No OpenRouter providers for {_model} support tool calling with your current settings.",
                            force=True,
                        )
                        if self.providers_allowed:
                            self._vprint(
                                f"{self.log_prefix}      Your provider_routing.only restriction is filtering out tool-capable providers.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      Try removing the restriction or adding providers that support tools for this model.",
                                force=True,
                            )
                        self._vprint(
                            f"{self.log_prefix}      Check which providers support tools: https://openrouter.ai/models/{_model}",
                            force=True,
                        )

                    # 重试前检查是否有中断请求
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: handling API error ({error_type}: {self._clean_error_message(str(api_error))}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    
                    # 在通用 4xx 处理前检查 413 payload-too-large。
                    # 413 是负载大小错误 — 正确的做法是压缩历史并重试，
                    # 而非立即中止。
                    status_code = getattr(api_error, "status_code", None)

                    # ── Anthropic Sonnet long-context 层级门控 ───────────
                    # 当 Claude Max (或类似) 订阅不包含 1M 上下文层级时，
                    # Anthropic 返回 HTTP 429 "Extra usage is required for
                    # long context requests"。这不是瞬时 rate limit ——
                    # 重试或切换凭据无济于事。将上下文减少到 200k
                    # (标准层级) 并压缩。
                    if classified.reason == FailoverReason.long_context_tier:
                        _reduced_ctx = 200000
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length
                        if old_ctx > _reduced_ctx:
                            compressor.update_model(
                                model=self.model,
                                context_length=_reduced_ctx,
                                base_url=self.base_url,
                                api_key=getattr(self, "api_key", ""),
                                provider=self.provider,
                            )
                            # 上下文探测标志 — 仅在内置 compressor 上设置
                            # (插件引擎自行管理)。
                            if hasattr(compressor, "_context_probed"):
                                compressor._context_probed = True
                                # 不持久化 — 这是订阅层级限制而非模型能力。
                                # 若用户后续启用 extra usage，1M 限制应自动恢复。
                                compressor._context_probe_persistable = False
                            self._vprint(
                                f"{self.log_prefix}⚠️  Anthropic long-context tier "
                                f"requires extra usage — reducing context: "
                                f"{old_ctx:,} → {_reduced_ctx:,} tokens",
                                force=True,
                            )

                        compression_attempts += 1
                        if compression_attempts <= max_compression_attempts:
                            original_len = len(messages)
                            messages, active_system_prompt = self._compress_context(
                                messages, system_message,
                                approx_tokens=approx_tokens,
                                task_id=effective_task_id,
                            )
                            # 压缩创建了新 session — 清除历史记录，
                            # 使 _flush_messages_to_session_db 将压缩后的
                            # 消息写入新 session 而非跳过。
                            conversation_history = None
                            if len(messages) < original_len or old_ctx > _reduced_ctx:
                                self._emit_status(
                                    f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                    f"(was {old_ctx:,}), retrying..."
                                )
                                time.sleep(2)
                                restart_with_compressed_messages = True
                                break
                        # 若压缩耗尽或无效，回退到常规错误处理。

                    # rate limit 错误 (429 或配额耗尽) 时立即 fallback。
                    # 配置了 fallback 模型时，直接切换而非通过指数退避
                    # 消耗重试次数 — 主 provider 不会在重试窗口内恢复。
                    is_rate_limited = classified.reason in (
                        FailoverReason.rate_limit,
                        FailoverReason.billing,
                    )
                    if is_rate_limited and self._fallback_index < len(self._fallback_chain):
                        # 如果凭据池轮换仍可能恢复，则不要立即 fallback。
                        # 凭据池的 retry-then-rotate 循环需要至少再尝试一次 ——
                        # 此处直接跳到 fallback provider 会短路该循环。
                        pool = self._credential_pool
                        pool_may_recover = pool is not None and pool.has_available()
                        if not pool_may_recover:
                            self._emit_status("⚠️ Rate limited — switching to fallback provider...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue

                    # ── Nous Portal: 记录 rate limit 并跳过重试 ─────
                    # 当 Nous 返回 429 时，将重置时间记录到共享文件，
                    # 使所有 session (cron、gateway、辅助进程) 不再叠加请求。
                    # 然后跳过后续重试 — 每次重试消耗另一个 RPH 请求
                    # 并加深 rate limit 限制。重试循环顶部的守卫会在
                    # 下次迭代时捕获此状态并尝试 fallback 或明确退出。
                    if (
                        is_rate_limited
                        and self.provider == "nous"
                        and classified.reason == FailoverReason.rate_limit
                        and not recovered_with_pool
                    ):
                        try:
                            from agent.nous_rate_guard import record_nous_rate_limit
                            _err_resp = getattr(api_error, "response", None)
                            _err_hdrs = (
                                getattr(_err_resp, "headers", None)
                                if _err_resp else None
                            )
                            record_nous_rate_limit(
                                headers=_err_hdrs,
                                error_context=error_context,
                            )
                        except Exception:
                            pass
                        # 直接跳到 max_retries — 循环顶部的守卫会
                        # 处理 fallback 或干净退出。
                        retry_count = max_retries
                        continue

                    is_payload_too_large = (
                        classified.reason == FailoverReason.payload_too_large
                    )

                    if is_payload_too_large:
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        self._emit_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # 压缩创建了新 session — 清除历史记录，
                        # 使 _flush_messages_to_session_db 将压缩后的
                        # 消息写入新 session 而非跳过。
                        conversation_history = None

                        if len(messages) < original_len:
                            self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # 压缩重试前短暂暂停
                            restart_with_compressed_messages = True
                            break
                        else:
                            self._vprint(f"{self.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 payload too large. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Request payload too large (413). Cannot compress further.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }

                    # 在通用 4xx 处理前检查 context-length 错误。
                    # 分类器通过以下方式检测上下文溢出: 显式错误消息、
                    # 通用 400 + 大 session 启发式 (#1630)、
                    # 服务器断开 + 大 session 模式 (#2153)。
                    is_context_length_error = (
                        classified.reason == FailoverReason.context_overflow
                    )

                    if is_context_length_error:
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length

                        # ── 区分两种截然不同的错误 ───────────
                        # 1. "Prompt too long": 输入超出上下文窗口。
                        #    修复: 减小 context_length + 压缩历史。
                        # 2. "max_tokens too large": 输入正常，但
                        #    input_tokens + 请求的 max_tokens > context_window。
                        #    修复: 减小本次调用的 max_tokens (输出上限)。
                        #    不要缩小 context_length — 窗口大小未变。
                        #
                        # 注意: max_tokens = 输出 token 上限 (单次响应)。
                        #       context_length = 总窗口 (输入+输出合计)。
                        available_out = parse_available_output_tokens_from_error(error_msg)
                        if available_out is not None:
                            # 错误纯粹是因为输出上限过大。
                            # 将输出限制在可用空间内并重试，
                            # 不动 context_length 也不触发压缩。
                            safe_out = max(1, available_out - 64)  # 小幅安全余量
                            self._ephemeral_max_output_tokens = safe_out
                            self._vprint(
                                f"{self.log_prefix}⚠️  Output cap too large for current prompt — "
                                f"retrying with max_tokens={safe_out:,} "
                                f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})",
                                force=True,
                            )
                            # 仍计入 compression_attempts 以防止错误持续时无限循环。
                            compression_attempts += 1
                            if compression_attempts > max_compression_attempts:
                                self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                                self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                                logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                                self._persist_session(messages, conversation_history)
                                return {
                                    "messages": messages,
                                    "completed": False,
                                    "api_calls": api_call_count,
                                    "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                    "partial": True,
                                    "failed": True,
                                    "compression_exhausted": True,
                                }
                            restart_with_compressed_messages = True
                            break

                        # 错误是因为输入过大 — 减小 context_length。
                        # 尝试从错误消息中解析实际限制
                        parsed_limit = parse_context_limit_from_error(error_msg)
                        if parsed_limit and parsed_limit < old_ctx:
                            new_ctx = parsed_limit
                            self._vprint(f"{self.log_prefix}⚠️  Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
                        else:
                            # 降低到下一探测等级
                            new_ctx = get_next_probe_tier(old_ctx)

                        if new_ctx and new_ctx < old_ctx:
                            compressor.update_model(
                                model=self.model,
                                context_length=new_ctx,
                                base_url=self.base_url,
                                api_key=getattr(self, "api_key", ""),
                                provider=self.provider,
                            )
                            # 上下文探测标志 — 仅在内置 compressor 上设置
                            # (插件引擎自行管理)。
                            if hasattr(compressor, "_context_probed"):
                                compressor._context_probed = True
                                # 仅持久化从 provider 错误消息解析出的限制 (真实数值)。
                                # get_next_probe_tier() 猜测的 fallback 等级应仅
                                # 保存在内存中 — 持久化会用错误值污染缓存。
                                compressor._context_probe_persistable = bool(
                                    parsed_limit and parsed_limit == new_ctx
                                )
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        self._emit_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # 压缩创建了新 session — 清除历史记录，
                        # 使 _flush_messages_to_session_db 将压缩后的
                        # 消息写入新 session 而非跳过。
                        conversation_history = None

                        if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                            if len(messages) < original_len:
                                self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # 压缩重试前短暂暂停
                            restart_with_compressed_messages = True
                            break
                        else:
                            # 无法进一步压缩且已处于最低等级
                            self._vprint(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                            logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }

                    # 检查不可重试的客户端错误。分类器已处理
                    # 413、429、529 (瞬时)、上下文溢出和通用 400 启发式。
                    # 本地验证错误 (ValueError, TypeError) 是编程 bug。
                    is_local_validation_error = (
                        isinstance(api_error, (ValueError, TypeError))
                        and not isinstance(api_error, UnicodeEncodeError)
                    )
                    is_client_error = (
                        is_local_validation_error
                        or (
                            not classified.retryable
                            and not classified.should_compress
                            and classified.reason not in (
                                FailoverReason.rate_limit,
                                FailoverReason.billing,
                                FailoverReason.overloaded,
                                FailoverReason.context_overflow,
                                FailoverReason.payload_too_large,
                                FailoverReason.long_context_tier,
                                FailoverReason.thinking_signature,
                            )
                        )
                    ) and not is_context_length_error

                    if is_client_error:
                        # 中止前先尝试 fallback — 不同的 provider
                        # 可能不存在相同的问题 (rate limit、auth 等)
                        self._emit_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        if api_kwargs is not None:
                            self._dump_api_request_debug(
                                api_kwargs, reason="non_retryable_client_error", error=api_error,
                            )
                        self._emit_status(
                            f"❌ Non-retryable error (HTTP {status_code}): "
                            f"{self._summarize_api_error(api_error)}"
                        )
                        self._vprint(f"{self.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                        self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                        self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                        # 常见 auth 错误的操作指引
                        if classified.is_auth or classified.reason == FailoverReason.billing:
                            if _provider == "openai-codex" and status_code == 401:
                                self._vprint(f"{self.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                self._vprint(f"{self.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                self._vprint(f"{self.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                self._vprint(f"{self.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            else:
                                self._vprint(f"{self.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                                self._vprint(f"{self.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                                self._vprint(f"{self.log_prefix}      • Does your account have access to {_model}?", force=True)
                                if base_url_host_matches(str(_base), "openrouter.ai"):
                                    self._vprint(f"{self.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                        logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")
                        # 当错误可能与上下文溢出相关 (状态 400 + 大 session) 时，
                        # 跳过 session 持久化。持久化失败的用户消息会使
                        # session 更大，导致下次尝试出现相同失败。(#1630)
                        if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Skipping session persistence "
                                f"for large failed session to prevent growth loop.",
                                force=True,
                            )
                        else:
                            self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": str(api_error),
                        }

                    if retry_count >= max_retries:
                        # fallback 前先尝试重建主 client 一次，
                        # 用于瞬时 transport 错误 (连接池过期、TCP reset)。
                        # 每个 API 调用块仅尝试一次。
                        if not primary_recovery_attempted and self._try_recover_primary_transport(
                            api_error, retry_count=retry_count, max_retries=max_retries,
                        ):
                            primary_recovery_attempted = True
                            retry_count = 0
                            continue
                        # 彻底放弃前先尝试 fallback
                        self._emit_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        _final_summary = self._summarize_api_error(api_error)
                        if is_rate_limited:
                            self._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                        else:
                            self._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                        self._vprint(f"{self.log_prefix}   💀 Final error: {_final_summary}", force=True)

                        # 检测 SSE 流断开模式 (如 "Network connection lost")
                        # 并提供操作指引。这通常发生在模型生成非常大的
                        # tool call (write_file 含大量内容) 且代理/CDN
                        # 在响应中途断开流时。
                        _is_stream_drop = (
                            not getattr(api_error, "status_code", None)
                            and any(p in error_msg for p in (
                                "connection lost", "connection reset",
                                "connection closed", "network connection",
                                "network error", "terminated",
                            ))
                        )
                        if _is_stream_drop:
                            self._vprint(
                                f"{self.log_prefix}   💡 The provider's stream "
                                f"connection keeps dropping. This often happens "
                                f"when the model tries to write a very large "
                                f"file in a single tool call.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      Try asking the model "
                                f"to use execute_code with Python's open() for "
                                f"large files, or to write the file in smaller "
                                f"sections.",
                                force=True,
                            )

                        logging.error(
                            "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                            self.log_prefix, max_retries, _final_summary,
                            _provider, _model, len(api_messages), f"{approx_tokens:,}",
                        )
                        if api_kwargs is not None:
                            self._dump_api_request_debug(
                                api_kwargs, reason="max_retries_exhausted", error=api_error,
                            )
                        self._persist_session(messages, conversation_history)
                        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                        if _is_stream_drop:
                            _final_response += (
                                "\n\nThe provider's stream connection keeps "
                                "dropping — this often happens when generating "
                                "very large tool call responses (e.g. write_file "
                                "with long content). Try asking me to use "
                                "execute_code with Python's open() for large "
                                "files, or to write in smaller sections."
                            )
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _final_summary,
                        }

                    # 对于速率限制，优先使用 Retry-After 响应头
                    _retry_after = None
                    if is_rate_limited:
                        _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                        if _resp_headers and hasattr(_resp_headers, "get"):
                            _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                            if _ra_raw:
                                try:
                                    _retry_after = min(int(_ra_raw), 120)  # 上限 2 分钟
                                except (TypeError, ValueError):
                                    pass
                    wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                    if is_rate_limited:
                        self._emit_status(f"⏱️ Rate limited. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries})...")
                    else:
                        self._emit_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                    logger.warning(
                        "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                        wait_time,
                        retry_count,
                        max_retries,
                        self._client_log_context(),
                        api_error,
                    )
                    # 分段小睡以快速响应中断，避免在单个 sleep() 调用中阻塞整个等待时间
                    sleep_end = time.time() + wait_time
                    _backoff_touch_counter = 0
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)  # 每 200ms 检查一次中断
                        # 每约 30s 刷新一次活动状态，让网关的不活跃监控
                        # 知道我们在退避等待期间仍然存活。
                        _backoff_touch_counter += 1
                        if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                            self._touch_activity(
                                f"error retry backoff ({retry_count}/{max_retries}), "
                                f"{int(sleep_end - time.time())}s remaining"
                            )
            
            # 如果 API 调用被中断，跳过响应处理
            if interrupted:
                _turn_exit_reason = "interrupted_during_api_call"
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                # 将压缩重启计入重试限制，防止压缩减少了消息但仍不够
                # 放入上下文窗口时出现无限循环。
                retry_count += 1
                restart_with_compressed_messages = False
                continue

            if restart_with_length_continuation:
                # 每次重试逐步提升输出 token 预算。
                # 重试 1 → 2× 基础，重试 2 → 3× 基础，上限 32768。
                # 通过 _ephemeral_max_output_tokens 对所有 provider 生效。
                _boost_base = self.max_tokens if self.max_tokens else 4096
                _boost = _boost_base * (length_continue_retries + 1)
                self._ephemeral_max_output_tokens = min(_boost, 32768)
                continue

            # 守卫: 如果所有重试耗尽仍无成功响应
            # (如重复的 context-length 错误耗尽 retry_count)，
            # response 变量仍为 None。干净地退出循环。
            if response is None:
                _turn_exit_reason = "all_retries_exhausted_no_response"
                print(f"{self.log_prefix}❌ All API retries exhausted with no successful response.")
                self._persist_session(messages, conversation_history)
                break

            try:
                _transport = self._get_transport()
                _normalize_kwargs = {}
                if self.api_mode == "anthropic_messages":
                    _normalize_kwargs["strip_tool_prefix"] = self._is_anthropic_oauth
                _nr = _transport.normalize_response(response, **_normalize_kwargs)
                assistant_message = self._nr_to_assistant_message(_nr)
                finish_reason = _nr.finish_reason
                
                # 将 content 归一化为字符串 — 部分 OpenAI 兼容服务器
                # (llama-server 等) 返回 dict 或 list 而非纯字符串，
                # 会导致下游 .strip() 调用崩溃。
                if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                    raw = assistant_message.content
                    if isinstance(raw, dict):
                        assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                    elif isinstance(raw, list):
                        # 多模态 content 列表 — 提取文本部分
                        parts = []
                        for part in raw:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                        assistant_message.content = "\n".join(parts)
                    else:
                        assistant_message.content = str(raw)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                    _assistant_text = assistant_message.content or ""
                    _invoke_hook(
                        "post_api_request",
                        task_id=effective_task_id,
                        session_id=self.session_id or "",
                        platform=self.platform or "",
                        model=self.model,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_mode=self.api_mode,
                        api_call_count=api_call_count,
                        api_duration=api_duration,
                        finish_reason=finish_reason,
                        message_count=len(api_messages),
                        response_model=getattr(response, "model", None),
                        usage=self._usage_summary_for_api_request_hook(response),
                        assistant_content_chars=len(_assistant_text),
                        assistant_tool_call_count=len(_assistant_tool_calls),
                    )
                except Exception:
                    pass

                # 处理 assistant 响应
                if assistant_message.content and not self.quiet_mode:
                    if self.verbose_logging:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
                    else:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

                # 通知进度回调模型的思考内容（用于 subagent 委托，
                # 将子 agent 的推理过程转发到父级显示）。
                if (assistant_message.content and self.tool_progress_callback):
                    _think_text = assistant_message.content.strip()
                    # 移除不应泄露到父级显示的推理 XML 标签
                    _think_text = re.sub(
                        r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                    ).strip()
                    # 对于 subagent: 将首行转发到父级显示（保留已有行为）。
                    # 对于所有带结构化回调的 agent: 发出 reasoning.available 事件。
                    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                    if first_line and getattr(self, '_delegate_depth', 0) > 0:
                        try:
                            self.tool_progress_callback("_thinking", first_line)
                        except Exception:
                            pass
                    elif _think_text:
                        try:
                            self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                        except Exception:
                            pass
                
                # 检测未闭合的 <REASONING_SCRATCHPAD>（已打开但未关闭）
                # 说明模型在推理过程中耗尽输出 token — 最多重试 2 次
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    self._incomplete_scratchpad_retries += 1
                    
                    self._vprint(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                    
                    if self._incomplete_scratchpad_retries <= 2:
                        self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                        # 不添加损坏的消息，直接重试
                        continue
                    else:
                        # 重试上限 — 丢弃本轮并保存为 partial
                        self._vprint(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                        self._incomplete_scratchpad_retries = 0
                        
                        rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)
                        
                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                        }
                
                # 响应正常时重置不完整 scratchpad 计数器
                self._incomplete_scratchpad_retries = 0

                if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                    self._codex_incomplete_retries += 1

                    interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                    interim_has_content = bool((interim_msg.get("content") or "").strip())
                    interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))

                    if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning:
                        last_msg = messages[-1] if messages else None
                        # 去重检测：两条连续的 incomplete assistant 消息
                        # 若 content 和 reasoning 完全相同则折叠。
                        # 对于纯推理消息（codex_reasoning_items 不同但
                        # 可见 content/reasoning 均为空），也比对加密项
                        # 以避免静默丢弃新状态。
                        last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                        interim_codex_items = interim_msg.get("codex_reasoning_items")
                        duplicate_interim = (
                            isinstance(last_msg, dict)
                            and last_msg.get("role") == "assistant"
                            and last_msg.get("finish_reason") == "incomplete"
                            and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                            and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                            and last_codex_items == interim_codex_items
                        )
                        if not duplicate_interim:
                            messages.append(interim_msg)
                            self._emit_interim_assistant_message(interim_msg)

                    if self._codex_incomplete_retries < 3:
                        if not self.quiet_mode:
                            self._vprint(f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)")
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    self._codex_incomplete_retries = 0
                    self._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Codex response remained incomplete after 3 continuation attempts",
                    }
                elif hasattr(self, "_codex_incomplete_retries"):
                    self._codex_incomplete_retries = 0
                
                # 检查工具调用
                # [中文] 工具调用分支 — 当模型返回 tool_calls 时进入:
                #   1. 验证工具名 (模糊匹配自修复, 无效名最多重试3次)
                #   2. 验证 JSON 参数 (截断检测, 格式错误最多重试3次)
                #   3. 限制/去重 delegate_task 调用
                #   4. 执行工具 (_execute_tool_calls, 并发或顺序)
                #   5. 检查是否需要上下文压缩
                #   6. continue → 下一轮迭代
                if assistant_message.tool_calls:
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
                    # 验证工具调用名称 — 检测模型幻觉
                    # 验证前先修复不匹配的工具名
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in self.valid_tool_names:
                            repaired = self._repair_tool_call(tc.function.name)
                            if repaired:
                                print(f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                                tc.function.name = repaired
                    invalid_tool_calls = [
                        tc.function.name for tc in assistant_message.tool_calls
                        if tc.function.name not in self.valid_tool_names
                    ]
                    if invalid_tool_calls:
                        # 追踪无效工具调用的重试次数
                        self._invalid_tool_retries += 1

                        # 向模型返回有用的错误信息 — 模型可在下一轮自行修正
                        available = ", ".join(sorted(self.valid_tool_names))
                        invalid_name = invalid_tool_calls[0]
                        invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                        self._vprint(f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)")

                        if self._invalid_tool_retries >= 3:
                            self._vprint(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                            self._invalid_tool_retries = 0
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}"
                            }

                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        messages.append(assistant_msg)
                        for tc in assistant_message.tool_calls:
                            if tc.function.name not in self.valid_tool_names:
                                content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                            else:
                                content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": content,
                            })
                        continue
                    # 工具调用验证成功后重置重试计数器
                    self._invalid_tool_retries = 0
                    
                    # 验证工具调用参数是否为合法 JSON
                    # 将空字符串视为空对象（常见模型怪癖）
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, (dict, list)):
                            tc.function.arguments = json.dumps(args)
                            continue
                        if args is not None and not isinstance(args, str):
                            tc.function.arguments = str(args)
                            args = tc.function.arguments
                        # 将空白字符串视为空对象
                        if not args or not args.strip():
                            tc.function.arguments = "{}"
                            continue
                        try:
                            json.loads(args)
                        except json.JSONDecodeError as e:
                            invalid_json_args.append((tc.function.name, str(e)))
                    
                    if invalid_json_args:
                        # 判断无效 JSON 是因为截断还是模型格式错误。
                        # 路由器有时会将 finish_reason 从 "length" 改写为
                        # "tool_calls"，使上方的长度处理器无法检测截断。
                        # 截断检测：参数末尾不是 } 或 ]（去除空白后）
                        # 即表示被中途截断。
                        _truncated = any(
                            not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                            for tc in assistant_message.tool_calls
                            if tc.function.name in {n for n, _ in invalid_json_args}
                        )
                        if _truncated:
                            self._vprint(
                                f"{self.log_prefix}⚠️  Truncated tool call arguments detected "
                                f"(finish_reason={finish_reason!r}) — refusing to execute.",
                                force=True,
                            )
                            self._invalid_json_retries = 0
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit",
                            }

                        # 追踪无效 JSON 参数的重试次数
                        self._invalid_json_retries += 1

                        tool_name, error_msg = invalid_json_args[0]
                        self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                        if self._invalid_json_retries < 3:
                            self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            # 不向消息添加任何内容，直接重试 API 调用
                            continue
                        else:
                            # 不返回 partial，而是注入工具错误结果让模型自行恢复。
                            # 使用 tool 结果（而非 user 消息）以保持角色交替。
                            self._vprint(f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                            self._invalid_json_retries = 0  # Reset for next attempt
                            
                            # 追加带有（损坏的）tool_calls 的 assistant 消息
                            recovery_assistant = self._build_assistant_message(assistant_message, finish_reason)
                            messages.append(recovery_assistant)
                            
                            # 为每个工具调用返回错误结果
                            invalid_names = {name for name, _ in invalid_json_args}
                            for tc in assistant_message.tool_calls:
                                if tc.function.name in invalid_names:
                                    err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                    tool_result = (
                                        f"Error: Invalid JSON arguments. {err}. "
                                        f"For tools with no required parameters, use an empty object: {{}}. "
                                        f"Please retry with valid JSON."
                                    )
                                else:
                                    tool_result = "Skipped: other tool call in this response had invalid JSON."
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                })
                            continue
                    
                    # JSON 验证成功后重置重试计数器
                    self._invalid_json_retries = 0

                    # ── 调用后防护 ──────────────────────────
                    assistant_message.tool_calls = self._cap_delegate_task_calls(
                        assistant_message.tool_calls
                    )
                    assistant_message.tool_calls = self._deduplicate_tool_calls(
                        assistant_message.tool_calls
                    )

                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    # 若本轮同时有 content 和 tool_calls，捕获 content 作为
                    # fallback 最终响应。常见模式：模型在同一轮中给出答案
                    # 并调用 memory/skill 工具。若工具执行后的下一轮为空
                    # 则使用此内容。
                    turn_content = assistant_message.content or ""
                    if turn_content and self._has_content_after_think_block(turn_content):
                        self._last_content_with_tools = turn_content
                        # 仅当本轮所有工具调用都是响应后维护类（memory、todo、
                        # skill_manage 等）时才静默后续输出。若有实质工具
                        # （search_files、read_file、write_file、terminal 等），
                        # 保持输出可见以让用户看到进度。
                        _HOUSEKEEPING_TOOLS = frozenset({
                            "memory", "todo", "skill_manage", "session_search",
                        })
                        _all_housekeeping = all(
                            tc.function.name in _HOUSEKEEPING_TOOLS
                            for tc in assistant_message.tool_calls
                        )
                        self._last_content_tools_all_housekeeping = _all_housekeeping
                        if _all_housekeeping and self._has_stream_consumers():
                            self._mute_post_response = True
                        elif self._should_emit_quiet_tool_messages():
                            clean = self._strip_think_blocks(turn_content).strip()
                            if clean:
                                self._vprint(f"  ┊ 💬 {clean}")
                    
                    # 追加前弹出纯思考预填充消息
                    # （tool-call 路径 — 与 final-response 路径同理）。
                    _had_prefill = False
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()
                        _had_prefill = True

                    # 工具调用跟随 prefill 恢复后重置计数器。
                    # 否则计数器会在整个对话中累积 — 间歇性空响应的模型
                    # （empty → prefill → tools → empty → prefill → tools）
                    # 会耗尽两次 prefill 尝试，第三次空响应无法恢复。
                    # 在此处重置，使每次工具调用成功视为全新开始。
                    if _had_prefill:
                        self._thinking_prefill_retries = 0
                        self._empty_content_retries = 0
                    # 工具执行成功 — 重置工具后 nudge 标志，
                    # 以便模型在后续工具轮次中再次为空时可触发。
                    self._post_tool_empty_retried = False

                    messages.append(assistant_msg)
                    self._emit_interim_assistant_message(assistant_msg)

                    # 工具执行前关闭所有已打开的流式显示（响应框、推理框）。
                    # 中间轮次可能已流式输出并打开了响应框；
                    # 在此处刷新可防止其包裹工具输出行。
                    # 仅通知显示回调 — TTS（_stream_callback）
                    # 不应收到 None（它将 None 用作流结束信号）。
                    if self.stream_delta_callback:
                        try:
                            self.stream_delta_callback(None)
                        except Exception:
                            pass

                    # [中文] 执行所有工具调用 — 根据工具类型选择并发(ThreadPoolExecutor)或顺序执行
                    # 插件钩子 pre_tool_call 可阻止执行, post_tool_call 为观察性质
                    # 结果经过三层预算系统: 每工具阈值(100K) → 每轮聚合(200K) → 预览(1.5K)
                    self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                    # 工具执行成功后重置每轮重试计数器，
                    # 避免单次截断影响整个对话。
                    truncated_tool_call_retries = 0

                    # 标记下次流式文本输出前需要段落分隔。
                    # 不立即输出，因为连续的工具迭代会产生多余空行。
                    # 改由 _fire_stream_delta() 在下次实际文本到达时
                    # 前置一个 "\n\n"。
                    self._stream_needs_break = True

                    # 若本轮仅调用了 execute_code（编程式工具调用），
                    # 则退还迭代次数。这些廉价的 RPC 调用不应消耗预算。
                    _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                    if _tc_names == {"execute_code"}:
                        self.iteration_budget.refund()
                    
                    # ── 上下文压缩检查 ──
                    # 工具执行完成后，检查是否需要压缩上下文。
                    #
                    # Token 数的选择策略:
                    #   使用 API 响应中的 prompt_tokens（而非 prompt_tokens + completion_tokens）。
                    #
                    #   原因：completion_tokens 和 reasoning_tokens 不占用上下文窗口空间——
                    #   它们代表模型输出，不会传递给下一次 API 调用。
                    #
                    #   特别地，思考模型（GLM-5.1、QwQ、DeepSeek R1）会在推理阶段产生
                    #   大量 completion_tokens（几十万 reasoning tokens），如果将其计入，
                    #   会错误地过早触发压缩。只用 prompt_tokens 避免了这个问题。(#12026)
                    #
                    # 零值回退:
                    #   若 last_prompt_tokens == 0（API 断连后状态过期，或提供商未返回
                    #   usage 数据），回退到本地粗略估算 estimate_messages_tokens_rough()。
                    #   否则断连后 should_compress(0) 永不触发，会话会无限增长（#2153）。
                    #
                    # 阈值余量:
                    #   触发阈值默认 50% 上下文窗口，留有充足空间。此时刚追加的
                    #   工具结果尚未计入 token 数——如果它们把总量推到超出窗口，
                    #   下一次 API 调用会报告实际总量并触发压缩。
                    _compressor = self.context_compressor
                    if _compressor.last_prompt_tokens > 0:
                        _real_tokens = _compressor.last_prompt_tokens
                    else:
                        _real_tokens = estimate_messages_tokens_rough(messages)

                    # [中文] 工具执行后检查是否需要上下文压缩 — 基于实际 token 数决定
                    if self.compression_enabled and _compressor.should_compress(_real_tokens):
                        self._safe_print("  ⟳ compacting context…")
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=self.context_compressor.last_prompt_tokens,
                            task_id=effective_task_id,
                        )
                        # 压缩创建了新 session_id，旧 session 被标记为 end_reason='compression'。
                        # 清除 conversation_history，使后续的 _flush_messages_to_session_db
                        # 将压缩后的消息重新写入新 session（否则会追加到旧 session）。
                        conversation_history = None
                    
                    # 增量保存会话日志（即使中断也能看到进度）
                    self._session_messages = messages
                    self._save_session_log(messages)
                    
                    # 继续循环获取下一次响应
                    continue
                
                else:
                    # 无工具调用 — 这是最终响应
                    # [中文] 最终回复分支 — 模型不再调用工具，产出最终文本:
                    #   空响应恢复策略 (6层):
                    #     ① 部分流恢复 — 已流式传输给用户的内容
                    #     ② 前轮后备 — housekeeping 工具后模型已说过的内容
                    #     ③ 工具后空响应 nudge — 注入用户提示让模型继续
                    #     ④ 思考预填充续接 — 模型有推理但无可见文本 (最多2次)
                    #     ⑤ 空响应重试 — 直接重试 API 调用 (最多3次)
                    #     ⑥ Fallback 提供商切换
                    #     ⑦ 终态: "(empty)"
                    final_response = assistant_message.content or ""
                    
                    # 修复：进入无工具调用分支时取消静默，
                    # 以便用户看到空响应警告和恢复状态消息。
                    # _mute_post_response 在之前的维护工具轮次中设置，
                    # 不应静默最终响应路径。
                    self._mute_post_response = False
                    
                    # 检查响应是否只有 think 块而无实际内容
                    if not self._has_content_after_think_block(final_response):
                        # ── 部分流恢复 ─────────────────────
                        # 若内容在连接断开前已流式传输给用户，
                        # 直接用作最终响应，而非落入前轮 fallback
                        # 或浪费 API 调用进行重试。
                        _partial_streamed = (
                            getattr(self, "_current_streamed_assistant_text", "") or ""
                        )
                        if self._has_content_after_think_block(_partial_streamed):
                            _turn_exit_reason = "partial_stream_recovery"
                            _recovered = self._strip_think_blocks(_partial_streamed).strip()
                            logger.info(
                                "Partial stream content delivered (%d chars) "
                                "— using as final response",
                                len(_recovered),
                            )
                            self._emit_status(
                                "↻ Stream interrupted — using delivered content "
                                "as final response"
                            )
                            final_response = _recovered
                            self._response_was_previewed = True
                            break

                        # 若前一轮已在 HOUSEKEEPING 工具调用旁提供了实际内容
                        # （如 "You're welcome!" + memory 保存），模型没有更多
                        # 可说的。直接使用先前内容，避免浪费 API 调用重试。
                        # 注意：仅当该轮所有工具均为维护类（memory、todo 等）
                        # 时才使用此快捷路径。若调用了实质工具（terminal、
                        # search_files 等），内容可能是任务中叙述（"I'll scan
                        # the directory..."），空后续说明模型卡住了 — 应由
                        # 下方的工具后 nudge 处理，而非提前退出。
                        fallback = getattr(self, '_last_content_with_tools', None)
                        if fallback and getattr(self, '_last_content_tools_all_housekeeping', False):
                            _turn_exit_reason = "fallback_prior_turn_content"
                            logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                            self._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                            self._last_content_with_tools = None
                            self._last_content_tools_all_housekeeping = False
                            self._empty_content_retries = 0
                            # 不要修改 assistant 消息内容 — 旧代码注入的
                            # "Calling the X tools..." 会污染对话历史。
                            # 直接使用 fallback 文本作为最终响应并退出。
                            final_response = self._strip_think_blocks(fallback).strip()
                            self._response_was_previewed = True
                            break

                        # ── 工具调用后空响应 nudge ───────────
                        # 模型在执行工具调用后返回空响应。
                        # 涵盖两种情况：
                        #  (a) 前轮完全无内容 — 模型沉默
                        #  (b) 前轮有内容 + 实质工具（上方 fallback 因
                        #      内容是任务中叙述而非最终答案而跳过）
                        # 不放弃，而是通过追加用户级提示 nudge 模型继续。
                        # 这是 #9400 的场景：较弱模型（mimo-v2-pro、
                        # GLM-5 等）有时在工具结果后返回空而非继续。
                        # 一次 nudge 重试通常能修复。
                        _prior_was_tool = any(
                            m.get("role") == "tool"
                            for m in messages[-5:]  # check recent messages
                        )
                        if (
                            _prior_was_tool
                            and not getattr(self, "_post_tool_empty_retried", False)
                        ):
                            self._post_tool_empty_retried = True
                            # 清除过期叙述内容，防止在 nudge 后
                            # 的后续空响应中重新出现。
                            self._last_content_with_tools = None
                            self._last_content_tools_all_housekeeping = False
                            logger.info(
                                "Empty response after tool calls — nudging model "
                                "to continue processing"
                            )
                            self._emit_status(
                                "⚠️ Model returned empty after tool calls — "
                                "nudging to continue"
                            )
                            # 先追加空 assistant 消息以保持消息序列有效：
                            #   tool(result) → assistant("(empty)") → user(nudge)
                            # 否则会形成 tool → user 序列，大多数 API 会拒绝。
                            _nudge_msg = self._build_assistant_message(assistant_message, finish_reason)
                            _nudge_msg["content"] = "(empty)"
                            messages.append(_nudge_msg)
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You just executed tool calls but returned an "
                                    "empty response. Please process the tool "
                                    "results above and continue with the task."
                                ),
                            })
                            continue

                        # ── 纯思考 prefill 续接 ──────────
                        # 模型产生了结构化推理（通过 API 字段）
                        # 但无可见文本内容。不放弃，直接追加
                        # assistant 消息并继续 — 模型在下一轮
                        # 会看到自己的推理并产出文本部分。
                        # 灵感来自 clawdbot 的 "incomplete-text" 恢复。
                        _has_structured = bool(
                            getattr(assistant_message, "reasoning", None)
                            or getattr(assistant_message, "reasoning_content", None)
                            or getattr(assistant_message, "reasoning_details", None)
                        )
                        if _has_structured and self._thinking_prefill_retries < 2:
                            self._thinking_prefill_retries += 1
                            logger.info(
                                "Thinking-only response (no visible content) — "
                                "prefilling to continue (%d/2)",
                                self._thinking_prefill_retries,
                            )
                            self._emit_status(
                                f"↻ Thinking-only response — prefilling to continue "
                                f"({self._thinking_prefill_retries}/2)"
                            )
                            interim_msg = self._build_assistant_message(
                                assistant_message, "incomplete"
                            )
                            interim_msg["_thinking_prefill"] = True
                            messages.append(interim_msg)
                            self._session_messages = messages
                            self._save_session_log(messages)
                            continue

                        # ── 空响应重试 ──────────────────────
                        # 模型未返回可用内容。最多重试 3 次后再
                        # 尝试 fallback。涵盖真正的空响应（无 content、
                        # 无 reasoning）和 prefill 耗尽后的纯推理响应
                        # — 如 mimo-v2-pro 通过 OpenRouter 始终填充
                        # reasoning 字段，旧的 `not _has_structured` 守卫
                        # 会在 prefill 后阻止所有推理模型的重试。
                        _truly_empty = not self._strip_think_blocks(
                            final_response
                        ).strip()
                        _prefill_exhausted = (
                            _has_structured
                            and self._thinking_prefill_retries >= 2
                        )
                        if _truly_empty and (not _has_structured or _prefill_exhausted) and self._empty_content_retries < 3:
                            self._empty_content_retries += 1
                            logger.warning(
                                "Empty response (no content or reasoning) — "
                                "retry %d/3 (model=%s)",
                                self._empty_content_retries, self.model,
                            )
                            self._emit_status(
                                f"⚠️ Empty response from model — retrying "
                                f"({self._empty_content_retries}/3)"
                            )
                            continue

                        # ── 重试耗尽 — 尝试 fallback 提供商 ──
                        # 在放弃并返回 "(empty)" 之前，尝试切换到
                        # fallback 链中的下一个提供商。涵盖模型
                        # （如 GLM-4.5-Air）因上下文降级或提供商问题
                        # 持续返回空响应的情况。
                        if _truly_empty and self._fallback_chain:
                            logger.warning(
                                "Empty response after %d retries — "
                                "attempting fallback (model=%s, provider=%s)",
                                self._empty_content_retries, self.model,
                                self.provider,
                            )
                            self._emit_status(
                                "⚠️ Model returning empty responses — "
                                "switching to fallback provider..."
                            )
                            if self._try_activate_fallback():
                                self._empty_content_retries = 0
                                self._emit_status(
                                    f"↻ Switched to fallback: {self.model} "
                                    f"({self.provider})"
                                )
                                logger.info(
                                    "Fallback activated after empty responses: "
                                    "now using %s on %s",
                                    self.model, self.provider,
                                )
                                continue

                        # 重试和 fallback 链均已耗尽（或未配置 fallback）。
                        # 落入 "(empty)" 终态。
                        _turn_exit_reason = "empty_response_exhausted"
                        reasoning_text = self._extract_reasoning(assistant_message)
                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        assistant_msg["content"] = "(empty)"
                        messages.append(assistant_msg)

                        if reasoning_text:
                            reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                            logger.warning(
                                "Reasoning-only response (no visible content) "
                                "after exhausting retries and fallback. "
                                "Reasoning: %s", reasoning_preview,
                            )
                            self._emit_status(
                                "⚠️ Model produced reasoning but no visible "
                                "response after all retries. Returning empty."
                            )
                        else:
                            logger.warning(
                                "Empty response (no content or reasoning) "
                                "after %d retries. No fallback available. "
                                "model=%s provider=%s",
                                self._empty_content_retries, self.model,
                                self.provider,
                            )
                            self._emit_status(
                                "❌ Model returned no content after all retries"
                                + (" and fallback attempts." if self._fallback_chain else
                                   ". No fallback providers configured.")
                            )

                        final_response = "(empty)"
                        break
                    
                    # 内容成功时重置重试计数器/签名
                    self._empty_content_retries = 0
                    self._thinking_prefill_retries = 0

                    if (
                        self.api_mode == "codex_responses"
                        and self.valid_tool_names
                        and codex_ack_continuations < 2
                        and self._looks_like_codex_intermediate_ack(
                            user_message=user_message,
                            assistant_content=final_response,
                            messages=messages,
                        )
                    ):
                        codex_ack_continuations += 1
                        interim_msg = self._build_assistant_message(assistant_message, "incomplete")
                        messages.append(interim_msg)
                        self._emit_interim_assistant_message(interim_msg)

                        continue_msg = {
                            "role": "user",
                            "content": (
                                "[System: Continue now. Execute the required tool calls and only "
                                "send your final answer after completing the task.]"
                            ),
                        }
                        messages.append(continue_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    codex_ack_continuations = 0

                    if truncated_response_prefix:
                        final_response = truncated_response_prefix + final_response
                        truncated_response_prefix = ""
                        length_continue_retries = 0
                    
                    # 从用户可见响应中移除 <think> 块（消息中保留原始内容用于轨迹记录）
                    final_response = self._strip_think_blocks(final_response).strip()
                    
                    final_msg = self._build_assistant_message(assistant_message, finish_reason)

                    # 追加最终响应前弹出纯思考预填充消息。
                    # 避免连续 assistant 消息破坏严格交替提供商
                    # （Anthropic Messages API）并保持历史干净。
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()

                    messages.append(final_msg)
                    
                    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    logger.error(error_msg)
                
                logger.debug("Outer loop error in API call #%d", api_call_count, exc_info=True)
                
                # 若已追加带有 tool_calls 的 assistant 消息，
                # API 期望每个 tool_call_id 都有 role="tool" 的结果。
                # 为尚未应答的调用填充错误结果。
                for idx in range(len(messages) - 1, -1, -1):
                    msg = messages[idx]
                    if not isinstance(msg, dict):
                        break
                    if msg.get("role") == "tool":
                        continue
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        answered_ids = {
                            m["tool_call_id"]
                            for m in messages[idx + 1:]
                            if isinstance(m, dict) and m.get("role") == "tool"
                        }
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict): continue
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                messages.append(err_msg)
                    break
                
                # 非工具错误不需要注入合成消息。
                # 错误已打印给用户（上方），重试循环继续。
                # 注入虚假的 user/assistant 消息会污染历史、
                # 消耗 token，并有破坏角色交替不变量的风险。

                # 接近上限时退出以避免无限循环
                if api_call_count >= self.max_iterations - 1:
                    _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    # 以 assistant 角色追加，使历史对会话恢复有效
                    # （避免连续 user 消息）。
                    messages.append({"role": "assistant", "content": final_response})
                    break
        
        # ═══════════════════════════════════════════════════════════════
        # [阶段六: 循环后处理] 保存轨迹、清理资源、持久化会话
        # ═══════════════════════════════════════════════════════════════
        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ):
            # 预算耗尽 — 通过一次剥离工具的额外 API 调用请求模型生成摘要。
            # _handle_max_iterations 注入一条 user 消息并发起单次无工具请求。
            _turn_exit_reason = f"max_iterations_reached({api_call_count}/{self.max_iterations})"
            self._emit_status(
                f"⚠️ Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                "— asking model to summarise"
            )
            if not self.quiet_mode:
                self._safe_print(
                    f"\n⚠️  Iteration budget exhausted ({api_call_count}/{self.max_iterations}) "
                    "— requesting summary..."
                )
            final_response = self._handle_max_iterations(messages, api_call_count)
        
        # 判断对话是否成功完成
        completed = final_response is not None and api_call_count < self.max_iterations

        # 若启用则保存训练轨迹。``user_message`` 可能是多模态
        # [中文] 后处理: 保存训练轨迹 → 清理 VM/浏览器资源 → 持久化会话到 JSON + SQLite
        # 部件列表；轨迹格式期望纯字符串。
        self._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed)

        # 对话完成后清理此任务的 VM 和浏览器资源
        self._cleanup_task_resources(effective_task_id)

        # 将会话持久化到 JSON 日志和 SQLite
        self._persist_session(messages, conversation_history)

        # ── Turn-exit diagnostic log ─────────────────────────────────────
        # 始终以 INFO 记录，使 agent.log 捕获每轮结束的原因。
        # 当最后一条消息是工具结果时（Agent 仍在工作中），记录
        # at WARNING — this is the "just stops" scenario users report.
        _last_msg_role = messages[-1].get("role") if messages else None
        _last_tool_name = None
        if _last_msg_role == "tool":
            # 回退查找具有工具调用的 assistant 消息
            for _m in reversed(messages):
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    _tcs = _m["tool_calls"]
                    if _tcs and isinstance(_tcs[0], dict):
                        _last_tool_name = _tcs[-1].get("function", {}).get("name")
                    break

        _turn_tool_count = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
        )
        _resp_len = len(final_response) if final_response else 0
        _budget_used = self.iteration_budget.used if self.iteration_budget else 0
        _budget_max = self.iteration_budget.max_total if self.iteration_budget else 0

        _diag_msg = (
            "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
            "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
        )
        _diag_args = (
            _turn_exit_reason, self.model, api_call_count, self.max_iterations,
            _budget_used, _budget_max,
            _turn_tool_count, _last_msg_role, _resp_len,
            self.session_id or "none",
        )

        if _last_msg_role == "tool" and not interrupted:
            # Agent was mid-work — this is the "just stops" case.
            logger.warning(
                "Turn ended with pending tool result (agent may appear stuck). "
                + _diag_msg + " last_tool=%s",
                *_diag_args, _last_tool_name,
            )
        else:
            logger.info(_diag_msg, *_diag_args)

        # 插件钩子：post_llm_call
        # 每轮在工具调用循环完成后触发一次。
        # 插件可用此持久化对话数据（如同步到
        # 外部记忆系统）。
        if final_response and not interrupted:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "post_llm_call",
                    session_id=self.session_id,
                    user_message=original_user_message,
                    assistant_response=final_response,
                    conversation_history=list(messages),
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("post_llm_call hook failed: %s", exc)

        # 从最后一条 assistant 消息提取推理（如有）
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        # 构建结果，如适用包含中断信息
        result = {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "partial": False,  # True only when stopped due to invalid tool calls
            "interrupted": interrupted,
            "response_previewed": getattr(self, "_response_was_previewed", False),
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "input_tokens": self.session_input_tokens,
            "output_tokens": self.session_output_tokens,
            "cache_read_tokens": self.session_cache_read_tokens,
            "cache_write_tokens": self.session_cache_write_tokens,
            "reasoning_tokens": self.session_reasoning_tokens,
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "last_prompt_tokens": getattr(self.context_compressor, "last_prompt_tokens", 0) or 0,
            "estimated_cost_usd": self.session_estimated_cost_usd,
            "cost_status": self.session_cost_status,
            "cost_source": self.session_cost_source,
        }
        # 若 /steer 在最后一个 assistant 轮次后到达（无更多工具
        # 批次可排入），交还给调用者，使其能被
        # 作为下一个用户轮次传递，而非静默丢失。
        _leftover_steer = self._drain_pending_steer()
        if _leftover_steer:
            result["pending_steer"] = _leftover_steer
        self._response_was_previewed = False
        
        # 若中断消息触发则包含中断消息
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        # 处理后清除中断状态
        self.clear_interrupt()

        # 清除流回调，避免泄漏到未来调用中
        self._stream_callback = None

        # Check skill trigger NOW — based on how many tool iterations THIS turn used.
        _should_review_skills = False
        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            _should_review_skills = True
            self._iters_since_skill = 0

        # 外部记忆提供商：同步已完成的轮次 + 排队下次预取。
        # Use original_user_message (clean input) — user_message may contain
        # 注入的技能内容膨胀/破坏提供商查询。
        if self._memory_manager and final_response and original_user_message:
            try:
                self._memory_manager.sync_all(original_user_message, final_response)
                self._memory_manager.queue_prefetch_all(original_user_message)
            except Exception:
                pass

        # Background memory/skill review — runs AFTER the response is delivered
        # 使其永不与用户的任务竞争模型注意力。
        if final_response and not interrupted and (_should_review_memory or _should_review_skills):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                )
            except Exception:
                pass  # Background review is best-effort

        # 注意：记忆提供商的 on_session_end() + shutdown_all() 不
        # called here — run_conversation() is called once per user message in
        # 多轮会话。每轮后关闭会在第二条消息前
        # 杀死提供商。实际的会话结束清理由
        # CLI（atexit / /reset）和网关（会话过期 /
        # _reset_session）处理。

        # 插件钩子：on_session_end
        # 在每次 run_conversation 调用结束时触发。
        # 插件可用此进行清理、刷新缓冲区等。
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_end",
                session_id=self.session_id,
                completed=completed,
                interrupted=interrupted,
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("on_session_end hook failed: %s", exc)

        return result

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """简洁聊天接口 — 返回纯文本最终响应 (精简版 run_conversation).

        [中文] 对 run_conversation() 的精简包装, 仅返回最终文本响应。
        适用于不需要完整结果字典的简单场景。

        Args:
            message: 用户消息
            stream_callback: 可选的流式 delta 回调 (用于 TTS 等)

        Returns:
            最终的助手文本响应
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20
):
    """CLI 入口 — 独立运行 Hermes Agent 的命令行接口.

    [中文] 这是 `python run_agent.py` 的直接入口:
      1. 解析命令行参数
      2. 创建 AIAgent 实例
      3. 调用 run_conversation() 执行对话
      4. 打印最终结果和摘要

    支持工具集过滤 (--enabled_toolsets/--disabled_toolsets),
    轨迹保存 (--save_trajectories/--save_sample), 和调试日志。

    Args:
        query: 自然语言查询 (默认: Python 3.13 示例)
        model: 模型名称 (OpenRouter 格式: provider/model)
        api_key: API 认证密钥 (未提供时从环境变量读取)
        base_url: API 基础 URL
        max_turns: 最大 API 调用次数 (默认 10)
        enabled_toolsets: 逗号分隔的启用工具集列表
        disabled_toolsets: 逗号分隔的禁用工具集列表
        list_tools: 列出可用工具后退出
        save_trajectories: 保存对话轨迹到 JSONL
        save_sample: 保存单条轨迹样本到 UUID 命名的文件
        verbose: 启用详细调试日志
        log_prefix_chars: 日志预览字符数

    工具集示例:
        - "research": Web 搜索 + 提取 + 爬虫 + 视觉
        - "development": 终端 + 文件 + 代码分析
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    
    # 处理工具列表
    if list_tools:
        from model_tools import get_all_tool_names, get_available_toolsets
        from toolsets import get_all_toolsets, get_toolset_info
        
        print("📋 Available Tools & Toolsets:")
        print("-" * 50)
        
        # 显示新工具集系统
        print("\n🎯 Predefined Toolsets (New System):")
        print("-" * 40)
        all_toolsets = get_all_toolsets()
        
        # 按类别分组
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []
        
        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in ["web", "terminal", "vision", "creative", "reasoning"]:
                    basic_toolsets.append(entry)
                elif name in ["research", "development", "analysis", "content_creation", "full_stack"]:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)
        
        # 打印基本工具集
        print("\n📌 Basic Toolsets:")
        for name, info in basic_toolsets:
            tools_str = ', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Tools: {tools_str}")
        
        # 打印复合工具集
        print("\n📂 Composite Toolsets (built from other toolsets):")
        for name, info in composite_toolsets:
            includes_str = ', '.join(info['includes']) if info['includes'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Includes: {includes_str}")
            print(f"    Total tools: {info['tool_count']}")
        
        # 打印场景特定工具集
        print("\n🎭 Scenario-Specific Toolsets:")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    Total tools: {info['tool_count']}")
        
        
        # 显示旧版工具集兼容性
        print("\n📦 Legacy Toolsets (for backward compatibility):")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")
        
        # 显示单个工具
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")
        
        print("\n💡 Usage Examples:")
        print("  # Use predefined toolsets")
        print("  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print("  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print("  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print("  ")
        print("  # Combine multiple toolsets")
        print("  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print("  ")
        print("  # Disable toolsets")
        print("  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print("  ")
        print("  # Run with trajectory saving enabled")
        print("  python run_agent.py --save_trajectories --query='your question here'")
        return
    
    # 解析工具集选择参数
    enabled_toolsets_list = None
    disabled_toolsets_list = None
    
    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")
    
    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")
    
    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")
    
    # 使用提供的参数初始化 Agent
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            log_prefix_chars=log_prefix_chars
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # 使用提供的查询或默认 Python 3.13 示例
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query
    
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)
    
    # 运行对话
    result = agent.run_conversation(user_query)
    
    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")
    
    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result['final_response'])
    
    # 如请求，将示例轨迹保存到 UUID 命名的文件
    if save_sample:
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.json"
        
        # 将消息转换为轨迹格式（同 batch_runner）
        trajectory = agent._convert_to_trajectory_format(
            result['messages'], 
            user_query, 
            result['completed']
        )
        
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result['completed'],
            "query": user_query
        }
        
        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                # 为可读性以缩进美化 JSON
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")
    
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    fire.Fire(main)
