"""可插拔记忆 Provider 的抽象基类。

================================================================================
设计目的
================================================================================
MemoryProvider 定义了记忆系统的统一接口。MemoryProvider 的子类可以是：

  - BuiltinMemoryProvider：管理 MEMORY.md / USER.md（始终启用，全量加载）
  - Honcho / Mem0 / Hindsight / Supermemory：云端或向量存储（按需检索）

每个 provider 负责实现自己的持久化策略，run_agent.py 只通过抽象接口调用。

================================================================================
加载策略对比
================================================================================
内置（BuiltinMemoryProvider — tools/memory_tool.py）：

  加载时机：会话启动时一次（load_from_disk）
  加载方式：全量读取文件
  过滤逻辑：仅精确去重（dict.fromkeys）
  注入位置：system prompt（冻结快照，保护 prefix cache）
  字符上限：MEMORY.md 2200 / USER.md 1375

外部（External Provider）：

  加载时机：每轮对话前
  加载方式：基于当前 query 动态检索
  过滤逻辑：语义相似度 / 向量检索 / 时序推理
  注入位置：<memory-context> 标签包裹，注入到 user message 前
  容量：可达 MB 级

================================================================================
矛盾解决与淘汰机制
================================================================================
内置记忆的矛盾处理完全由 LLM 自主决策：
  - 模型检测到矛盾时调用 memory(action="replace", old_text=..., content=...)
  - 模型发现事实不再成立时调用 memory(action="remove", old_text=...)
  - 字符上限满时 add 失败，错误消息提示需先 replace/remove
  - 仅 hermes memory reset 提供全局擦除

外部 provider 可能提供更高级能力：
  - Honcho：时序推理 + 用户/agent 表示，自动检测矛盾陈述
  - Mem0：LLM-based 事实提取，语义去重
  - Hindsight：时序记忆，事件排序

================================================================================
限制
================================================================================
内置层（BuiltinMemoryProvider）刻意没有以下能力：
  - 自动遗忘 / TTL（条目永不过期）
  - 基于相似度的去重（仅检测完全相同）
  - 自动矛盾检测（不做语义冲突检测）
  - 使用频率衰减（不区分常用/罕见）

理由：记忆管理是判断密集型任务，自动算法难以做好。
设计哲学：LLM 自主决策 + 字符上限硬约束 = 比复杂启发式更可靠。

================================================================================
注册方式
================================================================================
  1. 内置：BuiltinMemoryProvider — 始终存在，不可移除
  2. 插件：plugins/memory/<name>/ — 通过 memory.provider 配置激活

================================================================================
生命周期（由 MemoryManager 调用，run_agent.py 串联）
================================================================================
  initialize(session_id, **kwargs)   — 连接、创建资源、预热
  system_prompt_block()              — system prompt 中的静态文本
  prefetch(query)                    — 每轮前的后台检索
  sync_turn(user, asst)              — 每轮后的异步写入
  get_tool_schemas()                 — 暴露给模型的工具 schema
  handle_tool_call(name, args)       — 工具调用分发
  shutdown()                         — 清理退出

================================================================================
可选钩子（覆盖以启用）
================================================================================
  on_turn_start(turn, message, **kwargs)    — 每轮节奏通知
  on_session_end(messages)                  — 会话结束提取
  on_pre_compress(messages) -> str          — 压缩前提取（返回内容注入压缩摘要）
  on_memory_write(action, target, content)  — 桥接内置 memory 写入
  on_delegation(task, result, **kwargs)     — 父端观察子 agent 工作
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """记忆 provider 的抽象基类。

    加载策略对比：
      - BuiltinMemoryProvider：启动时全量加载所有条目到 system prompt（冻结快照）
      - 外部 provider：每轮对话前基于 query 动态检索（按需过滤）

    矛盾解决：
      - 内置层没有自动矛盾检测，由 LLM 自主调用 replace/remove
      - 外部 provider 可能提供更高级能力（Honcho 自动检测、Mem0 语义去重）

    注册限制：
      - BuiltinMemoryProvider（name == "builtin"）始终启用，不可移除
      - 外部 provider 最多同时激活一个（防止 tool schema 膨胀和后端冲突）
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """provider 短名称（如 'builtin', 'honcho', 'hindsight'）。"""

    # -- 核心生命周期（必须实现）-------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """返回 True 如果此 provider 已配置、有凭据、就绪。

        在 agent 初始化期间调用以决定是否激活 provider。
        不应发起网络调用 — 仅检查配置和已安装的依赖。
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """为会话初始化 provider。

        在 agent 启动时调用一次。可以创建资源（banks、tables）、
        建立连接、启动后台线程等。

        kwargs 始终包含：
          - hermes_home (str): 活跃的 HERMES_HOME 目录路径。
            使用此路径实现 profile 作用域存储，而不是硬编码 ``~/.hermes``。
          - platform (str): "cli", "telegram", "discord", "cron" 等。

        kwargs 还可能包含：
          - agent_context (str): "primary", "subagent", "cron", 或 "flush"。
            Provider 应跳过非 primary 上下文的写入
            （cron system prompts 会污染用户表示）。
          - agent_identity (str): Profile 名（如 "coder"）。用于按 profile 区分。
          - agent_workspace (str): 共享 workspace 名（如 "hermes"）。
          - parent_session_id (str): 子 agent 时，父 agent 的 session_id。
          - user_id (str): 平台用户标识符（gateway sessions）。
        """

    def system_prompt_block(self) -> str:
        """返回要包含在 system prompt 中的文本。

        在 system prompt 组装期间调用。返回空字符串跳过。
        这是 STATIC provider 信息（指令、状态）。预取的回忆上下文
        通过 prefetch() 单独注入。
        """
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """为即将到来的对话轮次回忆相关上下文。

        在每次 API 调用前调用。返回格式化的文本以注入为上下文，
        或返回空字符串表示无相关内容。实现应该快速 —
        使用后台线程执行实际回忆，并返回此处缓存的结果。

        这是**外部 provider 的加载策略**：
          - 与内置的"全量加载"不同，外部按 query 动态检索
          - 内置在 system prompt 中始终可见（冻结快照）
          - 外部按相关性筛选后注入到 user message 前（<memory-context>）

        session_id 为服务于并发会话的 provider（gateway 群聊、cached agents）提供。
        不需要 per-session 作用域的 provider 可以忽略它。
        """
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """为下一轮排队后台回忆。

        在每轮完成后调用。结果将在下一轮被 prefetch() 消费。
        默认无操作 — 执行后台预取的 provider 应覆盖此方法。
        """

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """将已完成的轮次持久化到后端。

        在每轮后调用。应该非阻塞 — 如果后端有延迟则排队后台处理。

        这是**写入路径**：
          - 与内置的"写入磁盘"不同，外部 provider 可能用云端 API
          - 异步执行避免阻塞对话循环
          - 失败不应中断主流程
        """

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this provider exposes.

        Each schema follows the OpenAI function calling format:
        {"name": "...", "description": "...", "parameters": {...}}

        Return empty list if this provider has no tools (context-only).
        """

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call for one of this provider's tools.

        Must return a JSON string (the tool result).
        Only called for tool names returned by get_tool_schemas().
        """
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # -- 可选钩子（覆盖以启用）---------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """在每轮开始时调用，传入用户消息。

        用于轮次计数、作用域管理、定期维护。

        kwargs 可能包含：remaining_tokens, model, platform, tool_count。
        Provider 按需使用，额外参数被忽略。
        """

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """会话结束时调用（显式退出或超时）。

        用于会话结束的事实提取、摘要等。
        messages 是完整的对话历史。

        不是每轮调用 — 仅在实际会话边界调用
        （CLI 退出、/reset、gateway 会话过期）。
        """

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """在上下文压缩丢弃旧消息前调用。

        用于从即将压缩的消息中提取见解。
        messages 是将被摘要/丢弃的列表。

        返回要包含在压缩摘要 prompt 中的文本，以便压缩器保留 provider 提取的见解。
        返回空字符串表示无贡献（向后兼容默认）。

        为什么重要？
          即将被压缩的消息中可能包含重要上下文。
          Provider 可在此钩子中提取关键事实，确保它们保留在压缩摘要中。
        """
        return ""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """在子 agent 完成后，**父 agent** 被调用。

        父 agent 的 memory provider 收到 task+result 对作为对委派内容的观察。
        子 agent 本身没有 provider 会话（skip_memory=True）。

        task: 委派 prompt
        result: 子 agent 的最终响应
        child_session_id: 子 agent 的 session_id
        """

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """返回此 provider 设置所需的配置字段。

        用于 'hermes memory setup' 引导用户完成配置。
        每个字段是一个 dict：
          key:         配置键名（如 'api_key', 'mode'）
          description: 人类可读的描述
          secret:      True 如果应放到 .env（默认 False）
          required:    True 如果必需（默认 False）
          default:     默认值（可选）
          choices:     有效值列表（可选）
          url:         获取凭据的 URL（可选）
          env_var:     secrets 的显式环境变量名（默认自动生成）

        如果不需要配置（如纯本地 provider），返回空列表。
        """
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """将非 secret 配置写入 provider 的原生位置。

        在 'hermes memory setup' 收集用户输入后调用。
        ``values`` 仅包含非 secret 字段（secrets 进入 .env）。
        ``hermes_home`` 是活跃的 HERMES_HOME 目录路径。

        有原生配置文件（JSON、YAML）的 provider 应覆盖此方法以写入预期位置。
        仅使用环境变量的 provider 可保留默认（无操作）。

        所有新的 memory provider 插件必须实现以下之一：
          - save_config() 用于原生配置文件格式，或
          - 仅使用环境变量（get_config_schema() 字段都应有 ``env_var`` 设置）
        """

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """当内置 memory 工具写入条目时调用 — **桥接机制**。

        触发流程：
          1. LLM 调用内置 memory 工具（add/replace/remove）
          2. MemoryStore 修改文件
          3. MemoryManager.on_memory_write 通知所有外部 provider
          4. 外部 provider 镜像写入到自己的后端

        这实现了**单一写入路径**：模型只需调用一个工具，
        内置存储和外部后端都会被更新。

        action: 'add', 'replace', 或 'remove'
        target: 'memory' 或 'user'
        content: 条目内容
        """
