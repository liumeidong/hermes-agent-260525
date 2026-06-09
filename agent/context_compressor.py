"""对话上下文自动压缩 — 长会话的"记忆压缩"机制。

================================================================================
设计动机
================================================================================
LLM 的上下文窗口有限（如 128K/200K tokens）。长对话会积累大量消息——
工具调用参数、工具返回结果、多轮思考——很快超出窗口限制，导致 API 调用失败。

ContextCompressor 的解决思路：
  - 在超限前自动压缩"中间"的对话轮次
  - 用 LLM 将中间轮次总结为结构化摘要
  - 保护头部（system prompt + 最早几轮）和尾部（最近 N 轮）
  - 压缩后消息数大幅减少，token 消耗回到阈值以下

================================================================================
压缩算法（5 阶段）
================================================================================

Phase 1: 工具结果裁剪（零 LLM 调用，纯本地计算）
  - 旧工具返回值替换为 1 行摘要，如：
    [terminal] ran `npm test` -> exit 0, 47 lines output
    [read_file] read config.py from line 1 (3,400 chars)
  - 相同文件被多次读取时，只保留最新一次完整内容，其余标为重复

Phase 2: 边界定位
  - 头部保护: 前 N 条消息不动（默认 3 条）
  - 尾部保护: 按 token 预算从后往前累积消息（默认 threshold * 0.20）
  - 边界对齐: 不切断 tool_call/result 配对

Phase 3: LLM 摘要生成
  - 使用辅助模型（便宜/快速）生成结构化摘要
  - 模板包含: Active Task, Goal, Completed Actions, Active State,
    In Progress, Blocked, Key Decisions, Resolved Questions,
    Pending User Asks, Relevant Files, Remaining Work, Critical Context
  - 首次压缩: 从零总结
  - 后续压缩: 在已有摘要基础上迭代更新（增量合并）

Phase 4: 消息组装
  - 头消息 → 摘要消息 → 尾消息
  - 修复孤立的 tool_call/result 配对
  - 在 system prompt 中注入压缩提示

Phase 5: 会话拆分（run_agent.py 负责）
  - SQLite 中 end_session(old_id, "compression")
  - 创建新 session，通过 parent_session_id 链式连接
  - 标题自动编号: "my session" → "my session #2"

================================================================================
关键设计决策
================================================================================

Token-budget 尾部保护:
  使用 token_budget（默认 context_length * 0.20）而非固定消息数来
  决定尾部保留多少条消息。大型上下文窗口的模型会保留更多上下文。

迭代摘要更新:
  首次压缩从零总结；之后每次压缩将新的中间轮次合并到已有摘要中。
  避免了重复总结同一段历史——节省 token 且保持信息连续性。

结构化摘要模板:
  不是自由文本，而是有固定段落的结构化输出。确保关键信息
  (Active Task, Pending User Asks, Relevant Files) 不丢失。

反抖动保护 (Anti-thrashing):
  如果连续 2 次压缩各节省 <10%，说明压缩已无效——跳过后续压缩，
  提示用户执行 /new 或 /compress <topic>。

模型回退:
  如果配置的辅助摘要模型不可用 (404/503/"model_not_found")，
  自动回退到主模型。避免压缩功能整体不可用。

================================================================================
Compression algorithm (5 phases in English):

Phase 1: Tool output pruning (zero LLM calls, purely local)
  - Replace old tool results with 1-line summaries
  - Deduplicate identical tool results, keeping only the newest

Phase 2: Boundary determination
  - Head protection: first N messages untouched (default 3)
  - Tail protection: walk backward accumulating tokens until budget exhausted
  - Boundary alignment: don't split tool_call/result pairs

Phase 3: LLM summarization
  - Use auxiliary model (cheap/fast) for structured summary
  - Structured template: Active Task, Goal, Completed Actions, etc.
  - First compression: summarize from scratch
  - Subsequent: iteratively update previous summary

Phase 4: Message assembly
  - head messages → summary message → tail messages
  - Sanitize orphaned tool_call/result pairs
  - Inject compression note into system prompt

Phase 5: Session splitting (handled by run_agent.py)
  - SQLite: end old session, create new with parent_session_id chain
  - Auto-number titles: "my session" → "my session #2"
"""

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm
from agent.context_engine import ContextEngine
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    estimate_messages_tokens_rough,
)
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

# ============================================================================
# 摘要前缀 — 注入到压缩后的消息列表中，告诉模型"这是摘要，不是指令"
# ============================================================================
# 关键设计：
#   - 显式声明"handoff from a previous context window"，制造"不是同一个助手"的
#     心理框架（借鉴 Codex 的 "different assistant" 模式）
#   - "Do NOT answer questions" — 防止模型把摘要中已解决的问题重新回答一遍
#     （借鉴 OpenCode 的 "Do not respond to any questions" 模式）
#   - "Respond ONLY to the latest user message" — 将注意力引导到最新的
#     真实用户消息，而非摘要内容
#   - 摘要前导内容 + SUMMARY_PREFIX 共同构成了压缩后的"上下文交接"协议
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)
# 旧版摘要前缀 — 用于向后兼容地识别旧格式摘要
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# ---------------------------------------------------------------------------
# 摘要 token 预算控制
# ---------------------------------------------------------------------------
# 摘要的最少输出 token 数 — 太小了信息量不足，无法支撑后续工作
_MIN_SUMMARY_TOKENS = 2000
# 摘要占被压缩内容的比例 — 被压缩的内容越多，摘要预算越高
_SUMMARY_RATIO = 0.20
# 摘要 token 的绝对上限 — 即使上下文窗口很大（如 200K），摘要也不能太长
# 摘要本身也占用上下文窗口中的空间
_SUMMARY_TOKENS_CEILING = 12_000

# 工具结果裁剪时的占位符（旧版，现在用 _summarize_tool_result 替代）
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# ---------------------------------------------------------------------------
# 其他常量
# ---------------------------------------------------------------------------
# 粗略的每 token 字符数估计 — 用于快速 token 计数（非精确）
# Python/英文约 4 chars/token，中文约 2 chars/token，取 4 作为保守估计
_CHARS_PER_TOKEN = 4
# 摘要失败后的冷却时间（秒）— 避免对不可达的 API 端点连续重试
# 600 秒 = 10 分钟，足够让瞬态故障（网络波动、限流）恢复
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600


def _content_text_for_contains(content: Any) -> str:
    """Return a best-effort text view of message content.

    Used only for substring checks when we need to know whether we've already
    appended a note to a message. Keeps multimodal lists intact elsewhere.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _append_text_to_content(content: Any, text: str, *, prepend: bool = False) -> Any:
    """Append or prepend plain text to message content safely.

    Compression sometimes needs to add a note or merge a summary into an
    existing message. Message content may be plain text or a multimodal list of
    blocks, so direct string concatenation is not always safe.
    """
    if content is None:
        return text
    if isinstance(content, str):
        return text + content if prepend else content + text
    if isinstance(content, list):
        text_block = {"type": "text", "text": text}
        return [text_block, *content] if prepend else [*content, text_block]
    rendered = str(content)
    return text + rendered if prepend else rendered + text


def _truncate_tool_call_args_json(args: str, head_chars: int = 200) -> str:
    """Shrink long string values inside a tool-call arguments JSON blob while
    preserving JSON validity.

    The ``function.arguments`` field on a tool call is a JSON-encoded string
    passed through to the LLM provider; downstream providers strictly
    validate it and return a non-retryable 400 when it is not well-formed.
    An earlier implementation sliced the raw JSON at a fixed byte offset and
    appended ``...[truncated]`` — which routinely produced strings like::

        {"path": "/foo/bar", "content": "# long markdown
        ...[truncated]

    i.e. an unterminated string and a missing closing brace. MiniMax, for
    example, rejects this with ``invalid function arguments json string``
    and the session gets stuck re-sending the same broken history on every
    turn. See issue #11762 for the observed loop.

    This helper parses the arguments, shrinks long string leaves inside the
    parsed structure, and re-serialises. Non-string values (paths, ints,
    booleans) are preserved intact. If the arguments are not valid JSON
    to begin with — some model backends use non-JSON tool arguments — the
    original string is returned unchanged rather than replaced with
    something neither we nor the backend can parse.
    """
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return args

    def _shrink(obj: Any) -> Any:
        if isinstance(obj, str):
            if len(obj) > head_chars:
                return obj[:head_chars] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    shrunken = _shrink(parsed)
    # ensure_ascii=False preserves CJK/emoji instead of bloating with \uXXXX
    return json.dumps(shrunken, ensure_ascii=False)


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Create an informative 1-line summary of a tool call + result.

    Used during the pre-compression pruning pass to replace large tool
    outputs with a short but useful description of what the tool did,
    rather than a generic placeholder that carries zero information.

    Returns strings like::

        [terminal] ran `npm test` -> exit 0, 47 lines output
        [read_file] read config.py from line 1 (1,200 chars)
        [search_files] content search for 'compress' in agent/ -> 12 matches
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} search for '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars result)"

    if tool_name in ("browser_navigate", "browser_click", "browser_snapshot",
                     "browser_type", "browser_scroll", "browser_vision"):
        url = args.get("url", "")
        ref = args.get("ref", "")
        detail = f" {url}" if url else (f" ref={ref}" if ref else "")
        return f"[{tool_name}]{detail} ({content_len:,} chars)"

    if tool_name == "web_search":
        query = args.get("query", "?")
        return f"[web_search] query='{query}' ({content_len:,} chars result)"

    if tool_name == "web_extract":
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({content_len:,} chars)"

    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars result)"

    if tool_name == "execute_code":
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        return f"[execute_code] `{code_preview}` ({line_count} lines output)"

    if tool_name in ("skill_view", "skills_list", "skill_manage"):
        name = args.get("name", "?")
        return f"[{tool_name}] name={name} ({content_len:,} chars)"

    if tool_name == "vision_analyze":
        question = args.get("question", "")[:50]
        return f"[vision_analyze] '{question}' ({content_len:,} chars)"

    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "?")
        return f"[memory] {action} on {target}"

    if tool_name == "todo":
        return "[todo] updated task list"

    if tool_name == "clarify":
        return "[clarify] asked user a question"

    if tool_name == "text_to_speech":
        return f"[text_to_speech] generated audio ({content_len:,} chars)"

    if tool_name == "cronjob":
        action = args.get("action", "?")
        return f"[cronjob] {action}"

    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "?")
        return f"[process] {action} session={sid}"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


class ContextCompressor(ContextEngine):
    """默认上下文引擎 — 通过有损摘要压缩对话上下文。

    核心算法（5 个阶段）:
      1. 工具结果裁剪 — 旧的工具返回值替换为 1 行摘要，不调用 LLM
      2. 头部保护 — system prompt + 前几轮对话不变
      3. 尾部保护 — 按 token 预算保留最近的消息（约 20% 上下文窗口）
      4. LLM 摘要生成 — 用结构化 prompt 让辅助模型总结中间轮次
      5. 迭代更新 — 后续压缩在已有摘要基础上增量合并，而非从头总结

    反抖动保护：
      连续 2 次压缩各节省 <10% 时，跳过后续压缩，防止无效压缩死循环。

    会话生命周期：
      - /new 或 /reset 时调用 on_session_reset() 清除所有会话状态
      - 模型切换时调用 update_model() 重新计算上下文窗口相关参数
      - 每次 API 响应后 update_from_response() 更新 token 计数
      - should_compress() 判断是否需要触发压缩
      - compress() 执行实际压缩

    Attributes:
        threshold_percent: 触发压缩的阈值比例 (默认 50%，即上下文窗口用了一半时触发)
        protect_first_n: 头部保护的消息数 (默认 3)
        protect_last_n: 尾部保护的硬最小消息数 (默认 20)
        summary_target_ratio: 尾部 token 预算占阈值的比例 (默认 0.20)
        _previous_summary: 上次压缩的摘要文本，用于迭代更新
        _ineffective_compression_count: 连续无效压缩计数（反抖动）
    """

    @property
    def name(self) -> str:
        """引擎名称标识 — 用于 config.yaml 中 context.engine 选择。"""
        return "compressor"

    def on_session_reset(self) -> None:
        """重置所有会话级状态 — /new 或 /reset 时调用。

        清除内容：
          - _context_probed: 上下文探测标志（API 返回 context_length_exceeded 后设为 True）
          - _previous_summary: 上一次的压缩摘要（/reset 后应该重新开始）
          - 压缩效果追踪: 节省比例和无效计数归零
        """
        super().on_session_reset()
        self._context_probed = False
        self._context_probe_persistable = False
        self._previous_summary = None
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """模型切换或回退激活后更新模型信息。

        重新计算：
          - context_length: 模型上下文窗口大小
          - threshold_tokens: 压缩触发阈值 = max(context_length * threshold_percent, MINIMUM_CONTEXT_LENGTH)
            例如：200K * 0.50 = 100K 时触发压缩
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        self.threshold_tokens = max(
            int(context_length * self.threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )

    def __init__(
        self,
        model: str,                         # 主 LLM 模型名（用于获取上下文窗口大小）
        threshold_percent: float = 0.50,     # 触发阈值: 占用上下文窗口的比例（0.50 = 50%）
        protect_first_n: int = 3,           # 头部保护的消息数（含 system prompt）
        protect_last_n: int = 20,           # 尾部硬最小保护消息数（token 预算优先）
        summary_target_ratio: float = 0.20, # 尾部 token 预算占阈值的比例
        quiet_mode: bool = False,           # 静默模式，减少日志输出
        summary_model_override: str = None, # 强制指定摘要模型（None = 使用主模型）
        base_url: str = "",                 # API base URL
        api_key: str = "",                  # API key
        config_context_length: int | None = None,  # 用户配置覆盖的上下文窗口大小
        provider: str = "",                 # LLM 提供商
        api_mode: str = "",                 # API 模式
    ):
        # ---- 基本身份 ----
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode

        # ---- 压缩参数 ----
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        # summary_target_ratio 限制在 [0.10, 0.80] — 至少保留 10% 作为尾部，最多 80%
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))
        self.quiet_mode = quiet_mode

        # ---- 上下文窗口大小 ----
        # 从模型元数据获取上下文窗口长度，支持用户配置覆盖
        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        # 压缩触发阈值（token 数）
        # 例如 200K 上下文 × 50% = 100K 时触发压缩
        # 下限保护: 即使模型上下文很小，阈值不低于 MINIMUM_CONTEXT_LENGTH
        # 这防止了在大上下文窗口模型上过早触发压缩（50% 依然合理），
        # 同时对刚好达到最小值的模型保持百分比合理
        self.threshold_tokens = max(
            int(self.context_length * threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )
        self.compression_count = 0

        # ---- 推导 token 预算 ----
        # 尾部 token 预算: 基于阈值（不是总上下文）来计算
        # 例如 threshold=100K, ratio=0.20 → tail_token_budget=20K tokens
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        # 摘要输出上限: 上下文窗口的 5% 或 _SUMMARY_TOKENS_CEILING (12K) 中较小值
        # 原因：摘要本身也占用上下文空间，不能太大
        self.max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        if not quiet_mode:
            logger.info(
                "Context compressor initialized: model=%s context_length=%d "
                "threshold=%d (%.0f%%) target_ratio=%.0f%% tail_budget=%d "
                "provider=%s base_url=%s",
                model, self.context_length, self.threshold_tokens,
                threshold_percent * 100, self.summary_target_ratio * 100,
                self.tail_token_budget,
                provider or "none", base_url or "none",
            )
        # ---- 运行时状态 ----
        # 上下文探测标志: API 返回 context_length_exceeded 错误后设为 True，
        # 表示实际上下文窗口可能比元数据声称的小
        self._context_probed = False

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        # 摘要模型：如果未指定，使用主模型
        self.summary_model = summary_model_override or ""

        # ---- 迭代摘要状态 ----
        # 存储上一次压缩的完整摘要文本，用于后续压缩的增量更新
        self._previous_summary: Optional[str] = None
        # ---- 反抖动追踪 ----
        # 上次压缩的 token 节省百分比；连续 <10% 的压缩累计无效计数
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        # 摘要失败冷却: 避免对不可达的 API 端点连续重试
        self._summary_failure_cooldown_until: float = 0.0

    def update_from_response(self, usage: Dict[str, Any]):
        """Update tracked token usage from API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """检查是否需要触发上下文压缩。

        判断逻辑：
          1. 当前 prompt_tokens 是否超过 threshold_tokens（默认 50% 上下文窗口）
          2. 反抖动保护：如果连续 2 次压缩各节省 <10%，跳过

        为什么在 50% 时触发而不是 90%？
          - 压缩本身需要额外 LLM 调用，需要时间
          - 如果等到 90% 才压缩，下一次 API 调用可能超出 100% 导致失败
          - 50% 给了充足的缓冲空间

        为什么 10% 节省算无效？
          - 如果每次压缩只能移除 1-2 条消息，说明对话结构过于紧密
            （大量短轮次相互依赖），压缩无法有效缩减
          - 此时继续压缩只会浪费 LLM 调用费用，建议用户手动 /new 或 /compress <topic>
        """
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        # 反抖动保护: 连续 2 次压缩无效时停止
        if self._ineffective_compression_count >= 2:
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped — last %d compressions saved <10%% each. "
                    "Consider /new to start a fresh session, or /compress <topic> "
                    "for focused compression.",
                    self._ineffective_compression_count,
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Phase 1: 工具结果裁剪 — 零 LLM 调用，纯本地去重 + 摘要化
    # ------------------------------------------------------------------
    # 这是压缩前最便宜的预处理步骤，不做任何 LLM 调用：
    #   Pass 1: 去重 — 相同文件被多次读取时，只保留最新完整内容
    #   Pass 2: 摘要化 — 旧工具结果替换为 1 行描述（如 [terminal] ran `npm test` -> exit 0）
    #   Pass 3: 参数截断 — 旧 assistant 消息中过大的 tool_call arguments 截断

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """将旧的工具返回值替换为 1 行信息摘要。

        三种操作：
          Pass 1 — 去重: 相同内容的工具结果只保留最新一份，旧版本标记为 [Duplicate...]
          Pass 2 — 摘要化: 尾部边界之前的旧工具结果（>200 chars）替换为 1 行描述
          Pass 3 — 参数截断: 旧 assistant 消息中 >500 chars 的 tool_call arguments 截断为 200 chars

        尾部保护:
          - token_budget 优先: 从后往前累积 token，超过 budget 时停止
          - protect_tail_count 作为硬最小下限
          - 用 MD5 去重而非原文比较（避免在上下文中存两次完整输出）

        返回值:
          (pruned_messages, pruned_count): 处理后的消息列表和被修改的消息数
        """
        if not messages:
            return messages, 0

        result = [m.copy() for m in messages]
        pruned = 0

        # Build index: tool_call_id -> (tool_name, arguments_json)
        call_id_to_tool: Dict[str, tuple] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                    else:
                        cid = getattr(tc, "id", "") or ""
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "unknown") if fn else "unknown"
                        args_str = getattr(fn, "arguments", "") if fn else ""
                        call_id_to_tool[cid] = (name, args_str)

        # Determine the prune boundary
        if protect_tail_tokens is not None and protect_tail_tokens > 0:
            # Token-budget approach: walk backward accumulating tokens
            accumulated = 0
            boundary = len(result)
            min_protect = min(protect_tail_count, len(result) - 1)
            for i in range(len(result) - 1, -1, -1):
                msg = result[i]
                raw_content = msg.get("content") or ""
                content_len = sum(len(p.get("text", "")) for p in raw_content) if isinstance(raw_content, list) else len(raw_content)
                msg_tokens = content_len // _CHARS_PER_TOKEN + 10
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        args = tc.get("function", {}).get("arguments", "")
                        msg_tokens += len(args) // _CHARS_PER_TOKEN
                if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                    boundary = i
                    break
                accumulated += msg_tokens
                boundary = i
            prune_boundary = max(boundary, len(result) - min_protect)
        else:
            prune_boundary = len(result) - protect_tail_count

        # Pass 1: Deduplicate identical tool results.
        # When the same file is read multiple times, keep only the most recent
        # full copy and replace older duplicates with a back-reference.
        content_hashes: dict = {}  # hash -> (index, tool_call_id)
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            # Skip multimodal content (list of content blocks)
            if isinstance(content, list):
                continue
            if len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in content_hashes:
                # This is an older duplicate — replace with back-reference
                result[i] = {**msg, "content": "[Duplicate tool output — same content as a more recent call]"}
                pruned += 1
            else:
                content_hashes[h] = (i, msg.get("tool_call_id", "?"))

        # Pass 2: Replace old tool results with informative summaries
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Skip multimodal content (list of content blocks)
            if isinstance(content, list):
                continue
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            # Skip already-deduplicated or previously-summarized results
            if content.startswith("[Duplicate tool output"):
                continue
            # Only prune if the content is substantial (>200 chars)
            if len(content) > 200:
                call_id = msg.get("tool_call_id", "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                summary = _summarize_tool_result(tool_name, tool_args, content)
                result[i] = {**msg, "content": summary}
                pruned += 1

        # Pass 3: Truncate large tool_call arguments in assistant messages
        # outside the protected tail. write_file with 50KB content, for
        # example, survives pruning entirely without this.
        #
        # The shrinking is done inside the parsed JSON structure so the
        # result remains valid JSON — otherwise downstream providers 400
        # on every subsequent turn until the broken call falls out of
        # the window. See ``_truncate_tool_call_args_json`` docstring.
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            modified = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 500:
                        new_args = _truncate_tool_call_args_json(args)
                        if new_args != args:
                            tc = {**tc, "function": {**tc["function"], "arguments": new_args}}
                            modified = True
                new_tcs.append(tc)
            if modified:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result, pruned

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _compute_summary_budget(self, turns_to_summarize: List[Dict[str, Any]]) -> int:
        """Scale summary token budget with the amount of content being compressed.

        The maximum scales with the model's context window (5% of context,
        capped at ``_SUMMARY_TOKENS_CEILING``) so large-context models get
        richer summaries instead of being hard-capped at 8K tokens.
        """
        content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    # Truncation limits for the summarizer input.  These bound how much of
    # each message the summary model sees — the budget is the *summary*
    # model's context window, not the main model's.
    _CONTENT_MAX = 6000       # total chars per message body
    _CONTENT_HEAD = 4000      # chars kept from the start
    _CONTENT_TAIL = 1500      # chars kept from the end
    _TOOL_ARGS_MAX = 1500     # tool call argument chars
    _TOOL_ARGS_HEAD = 1200    # kept from the start of tool args

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """Serialize conversation turns into labeled text for the summarizer.

        Includes tool call arguments and result content (up to
        ``_CONTENT_MAX`` chars per message) so the summarizer can preserve
        specific details like file paths, commands, and outputs.

        All content is redacted before serialization to prevent secrets
        (API keys, tokens, passwords) from leaking into the summary that
        gets sent to the auxiliary model and persisted across compactions.
        """
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = redact_sensitive_text(msg.get("content") or "")

            # Tool results: keep enough content for the summarizer
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            # Assistant messages: include tool call names AND arguments
            if role == "assistant":
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = redact_sensitive_text(fn.get("arguments", ""))
                            # Truncate long arguments but keep enough for context
                            if len(args) > self._TOOL_ARGS_MAX:
                                args = args[:self._TOOL_ARGS_HEAD] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            # User and other roles
            if len(content) > self._CONTENT_MAX:
                content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    def _generate_summary(self, turns_to_summarize: List[Dict[str, Any]], focus_topic: str = None) -> Optional[str]:
        """[Phase 3] 用辅助 LLM 生成中间对话轮次的结构化摘要。

        两条路径：
          首次压缩 — 从零总结，使用完整结构化模板
          迭代更新 — 已有 _previous_summary，只合并新增轮次
                     保留所有仍相关的旧信息，添加新完成的动作

        结构化模板（12 个段落）：
          Active Task:       用户最新未完成的任务（最重要！下一个助手从这里继续）
          Goal:              总体目标
          Constraints:       用户偏好、编码风格约束
          Completed Actions: 已完成的每个操作（含工具名、目标、结果）
          Active State:      当前工作目录、分支、已修改文件、测试状态
          In Progress:       正在进行的任务
          Blocked:           阻塞原因和详细错误信息
          Key Decisions:     技术决策及 WHY
          Resolved Questions:已回答的问题（附答案，防止重复回答）
          Pending User Asks: 未回答的用户问题/请求
          Relevant Files:    涉及的文件及操作类型
          Remaining Work:    剩余工作（作为上下文，非指令）
          Critical Context:  必须保留的具体值（错误、配置等；不保留 API keys）

        Focus topic (引导压缩, /compress <topic>):
          用户指定关注话题时，摘要模型会：
            - 对相关话题保留完整细节（具体值、路径、错误信息、决策）
            - 对不相关话题激进压缩（一行概括或直接省略）
            - 关注话题部分占用 60-70% 摘要 token 预算
          借鉴 Claude Code 的 /compact <focus> 模式

        安全措施：
          - 所有内容通过 redact_sensitive_text 脱敏后再发送给摘要模型
          - 摘要输出再次脱敏（防止 LLM 忽略 prompt 指令回显密钥）
          - preamble 中明确要求 "NEVER include API keys, tokens, passwords..."

        失败处理（多层回退）：
          1. 冷却期内跳过 (600s)
          2. 辅助模型不可用 → 回退到主模型做摘要
          3. 所有尝试失败 → 返回 None，调用方丢弃中间轮次不注入摘要
                            （比注入无用占位符好）

        Returns:
            带 SUMMARY_PREFIX 的完整摘要文本，或 None
        """
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            logger.debug(
                "Skipping context summary during cooldown (%.0fs remaining)",
                self._summary_failure_cooldown_until - now,
            )
            return None

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)

        # Preamble shared by both first-compaction and iterative-update prompts.
        # Inspired by OpenCode's "do not respond to any questions" instruction
        # and Codex's "another language model" framing.
        _summarizer_preamble = (
            "You are a summarization agent creating a context checkpoint. "
            "Your output will be injected as reference material for a DIFFERENT "
            "assistant that continues the conversation. "
            "Do NOT respond to any questions or requests in the conversation — "
            "only output the structured summary. "
            "Do NOT include any preamble, greeting, or prefix. "
            "Write the summary in the same language the user was using in the "
            "conversation — do not translate or switch to English. "
            "NEVER include API keys, tokens, passwords, secrets, credentials, "
            "or connection strings in the summary — replace any that appear "
            "with [REDACTED]. Note that the user had credentials present, but "
            "do not preserve their values."
        )

        # Shared structured template (used by both paths).
        _template_sections = f"""## Active Task
[THE SINGLE MOST IMPORTANT FIELD. Copy the user's most recent request or
task assignment verbatim — the exact words they used. If multiple tasks
were requested and only some are done, list only the ones NOT yet completed.
The next assistant must pick up exactly here. Example:
"User asked: 'Now refactor the auth module to use JWT instead of sessions'"
If no outstanding task exists, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Example:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: patch]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: terminal]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include:
- Working directory and branch (if applicable)
- Modified/created files with brief note on each
- Test status (X/Y passing)
- Any running processes or servers
- Environment details that matter]

## In Progress
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so the next assistant does not re-answer them]

## Pending User Asks
[Questions or requests from the user that have NOT yet been answered or fulfilled. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Remaining Work
[What remains to be done — framed as context, not instructions]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.

Write only the summary body. Do not include any preamble or prefix."""

        if self._previous_summary:
            # Iterative update: preserve existing info, add new progress
            prompt = f"""{_summarizer_preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move items from "In Progress" to "Completed Actions" when done. Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete. CRITICAL: Update "## Active Task" to reflect the user's most recent unfulfilled request — this is the most important field for task continuity.

{_template_sections}"""
        else:
            # First compaction: summarize from scratch
            prompt = f"""{_summarizer_preamble}

Create a structured handoff summary for a different assistant that will continue this conversation after earlier turns are compacted. The next assistant should be able to understand what happened without re-reading the original turns.

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{_template_sections}"""

        # Inject focus topic guidance when the user provides one via /compress <focus>.
        # This goes at the end of the prompt so it takes precedence.
        if focus_topic:
            prompt += f"""

FOCUS TOPIC: "{focus_topic}"
The user has requested that this compaction PRIORITISE preserving all information related to the focus topic above. For content related to "{focus_topic}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to the focus topic, summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget. Even for the focus topic, NEVER preserve API keys, tokens, passwords, or credentials — use [REDACTED]."""

        try:
            call_kwargs = {
                "task": "compression",
                "main_runtime": {
                    "model": self.model,
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "api_mode": self.api_mode,
                },
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(summary_budget * 1.3),
                # timeout resolved from auxiliary.compression.timeout config by call_llm
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            response = call_llm(**call_kwargs)
            content = response.choices[0].message.content
            # Handle cases where content is not a string (e.g., dict from llama.cpp)
            if not isinstance(content, str):
                content = str(content) if content else ""
            # Redact the summary output as well — the summarizer LLM may
            # ignore prompt instructions and echo back secrets verbatim.
            summary = redact_sensitive_text(content.strip())
            # Store for iterative updates on next compaction
            self._previous_summary = summary
            self._summary_failure_cooldown_until = 0.0
            self._summary_model_fallen_back = False
            return self._with_summary_prefix(summary)
        except RuntimeError:
            # No provider configured — long cooldown, unlikely to self-resolve
            self._summary_failure_cooldown_until = time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            logging.warning("Context compression: no provider available for "
                            "summary. Middle turns will be dropped without summary "
                            "for %d seconds.",
                            _SUMMARY_FAILURE_COOLDOWN_SECONDS)
            return None
        except Exception as e:
            # If the summary model is different from the main model and the
            # error looks permanent (model not found, 503, 404), fall back to
            # using the main model instead of entering cooldown that leaves
            # context growing unbounded.  (#8620 sub-issue 4)
            _status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            _err_str = str(e).lower()
            _is_model_not_found = (
                _status in (404, 503)
                or "model_not_found" in _err_str
                or "does not exist" in _err_str
                or "no available channel" in _err_str
            )
            if (
                _is_model_not_found
                and self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                self._summary_model_fallen_back = True
                logging.warning(
                    "Summary model '%s' not available (%s). "
                    "Falling back to main model '%s' for compression.",
                    self.summary_model, e, self.model,
                )
                self.summary_model = ""  # empty = use main model
                self._summary_failure_cooldown_until = 0.0  # no cooldown
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)  # retry immediately

            # Transient errors (timeout, rate limit, network) — shorter cooldown
            _transient_cooldown = 60
            self._summary_failure_cooldown_until = time.monotonic() + _transient_cooldown
            logging.warning(
                "Failed to generate context summary: %s. "
                "Further summary attempts paused for %d seconds.",
                e,
                _transient_cooldown,
            )
            return None

    @staticmethod
    def _with_summary_prefix(summary: str) -> str:
        """Normalize summary text to the current compaction handoff format."""
        text = (summary or "").strip()
        for prefix in (LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """Extract the call ID from a tool_call entry (dict or SimpleNamespace)."""
        if isinstance(tc, dict):
            return tc.get("id", "")
        return getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """[Phase 5] 修复压缩后孤立的 tool_call / tool_result 配对。

        压缩会总结或丢弃中间的 assistant 消息（含 tool_calls）或 tool 消息（结果），
        导致两种 API 会拒绝的格式错误：

        模式 1 — 孤儿 tool result:
          tool 消息的 tool_call_id 引用的 assistant tool_call 已被压缩移除。
          API 错误: "No tool call found for function call output with call_id ..."
          修复: 删除这些孤儿 tool result 消息

        模式 2 — 孤儿 tool call:
          assistant 消息包含 tool_calls，但对应的 tool result 被删除/裁剪。
          API 要求在 assistant 后的下一条消息必须是对应的 tool result。
          修复: 为每个缺失结果插入 stub 消息：
                "[Result from earlier conversation — see context summary above]"

        这个方法是最后的安全检查——确保消息列表始终是 API 兼容的格式。
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = self._get_tool_call_id(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Remove tool results whose call_id has no matching assistant tool_call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self.quiet_mode:
                logger.info("Compression sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

        # 2. Add stub results for assistant tool_calls whose results were dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = self._get_tool_call_id(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result from earlier conversation — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            if not self.quiet_mode:
                logger.info("Compression sanitizer: added %d stub tool result(s)", len(missing_results))

        return messages

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Push a compress-start boundary forward past any orphan tool results.

        If ``messages[idx]`` is a tool result, slide forward until we hit a
        non-tool message so we don't start the summarised region mid-group.
        """
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Pull a compress-end boundary backward to avoid splitting a
        tool_call / result group.

        If the boundary falls in the middle of a tool-result group (i.e.
        there are consecutive tool messages before ``idx``), walk backward
        past all of them to find the parent assistant message.  If found,
        move the boundary before the assistant so the entire
        assistant + tool_results group is included in the summarised region
        rather than being split (which causes silent data loss when
        ``_sanitize_tool_pairs`` removes the orphaned tail results).
        """
        if idx <= 0 or idx >= len(messages):
            return idx
        # Walk backward past consecutive tool results
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        # If we landed on the parent assistant with tool_calls, pull the
        # boundary before it so the whole group gets summarised together.
        if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
            idx = check
        return idx

    # ------------------------------------------------------------------
    # Tail protection by token budget
    # ------------------------------------------------------------------

    def _find_last_user_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """Return the index of the last user-role message at or after *head_end*, or -1."""
        for i in range(len(messages) - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

    def _ensure_last_user_message_in_tail(
        self,
        messages: List[Dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Guarantee the most recent user message is in the protected tail.

        Context compressor bug (#10896): ``_align_boundary_backward`` can pull
        ``cut_idx`` past a user message when it tries to keep tool_call/result
        groups together.  If the last user message ends up in the *compressed*
        middle region the LLM summariser writes it into "Pending User Asks",
        but ``SUMMARY_PREFIX`` tells the next model to respond only to user
        messages *after* the summary — so the task effectively disappears from
        the active context, causing the agent to stall, repeat completed work,
        or silently drop the user's latest request.

        Fix: if the last user-role message is not already in the tail
        (``messages[cut_idx:]``), walk ``cut_idx`` back to include it.  We
        then re-align backward one more time to avoid splitting any
        tool_call/result group that immediately precedes the user message.
        """
        last_user_idx = self._find_last_user_message_idx(messages, head_end)
        if last_user_idx < 0:
            # No user message found beyond head — nothing to anchor.
            return cut_idx

        if last_user_idx >= cut_idx:
            # Already in the tail; nothing to do.
            return cut_idx

        # The last user message is in the middle (compressed) region.
        # Pull cut_idx back to it directly — a user message is already a
        # clean boundary (no tool_call/result splitting risk), so there is no
        # need to call _align_boundary_backward here; doing so would
        # unnecessarily pull the cut further back into the preceding
        # assistant + tool_calls group.
        if not self.quiet_mode:
            logger.debug(
                "Anchoring tail cut to last user message at index %d "
                "(was %d) to prevent active-task loss after compression",
                last_user_idx,
                cut_idx,
            )
        # Safety: never go back into the head region.
        return max(last_user_idx, head_end + 1)

    def _find_tail_cut_by_tokens(
        self, messages: List[Dict[str, Any]], head_end: int,
        token_budget: int | None = None,
    ) -> int:
        """[Phase 2] 按 token 预算从后往前确定尾部边界。

        核心策略:
          从消息列表末尾往回走，逐条累积 token 数，直到达到预算。
          尾部内的消息不被压缩，直接保留为原始对话。

        预算参数:
          token_budget 默认 = threshold_tokens × summary_target_ratio
          例如: 100K × 0.20 = 20K tokens 的尾部

        软上限策略 (soft ceiling):
          标准预算 × 1.5 = soft_ceiling (例: 20K × 1.5 = 30K)
          这样做的目的是避免在超大消息（如工具输出、文件读取）的
          中间切断——允许预算超额最多 50% 来得到一个干净的边界。

        多层保护:
          1. 硬最小: 始终保护至少 3 条消息（min_tail）
          2. 软上限: 预算允许超额 50%（避免在超大消息中间切断）
          3. 边界对齐: 不切断 tool_call/result 分组
          4. 用户消息锚定: 确保最新用户消息始终在尾部（#10896）

        边界情况:
          - 小会话: 如果 token 预算可以覆盖所有消息，强制在头部之后切割
            以确保压缩仍能移除中间轮次
          - 超大尾部: 即使最小 3 条消息也超过 soft_ceiling，
            切割线仍放在头部之后，压缩继续运行

        Returns:
            尾部起始索引（含），即 messages[tail_idx:] 是被保护的尾部
        """
        if token_budget is None:
            token_budget = self.tail_token_budget
        n = len(messages)
        # Hard minimum: always keep at least 3 messages in the tail
        min_tail = min(3, n - head_end - 1) if n - head_end > 1 else 0
        soft_ceiling = int(token_budget * 1.5)
        accumulated = 0
        cut_idx = n  # start from beyond the end

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            content = msg.get("content") or ""
            msg_tokens = len(content) // _CHARS_PER_TOKEN + 10  # +10 for role/metadata
            # Include tool call arguments in estimate
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            # Stop once we exceed the soft ceiling (unless we haven't hit min_tail yet)
            if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i

        # Ensure we protect at least min_tail messages
        fallback_cut = n - min_tail
        if cut_idx > fallback_cut:
            cut_idx = fallback_cut

        # If the token budget would protect everything (small conversations),
        # force a cut after the head so compression can still remove middle turns.
        if cut_idx <= head_end:
            cut_idx = max(fallback_cut, head_end + 1)

        # Align to avoid splitting tool groups
        cut_idx = self._align_boundary_backward(messages, cut_idx)

        # Ensure the most recent user message is always in the tail so the
        # active task is never lost to compression (fixes #10896).
        cut_idx = self._ensure_last_user_message_in_tail(messages, cut_idx, head_end)

        return max(cut_idx, head_end + 1)

    # ------------------------------------------------------------------
    # Main compression entry point
    # ------------------------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, focus_topic: str = None) -> List[Dict[str, Any]]:
        """压缩对话消息列表 — 将中间轮次替换为结构化摘要。

        完整 5 阶段算法:
          Phase 1: 工具结果裁剪 — 零 LLM 调用的本地去重 + 摘要化
                   - 重复的工具结果去重（只保留最新完整内容）
                   - 旧结果替换为 1 行描述
                   - 旧 tool_call arguments 截断
          Phase 2: 边界定位 — 确定头部（不压缩）和尾部（不压缩）的范围
                   - 头部: 前 protect_first_n 条消息（默认 3）
                   - 尾部: 按 token 预算从后往前累积
                   - 边界对齐: 不切断 tool_call/result 配对
                   - 用户消息锚定: 确保最新用户消息一定在尾部
          Phase 3: LLM 摘要 — 用辅助模型生成结构化摘要
                   - 首次: 从零总结；后续: 迭代更新已有摘要
                   - 支持 focus_topic 引导压缩
          Phase 4: 消息组装 — 头 + 摘要消息 + 尾
                   - 摘要消息插入到头部和尾部之间
                   - 避免与邻居相同 role（防止 API 拒绝）
                   - 在 system prompt 中注入压缩提示
          Phase 5: 工具对修复 — 清理孤立的 tool_call/result
                   - 删除缺少 tool_call 的 tool result
                   - 为缺少 tool result 的 tool_call 插入 stub

        返回压缩后的消息列表。如果消息太少无法压缩，返回原列表。

        Args:
            messages: 完整的 OpenAI 格式消息列表
            current_tokens: 当前估算的 token 数（用于日志和反抖动）
            focus_topic: 可选话题，优先保留相关信息（/compress <topic>）
        """
        n_messages = len(messages)
        # Only need head + 3 tail messages minimum (token budget decides the real tail size)
        _min_for_compress = self.protect_first_n + 3 + 1
        if n_messages <= _min_for_compress:
            if not self.quiet_mode:
                logger.warning(
                    "Cannot compress: only %d messages (need > %d)",
                    n_messages, _min_for_compress,
                )
            return messages

        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # Phase 1: Prune old tool results (cheap, no LLM call)
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 2: Determine boundaries
        compress_start = self.protect_first_n
        compress_start = self._align_boundary_forward(messages, compress_start)

        # Use token-budget tail protection instead of fixed message count
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            return messages

        turns_to_summarize = messages[compress_start:compress_end]

        if not self.quiet_mode:
            logger.info(
                "Context compression triggered (%d tokens >= %d threshold)",
                display_tokens,
                self.threshold_tokens,
            )
            logger.info(
                "Model context limit: %d tokens (%.0f%% = %d)",
                self.context_length,
                self.threshold_percent * 100,
                self.threshold_tokens,
            )
            tail_msgs = n_messages - compress_end
            logger.info(
                "Summarizing turns %d-%d (%d turns), protecting %d head + %d tail messages",
                compress_start + 1,
                compress_end,
                len(turns_to_summarize),
                compress_start,
                tail_msgs,
            )

        # Phase 3: Generate structured summary
        summary = self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

        # Phase 4: Assemble compressed message list
        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content")
                _compression_note = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work.]"
                if _compression_note not in _content_text_for_contains(existing):
                    msg["content"] = _append_text_to_content(
                        existing,
                        "\n\n" + _compression_note if isinstance(existing, str) and existing else _compression_note,
                    )
            compressed.append(msg)

        # If LLM summary failed, insert a static fallback so the model
        # knows context was lost rather than silently dropping everything.
        if not summary:
            if not self.quiet_mode:
                logger.warning("Summary generation failed — inserting static fallback context marker")
            n_dropped = compress_end - compress_start
            summary = (
                f"{SUMMARY_PREFIX}\n"
                f"Summary generation was unavailable. {n_dropped} conversation turns were "
                f"removed to free context space but could not be summarized. The removed "
                f"turns contained earlier work in this session. Continue based on the "
                f"recent messages below and the current state of any files or resources."
            )

        _merge_summary_into_tail = False
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"
        # Pick a role that avoids consecutive same-role with both neighbors.
        # Priority: avoid colliding with head (already committed), then tail.
        if last_head_role in ("assistant", "tool"):
            summary_role = "user"
        else:
            summary_role = "assistant"
        # If the chosen role collides with the tail AND flipping wouldn't
        # collide with the head, flip it.
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped
            else:
                # Both roles would create consecutive same-role messages
                # (e.g. head=assistant, tail=user — neither role works).
                # Merge the summary into the first tail message instead
                # of inserting a standalone message that breaks alternation.
                _merge_summary_into_tail = True
        if not _merge_summary_into_tail:
            compressed.append({"role": summary_role, "content": summary})

        for i in range(compress_end, n_messages):
            msg = messages[i].copy()
            if _merge_summary_into_tail and i == compress_end:
                merged_prefix = (
                    summary
                    + "\n\n--- END OF CONTEXT SUMMARY — "
                    "respond to the message below, not the summary above ---\n\n"
                )
                msg["content"] = _append_text_to_content(
                    msg.get("content"),
                    merged_prefix,
                    prepend=True,
                )
                _merge_summary_into_tail = False
            compressed.append(msg)

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        new_estimate = estimate_messages_tokens_rough(compressed)
        saved_estimate = display_tokens - new_estimate

        # Anti-thrashing: track compression effectiveness
        savings_pct = (saved_estimate / display_tokens * 100) if display_tokens > 0 else 0
        self._last_compression_savings_pct = savings_pct
        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

        if not self.quiet_mode:
            logger.info(
                "Compressed: %d -> %d messages (~%d tokens saved, %.0f%%)",
                n_messages,
                len(compressed),
                saved_estimate,
                savings_pct,
            )
            logger.info("Compression #%d complete", self.compression_count)

        return compressed
