#!/usr/bin/env python3
"""
Memory Tool Module — 持久化、可筛选的文件记忆系统。

================================================================================
设计概览
================================================================================
本模块实现 Hermes 的内置记忆系统（BuiltinMemoryProvider）。所有内容都存储在
文件系统中，跨 session 持久化，并被注入到 system prompt 作为基础上下文。

双存储：
  - MEMORY.md：Agent 自己的笔记（环境事实、项目惯例、工具 quirks、教训）
  - USER.md：对用户的了解（偏好、沟通风格、期望、工作流习惯）

字符上限（不是 token，因为字符数与模型无关）：
  - MEMORY.md: 2200 chars（~800 tokens）
  - USER.md:   1375 chars（~500 tokens）

================================================================================
关键设计：冻结快照模式（Frozen Snapshot Pattern）
================================================================================
会话启动时：
  1. 从磁盘读取 MEMORY.md / USER.md
  2. 精确去重（dict.fromkeys）
  3. 生成"冻结快照"注入到 system prompt
  4. 此快照在整个会话期间永不变

会话进行中（模型调用 memory 工具）：
  1. 修改实时状态（memory_entries / user_entries）
  2. 立即原子写入磁盘（崩溃可恢复）
  3. **不修改**冻结快照

下次会话启动：重新从磁盘加载 + 刷新快照。

为什么这样设计？
  - System prompt 的任何修改都会使前缀缓存（prefix cache）失效
  - 冻结快照保证 prefix cache 100% 命中，节省 50%+ 输入 token
  - 中途修改 system prompt 是不必要的成本

================================================================================
加载策略
================================================================================
方式：启动时**全量加载**（load_from_disk）
  - 一次性读取所有条目
  - 仅做精确去重
  - 不做语义过滤或相似度筛选
  - 文件字符上限保证全量加载也无开销

为什么全量加载？
  - 文件本身有字符上限（2200/1375 chars），容量可控
  - 保证模型永远能看到所有关键上下文
  - 避免"漏掉了重要记忆"导致错误

================================================================================
工具接口
================================================================================
单一工具，四个 action：
  - add     ：追加新条目（精确去重，字符上限校验）
  - replace ：用短子串定位条目并替换（语义矛盾由 LLM 决策）
  - remove  ：用短子串定位条目并删除
  - read    ：读取当前完整内容

匹配机制：replace/remove 用短子串匹配（不需要 ID）：
  - 1 个匹配 → 直接操作
  - 0 个匹配 → 报错"未找到"
  - 多条不同匹配 → 报错"更精确"，列出匹配项预览
  - 多条相同匹配（重复）→ 操作第一条

================================================================================
矛盾解决机制
================================================================================
内置记忆的矛盾处理**完全由 LLM 自主决策**：
  - 模型检测到矛盾时调用 replace（短子串匹配定位旧条目）
  - 模型发现事实不再成立时调用 remove
  - 字符上限满时 add 失败 → 错误消息提示需先 replace/remove

示例：处理矛盾陈述
  # 第一轮：模型保存
  memory(action="add", target="user", content="User prefers 4-space indentation")

  # 第二轮：用户纠正 "Actually I use 2 spaces now"
  # 模型应当调用：
  memory(action="replace", target="user",
         old_text="4-space indentation",
         content="2-space indentation (corrected 2026-06-10)")
  # 而不是 add 一条新条目与旧的并存

全局擦除：仅 `hermes memory reset` 提供，需用户确认。

================================================================================
缺失的能力（设计选择）
================================================================================
内置记忆刻意没有以下能力（设计哲学：LLM 自主判断 + 硬约束更可靠）：
  - 自动遗忘 / TTL（条目永不过期）
  - 基于相似度的去重（仅检测完全相同）
  - 自动矛盾检测（不做语义冲突检测）
  - 使用频率衰减（不区分常用/罕见）
  - 重要性评分（没有权重概念）

需要这些能力时，可配置外部 provider（Honcho/Mem0 等）。

================================================================================
安全防护
================================================================================
记忆内容会注入到 system prompt，所以必须扫描写入内容：
  - 提示注入模式（ignore previous instructions / you are now）
  - 凭据外泄模式（curl $KEY / cat .env）
  - 后门持久化（authorized_keys / ~/.ssh）
  - 不可见 Unicode 字符（零宽 / RTL 覆盖）

================================================================================
文件格式
================================================================================
条目分隔符：`§`（section sign）。条目可多行。
写入使用 temp file + atomic rename，避免并发读看到空文件。

================================================================================
文件锁
================================================================================
Unix 使用 fcntl.flock，Windows 使用 msvcrt.locking。
锁文件与数据文件分离（*.lock），数据文件本身通过原子 rename 替换。
"""

import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    # Exfiltration via curl/wget with secrets
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # Persistence via shell rc
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

# Subset of invisible chars for injection detection
_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    """扫描记忆内容中的注入/外泄模式。被阻止时返回错误字符串。

    扫描类型：
      1. 不可见 Unicode 字符（零宽、RTL 覆盖）
         - 攻击者用这些字符隐藏恶意指令或改变文本视觉顺序
      2. 提示注入模式
         - "ignore previous instructions" — 试图覆盖系统提示
         - "you are now ..." — 角色劫持
         - "do not tell the user" — 隐藏行为
         - "system prompt override" — 明确覆盖尝试
         - "disregard your rules" — 规则绕过
         - "act as if you have no restrictions" — 限制绕过
      3. 凭据外泄模式
         - curl / wget 配合 $KEY/$TOKEN/$SECRET 等环境变量
         - cat 读取 .env、credentials、.netrc 等敏感文件
      4. 后门持久化模式
         - authorized_keys 写入（SSH 后门）
         - ~/.ssh 路径引用
         - ~/.hermes/.env 访问

    为什么必须扫描？
      记忆内容会注入到 system prompt。如果记忆被污染，模型行为会被劫持。
      这是"信任边界保护" — 防止记忆本身成为攻击向量。
    """
    # 检查不可见 Unicode 字符
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."

    # 检查威胁模式
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."

    return None


class MemoryStore:
    """
    有界的、可筛选的文件持久化记忆存储。每个 AIAgent 一个实例。

    维护两套并行状态：
      - _system_prompt_snapshot：会话启动时拍一次，会话期间永不变。
        用于注入到 system prompt。永不在会话中途修改 — 这保护了 prefix cache。
      - memory_entries / user_entries：实时状态，由工具调用修改并落盘。
        工具响应始终反映这个实时状态。

    关键设计决策：
      - 冻结快照：system prompt 在整个会话期间稳定 = prefix cache 100% 命中
      - 字符上限：保证全量加载到 system prompt 也只占用 ~800 tokens
      - 精确去重：仅检测完全相同，不做语义去重（语义去重交给 LLM 决策）
      - 短子串匹配：replace/remove 用文本片段定位，无需 ID

    矛盾解决模型：
      内置层没有自动矛盾检测。LLM 通过调用 memory(action="replace") 或
      memory(action="remove") 来解决语义矛盾。无 TTL，无自动遗忘，无衰减。
    """

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        """
        初始化 MemoryStore，但不加载文件。必须调用 load_from_disk() 才有数据。

        Args:
            memory_char_limit: MEMORY.md 的字符上限（默认 2200）
            user_char_limit:   USER.md 的字符上限（默认 1375）
        """
        # 实时状态：工具调用修改 + 立即落盘
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []

        # 字符上限（不是 token，因为字符数与模型无关）
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit

        # 冻结快照：会话启动时拍一次，会话期间永不变
        # 用于 format_for_system_prompt() 注入到 system prompt
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    def load_from_disk(self):
        """
        从 MEMORY.md 和 USER.md 加载所有条目，拍系统提示快照。

        加载策略：**全量加载**（非过滤检索）。
          - 一次性读取所有条目到内存
          - 仅做精确去重（dict.fromkeys 保留顺序）
          - 不做语义过滤或相似度筛选
          - 文件字符上限保证全量加载也无开销

        为什么全量加载？
          - 文件本身有字符上限（2200/1375 chars），容量可控
          - 保证模型永远能看到所有关键上下文
          - 避免"漏掉了重要记忆"导致错误
          - 不需要额外的向量检索开销

        触发时机：会话启动时调用一次。会话期间不重新加载。
        """
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        # 读取所有条目（全量加载）
        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # 仅做精确去重（保留顺序，保留首次出现）
        # 注意：这不是语义去重，仅检测完全相同的条目
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # 拍冻结快照 — 会话期间永不变
        # format_for_system_prompt() 返回这个快照，保证 prefix cache 稳定
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
            lock_path.write_text(" ", encoding="utf-8")

        fd = open(lock_path, "r+" if msvcrt else "a+")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str):
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        """
        fresh = self._read_file(self._path_for(target))
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """追加新条目。返回错误如果超过字符上限。

        执行流程：
          1. 验证内容非空
          2. 扫描注入/外泄模式（提示注入、curl $KEY、cat .env 等）
          3. 获取文件锁 + 重新从磁盘读取（捕获其他 session 的写入）
          4. 精确去重检查（仅检测完全相同）
          5. 字符上限预检 — 超限则拒绝并提示先 replace/remove
          6. 追加到内存条目 + 原子写入磁盘

        字符上限是**间接的淘汰机制**：
          - 上限满时 add 失败
          - 错误消息提示"先 replace or remove existing entries"
          - 把淘汰决定权交给 LLM（而不是自动算法）

        Args:
            target: "memory" 或 "user"
            content: 要追加的条目内容

        Returns:
            成功：{"success": True, "entries": [...], "usage": "X/Y chars"}
            失败：{"success": False, "error": "...", "current_entries": [...]}
        """
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # 写入前扫描注入/外泄模式（记忆会注入到 system prompt）
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # 在锁内重新从磁盘读取（捕获其他 session / CLI 的并发写入）
            self._reload_target(target)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # 仅拒绝完全相同的条目（语义去重由 LLM 通过 replace 处理）
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # 预计算新总字符数，判断是否超限
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                # 字符上限已满 — 拒绝添加，要求先腾出空间
                # 这是"硬约束驱动的淘汰"机制：把决定权交给 LLM
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """用短子串定位条目并替换 — 内置记忆矛盾解决的核心路径。

        匹配机制：
          - old_text 在条目中作为子串查找（不需要完整文本或 ID）
          - 0 匹配 → 报错"未找到"
          - 1 匹配 → 直接替换
          - 多条不同匹配 → 报错并列出预览，要求 LLM 提供更精确的 old_text
          - 多条相同匹配（重复条目）→ 操作第一条

        这是内置记忆**矛盾解决的核心路径**：
          LLM 检测到旧记忆与新事实矛盾时调用此方法更新。
          不做语义匹配，只做子串匹配 — 简单可靠。

        Args:
            target: "memory" 或 "user"
            old_text: 标识要替换条目的子串
            new_content: 新的条目内容

        Returns:
            成功：{"success": True, "entries": [...]}
            失败：{"success": False, "error": "...", "matches": [...]}
        """
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            # 强制要求显式选择 remove，避免意外删除
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # 扫描替换内容（防止通过替换注入恶意内容）
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """删除包含 old_text 子串的条目。

        触发场景：模型判断某条记忆不再正确或不再有用时调用。
        匹配机制与 replace 相同：短子串 + 歧义拒绝。

        注意：
          - 删除是软决策（不可逆），仅当 LLM 明确调用才执行
          - 没有 TTL/自动遗忘 — 必须显式 remove
          - 全局擦除仅通过 `hermes memory reset`（CLI），需用户确认
          - 删除安全检查：扫描内容、文件锁、歧义拒绝

        Args:
            target: "memory" 或 "user"
            old_text: 标识要删除条目的子串

        Returns:
            成功：{"success": True, "entries": [...], "entry_count": N}
            失败：{"success": False, "error": "...", "matches": [...]}
        """
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = set(e for _, e in matches)
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        返回冻结快照用于注入到 system prompt。

        关键行为：返回 load_from_disk() 时拍下的状态，**不是实时状态**。
        会话期间的写入不会影响这个返回值。这保证 system prompt 在整个会话期间
        稳定不变，保护 prefix cache 命中率。

        加载策略回顾：
          - 启动时一次性全量加载所有条目到 _system_prompt_snapshot
          - 注入到 system prompt 后永不变
          - 模型中途调用 memory 工具修改的是实时状态，不影响快照
          - 下次会话启动时重新从磁盘加载 → 刷新快照

        Returns:
            格式化的快照字符串（带分隔线和字符用量指示），或 None（快照为空）
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """读取记忆文件并拆分为条目。

        全量加载：
          读取整个文件的所有条目，无任何过滤或截断。
          文件本身的字符上限保证加载量可控（2200/1375 chars）。

        不需要文件锁：
          _write_file 使用原子 rename，所以读者要么看到旧的完整文件，
          要么看到新的完整文件 — 永远不会看到部分写入状态。

        拆分逻辑：
          使用 ENTRY_DELIMITER (§) 拆分，保留顺序。
          必须用完整的分隔符 — 仅按 § 拆分会导致内容中含 § 的条目被错误切分。
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # 使用 ENTRY_DELIMITER 拆分，与 _write_file 保持一致
        # 仅按 § 拆分会错误切分内容中含 § 的条目
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """使用原子 temp-file + rename 写入记忆文件。

        为什么用原子 rename？
          之前的实现用 open("w") + flock，但 "w" 模式**先截断文件再获取锁**，
          造成竞争窗口：并发读者可能看到空文件。
          原子 rename 避免了这个问题：读者要么看到旧完整文件，要么看到新完整文件。

        步骤：
          1. 在同目录创建临时文件（保证同文件系统，原子 rename）
          2. 写入内容 + flush + fsync（确保落盘）
          3. os.replace(tmp, path) — 原子替换
          4. 失败时清理临时文件
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # 在同目录创建临时文件（同文件系统才能保证原子 rename）
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))  # 原子替换
            except BaseException:
                # 失败时清理临时文件
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """memory 工具的统一入口。分发到 MemoryStore 的对应方法。

    加载策略说明：
      - 此函数不直接读取文件
      - 实际的数据加载在会话启动时由 MemoryStore.load_from_disk() 完成
      - 本函数只负责调用 add/replace/remove 方法修改实时状态并落盘
      - system prompt 看到的是 load_from_disk() 时拍的冻结快照（永不变）

    Args:
        action: "add" / "replace" / "remove"
        target: "memory" (Agent 笔记) 或 "user" (用户画像)
        content: 条目内容（add/replace 必填）
        old_text: 标识条目的子串（replace/remove 必填）
        store: MemoryStore 实例（由 run_agent.py 注入）

    Returns:
        JSON 字符串，包含 success / entries / usage / message 等字段
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in ("memory", "user"):
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS:\n"
        "- add: append a new entry. If memory is at the char limit, add fails -- you must "
        "replace or remove existing entries first to make room.\n"
        "- replace: update an existing entry. old_text must be a unique substring that "
        "identifies the entry. If multiple entries match (and they aren't exact duplicates), "
        "the call fails with a preview list -- provide more specific old_text.\n"
        "- remove: delete an entry. Same matching rules as replace. Use when a fact is no "
        "longer true (e.g. user changed jobs, deprecated API).\n\n"
        "CONFLICT RESOLUTION:\n"
        "You are responsible for detecting and resolving contradictions. If the user corrects "
        "a previous statement, call replace with the old entry's distinctive substring. Do "
        "NOT just add a new entry -- the old one will conflict.\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": (
                    "The action to perform:\n"
                    "- 'add': append a new entry (exact-duplicate detection + char limit)\n"
                    "- 'replace': update existing entry matched by old_text substring\n"
                    "- 'remove': delete entry matched by old_text substring"
                ),
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": (
                    "Short unique substring identifying the entry to replace or remove. "
                    "Must be specific enough to match exactly one entry (or only identical "
                    "duplicates). If multiple distinct entries match, the call fails and "
                    "returns a preview list -- provide a longer/more specific substring."
                ),
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




