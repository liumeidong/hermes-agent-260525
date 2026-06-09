"""上下文引擎抽象基类 — 支持可插拔的上下文压缩策略。

================================================================================
设计目的
================================================================================
ContextEngine 定义了上下文管理的统一接口，允许将不同的压缩策略
作为插件使用。默认的内置实现是 ContextCompressor（基于 LLM 摘要的压缩），
但第三方可以通过 plugin 系统提供替代方案（如 LCM、长上下文记忆等）。

================================================================================
插件选择
================================================================================
通过 config.yaml 中的 context.engine 键指定：
  context:
    engine: "compressor"     # 默认：使用内置 ContextCompressor
    engine: "lcm"            # 第三方引擎（需安装对应 plugin）

引擎搜索路径（按优先级）：
  1. plugins/context_engine/<name>/   — 项目本地插件
  2. ~/.hermes/plugins/context_engine/<name>/  — 用户级插件
  3. 内置: "compressor" → ContextCompressor

================================================================================
引擎职责
================================================================================
每个引擎负责：
  - 判断何时需要压缩（should_compress）
  - 执行压缩（compress）— 返回压缩后的消息列表
  - 跟踪 token 使用情况（update_from_response）
  - 可选地提供 Agent 可调用的工具（get_tool_schemas）
  - 可选地处理会话生命周期（on_session_start/end）

================================================================================
生命周期
================================================================================
  1. 引擎被实例化并注册（plugin register() 或默认）
  2. on_session_start() — 新会话开始时调用
  3. update_from_response() — 每次 API 响应后更新 token 计数
  4. should_compress() — 每轮对话后检查
  5. compress() — 当 should_compress() 返回 True 时调用
  6. on_session_end() — 会话结束时调用（CLI 退出、/reset、gateway 过期）
     注意: 不是每轮调用，只在真正的会话边界调用

================================================================================
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ContextEngine(ABC):
    """所有上下文引擎必须实现的抽象基类。

    Attributes (引擎必须维护，run_agent.py 直接读取):
        last_prompt_tokens:   上次 API 调用的 prompt token 数
        last_completion_tokens:   上次 API 调用的 completion token 数
        last_total_tokens:    上次 API 调用的总 token 数
        threshold_tokens:     压缩触发阈值（token 数）
        context_length:       模型的上下文窗口大小（token 数）
        compression_count:    当前会话中压缩的次数
        threshold_percent:    触发压缩的阈值比例（默认 0.75，Compressor 默认 0.50）
        protect_first_n:      头部保护的消息数
        protect_last_n:       尾部保护的硬最小消息数
    """

    # -- 引擎标识 ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎短名称（如 'compressor', 'lcm'）— 用于配置选择。"""

    # -- Token 状态（由 run_agent.py 读取用于显示/日志）---------------------
    #
    # 引擎必须维护这些属性。run_agent.py 直接读取它们。

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # -- 压缩参数（由 run_agent.py 预检时读取）------------------------------

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # -- 核心接口 ----------------------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """从 API 响应更新跟踪的 token 使用量。

        每次 LLM 调用后使用响应的 usage dict 调用。
        """

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """判断本回合是否需要触发上下文压缩。"""

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
    ) -> List[Dict[str, Any]]:
        """压缩消息列表并返回新的消息列表。

        这是主入口点。引擎接收完整的消息列表，返回（可能更短的）
        列表。实现可以是摘要、DAG 构建或任何其他方式——
        只要返回的列表是有效的 OpenAI 格式消息序列。
        """

    # -- 可选: 预检 ---------------------------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """API 调用前的快速粗略检查（没有实际 token 计数）。

        默认返回 False（跳过预检）。如果你的引擎可以做便宜的估算，覆盖此方法。
        """
        return False

    # -- 可选: 会话生命周期 -------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """新会话开始时调用。

        用于加载会话的持久化状态（DAG、store 等）。
        kwargs 可能包含 hermes_home, platform, model 等。
        """

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """会话结束时调用（CLI 退出、/reset、gateway 过期）。

        用于刷新状态、关闭数据库连接等。
        注意: 不是每轮调用，只在真正的会话边界调用。
        """

    def on_session_reset(self) -> None:
        """/new 或 /reset 时调用。重置会话级状态。

        默认重置 compression_count 和 token 跟踪。
        """
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    # -- 可选: 工具 ---------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回引擎提供给 Agent 的工具 schema。

        默认返回空列表（无工具）。LCM 会在这里返回
        lcm_grep, lcm_describe, lcm_expand 等 schema。
        """
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理来自 Agent 的工具调用。

        只为 get_tool_schemas() 返回的工具名调用。
        必须返回 JSON 字符串。

        kwargs 可能包含:
          messages: 当前内存中的消息列表（用于实时摄入）
        """
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- 可选: 状态/显示 ----------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """返回状态 dict 用于显示/日志。

        默认返回 run_agent.py 期望的标准字段。
        """
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length else 0
            ),
            "compression_count": self.compression_count,
        }

    # -- 可选: 模型切换支持 ------------------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        """用户切换模型或回退激活时调用。

        默认更新 context_length 并根据 threshold_percent 重新计算 threshold_tokens。
        如果你的引擎需要更多操作（如重新计算 DAG 预算、切换摘要模型等），覆盖此方法。
        """
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
