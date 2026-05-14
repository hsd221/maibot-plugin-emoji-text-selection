"""基于表情包描述文本的表情包选择插件。

通过 ``@Tool("select_emoji", core_tool=True)`` 注册为直接可见的核心工具。
选择 LLM 直接阅读接近 planner 格式的对话上下文，自行判断情绪基调后选出最匹配的表情包。
1. 获取全量情绪标签，并发为每个标签取回一张代表性表情包
2. 获取当前对话上下文，构建为 planner 风格的结构化文本
3. 按 description 去重后编号，与上下文一起发给文本 LLM 选出最匹配的
4. 发送选中的表情包，匹配失败则记错并返回 failure（不随机兜底）
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from maibot_sdk import HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import HookMode, ToolParameterInfo, ToolParamType
from pydantic import Field


# ─── 配置模型 ───────────────────────────────────────────────────


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"

    enabled: bool = Field(
        default=True,
        description="是否启用插件",
        json_schema_extra={"label": "启用"},
    )
    config_version: str = Field(
        default="1.0.0",
        description="配置版本号",
        json_schema_extra={"label": "配置版本"},
    )


class EmojiSelectorSectionConfig(PluginConfigBase):
    """表情包选择行为配置。"""

    __ui_label__ = "表情包选择"

    max_emotion_tags: int = Field(
        default=50,
        description="传给 LLM 的最大情绪标签数量，0 表示不限制",
        json_schema_extra={"label": "最大情绪标签数"},
    )
    max_selected_tags: int = Field(
        default=5,
        description="LLM 最多选择的标签数（当前版本固定单选，此配置暂未启用）",
        json_schema_extra={"label": "最多选择标签数"},
    )
    llm_model: Literal["emoji", "utils", "planner", "reply"] = Field(
        default="emoji",
        description="标签选择用的模型任务名",
        json_schema_extra={"label": "LLM 模型"},
    )
    filter_send_emoji: bool = Field(
        default=False,
        description="启用后从 planner 工具列表移除内置 send_emoji，避免 LLM 绕过本插件直接发送",
        json_schema_extra={"label": "过滤原生 send_emoji"},
    )


class EmojiTextSelectorConfig(PluginConfigBase):
    """表情包文本选择器配置。"""

    __ui_label__ = "表情包文本选择器"

    plugin: PluginSectionConfig = Field(
        default_factory=PluginSectionConfig,
    )
    selector: EmojiSelectorSectionConfig = Field(
        default_factory=EmojiSelectorSectionConfig,
    )


# ─── 提示词模板 ───────────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent
_SELECTION_PROMPT_PATH = _PROMPT_DIR / "select_emoji.prompt"

_FALLBACK_SELECTION_PROMPT = """\
阅读以下对话上下文，根据对话的情绪基调，从{emoji_count}个表情包描述中选择最匹配的一个：

{conversation_context}
{emotion_hint_block}
{description_list}

仅返回JSON：{{"selected": 3}}（单个编号）"""


def _load_prompt_template() -> str:
    """加载提示词模板文件。文件不存在时返回内置默认值。"""
    try:
        return _SELECTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return _FALLBACK_SELECTION_PROMPT


# ─── 纯逻辑函数 ─────────────────────────────────────────────────


def _build_conversation_context(messages: list[dict[str, Any]]) -> str:
    """将原始消息列表构建为接近 planner 格式的对话上下文。

    planner 看到的格式是:
        [msg_id]xxx
        [时间]HH:MM:SS
        [用户名]xxx
        [用户群昵称]xxx
        [发言内容]xxx
    """
    blocks: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_info = msg.get("message_info", {})
        user_info = msg_info.get("user_info", {}) if isinstance(msg_info, dict) else {}

        msg_id = msg.get("message_id", "")
        timestamp = msg.get("timestamp", "")
        user_name = user_info.get("user_nickname", "")
        user_card = user_info.get("user_cardname", "")
        content = msg.get("processed_plain_text", "") or ""

        if not content.strip():
            continue

        # 时间格式化
        try:
            ts = float(timestamp)
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except (ValueError, TypeError):
            time_str = str(timestamp)

        lines = []
        if msg_id:
            lines.append(f"[msg_id]{msg_id}")
        lines.append(f"[时间]{time_str}")
        lines.append(f"[用户名]{user_name}")
        if user_card and user_card != user_name:
            lines.append(f"[用户群昵称]{user_card}")
        lines.append(f"[发言内容]{content}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _build_selection_prompt(
    descriptions: list[str],
    conversation_context: str = "",
    emotion_hint: str = "",
) -> str:
    """构建发给文本 LLM 的表情包选择 prompt。从 .prompt 文件加载模板。"""
    template = _load_prompt_template()
    return template.format(
        emoji_count=str(len(descriptions)),
        conversation_context=conversation_context,
        emotion_hint_block=(
            f"辅助参考——planner 判断的情绪倾向：{emotion_hint}"
            if emotion_hint else ""
        ),
        description_list="\n".join(
            f"{i+1}. {d}" for i, d in enumerate(descriptions)
        ),
    )


def _parse_llm_index(response_text: str, max_count: int) -> int | None:
    """从 LLM 返回的 JSON 中解析选中的单个编号。解析失败返回 None。"""
    text = (response_text or "").strip()
    if not text:
        return None

    candidates = [text]
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.find("```", start)
        if end > start:
            candidates.append(text[start:end].strip())
    if "```" in text:
        start = text.index("```") + 3
        end = text.find("```", start)
        if end > start:
            candidates.append(text[start:end].strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            # 尝试 {"selected": 3} 或 {"selected": [1, 3]}
            for key in ("selected", "index", "indices"):
                raw = data.get(key)
                if isinstance(raw, list) and raw:
                    try:
                        idx = int(raw[0])
                    except (ValueError, TypeError):
                        continue
                    if 1 <= idx <= max_count:
                        return idx
                if isinstance(raw, (int, float)):
                    idx = int(raw)
                    if 1 <= idx <= max_count:
                        return idx
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    return None


# ─── 插件主类 ───────────────────────────────────────────────────


class EmojiTextSelectorPlugin(MaiBotPlugin):
    """纯文本情绪标签匹配的表情包选择插件。"""

    config_model: ClassVar[type[PluginConfigBase] | None] = EmojiTextSelectorConfig

    def __init__(self) -> None:
        super().__init__()
        self._config: EmojiTextSelectorConfig | None = None

    @property
    def config(self) -> EmojiTextSelectorConfig:
        if self._config is None:
            self._config = EmojiTextSelectorConfig()
        return self._config

    @classmethod
    def build_config_schema(
        cls,
        *,
        plugin_id: str = "",
        plugin_name: str = "",
        plugin_version: str = "",
        plugin_description: str = "",
        plugin_author: str = "",
    ) -> dict[str, Any]:
        schema = super().build_config_schema(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_description=plugin_description,
            plugin_author=plugin_author,
        )
        # 隐藏 [plugin] 节（含 config_version），普通用户无需关心
        schema.get("sections", {}).pop("plugin", None)
        return schema

    # ─── 生命周期 ────────────────────────────────────────────

    async def on_load(self) -> None:
        self.ctx.logger.info("[EmojiTextSelector] 插件已加载")

    async def on_unload(self) -> None:
        self.ctx.logger.info("[EmojiTextSelector] 插件已卸载")

    async def on_config_update(
        self, scope: str, config_data: dict[str, Any], version: str
    ) -> None:
        if scope == "self":
            self.set_plugin_config(config_data)
            self._config = None  # 下次访问时重建

    # ─── Tool: select_emoji ──────────────────────────────────

    @Tool(
        name="select_emoji",
        description="根据当前对话情绪从表情包库中选择并发送合适的表情包。",
        parameters=[
            ToolParameterInfo(
                name="emotion_hint",
                param_type=ToolParamType.STRING,
                description="可选的情绪提示词，帮助模型更准确地选择表情包",
                required=False,
            ),
        ],
        core_tool=True,
    )
    async def handle_select_emoji(
        self,
        stream_id: str = "",
        emotion_hint: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """处理 select_emoji 工具调用。

        1. 获取全量情绪标签，并发为每个标签取回一张代表性表情包
        2. 获取当前对话上下文作为额外参考
        3. 去重后把表情包描述编号发给文本 LLM 选择一个
        4. 发送选中的表情包，匹配失败则返回 error（不随机兜底）
        """
        del kwargs

        try:
            # 1. 获取全量情绪标签
            emotions: list[str] = await self.ctx.emoji.get_emotions()
            if not emotions:
                return {"success": False, "error": "表情包库中没有可用标签"}

            max_tags = self.config.selector.max_emotion_tags
            if max_tags > 0 and len(emotions) > max_tags:
                emotions = emotions[:max_tags]

            # 2. 并发为每个标签取回一张代表性表情包
            semaphore = asyncio.Semaphore(10)

            async def _fetch_one(tag: str) -> tuple[str, Any]:
                async with semaphore:
                    try:
                        return tag, await self.ctx.emoji.get_by_description(tag)
                    except Exception as exc:
                        self.ctx.logger.debug(
                            f"[EmojiTextSelector] get_by_description('{tag}') 异常: {exc}"
                        )
                        return tag, None

            # 3. 并行拉取对话上下文（与表情包查询同时进行）
            async def _fetch_context() -> str:
                if not stream_id:
                    return ""
                try:
                    messages = await self.ctx.message.get_recent(stream_id, limit=30)
                    if not messages:
                        return ""
                    return _build_conversation_context(messages)
                except Exception as exc:
                    self.ctx.logger.debug(
                        f"[EmojiTextSelector] 获取对话上下文异常: {exc}"
                    )
                    return ""

            results, extra_context = await asyncio.gather(
                asyncio.gather(*(_fetch_one(t) for t in emotions)),
                _fetch_context(),
            )

            # 4. 按 description 去重，构建编号→表情包映射
            desc_to_emoji: dict[str, dict[str, str]] = {}
            ordered_descriptions: list[str] = []
            for _tag, emoji_dict in results:
                if not isinstance(emoji_dict, dict) or not emoji_dict.get("base64"):
                    continue
                desc = str(emoji_dict.get("description", "")).strip()
                if not desc or desc in desc_to_emoji:
                    continue
                desc_to_emoji[desc] = emoji_dict
                ordered_descriptions.append(desc)

            if not ordered_descriptions:
                self.ctx.logger.error(
                    "[EmojiTextSelector] 未能获取任何表情包描述，放弃发送"
                )
                return {"success": False, "error": "未能获取任何表情包描述"}

            self.ctx.logger.debug(
                f"[EmojiTextSelector] 去重后 {len(ordered_descriptions)} 个表情包描述"
            )

            # 5. 调用文本 LLM 选择
            prompt = _build_selection_prompt(
                ordered_descriptions,
                conversation_context=extra_context,
                emotion_hint=emotion_hint,
            )
            llm_result = await self.ctx.llm.generate(
                prompt=prompt,
                model=self.config.selector.llm_model,
                temperature=0.7,
                max_tokens=64,
            )

            response_text = ""
            if isinstance(llm_result, dict):
                response_text = str(
                    llm_result.get("response") or llm_result.get("content") or ""
                ).strip()

            selected_idx = _parse_llm_index(response_text, len(ordered_descriptions))

            # 6. 解析 LLM 选择结果
            if selected_idx is None:
                self.ctx.logger.error(
                    f"[EmojiTextSelector] LLM 索引解析失败，放弃发送。"
                    f" LLM 返回: {response_text[:200]}"
                )
                return {"success": False, "error": "LLM 索引解析失败"}

            desc = ordered_descriptions[selected_idx - 1]
            chosen = desc_to_emoji.get(desc)
            if not chosen or not chosen.get("base64"):
                self.ctx.logger.error(
                    f"[EmojiTextSelector] 选中编号[{selected_idx}]无匹配表情包，放弃发送"
                )
                return {"success": False, "error": "选中编号无匹配表情包"}

            # 7. 发送
            emoji_base64 = chosen.get("base64", "")
            if not emoji_base64:
                return {"success": False, "error": "选中表情包的 base64 数据为空"}

            send_result = await self.ctx.send.emoji(emoji_base64, stream_id)
            if not send_result:
                return {"success": False, "error": "发送表情包失败"}

            description = chosen.get("description", "")
            self.ctx.logger.info(
                f"[EmojiTextSelector] 表情包发送成功。"
                f" 描述: {description}, 命中描述[{selected_idx}]: {desc}"
            )

            return {
                "success": True,
                "content": f"表情包发送成功（{description}）",
                "description": description,
                "selected_index": selected_idx,
            }

        except Exception as exc:
            self.ctx.logger.error(
                f"[EmojiTextSelector] 工具执行异常: {exc}", exc_info=True
            )
            return {"success": False, "error": str(exc)}

    # ─── Hook: 从 planner 工具列表里移除 send_emoji ──────────────

    @HookHandler(
        "maisaka.planner.before_request",
        mode=HookMode.BLOCKING,
    )
    async def filter_send_emoji_tool(self, **kwargs: Any) -> dict[str, Any]:
        """根据配置决定是否从 planner 工具列表里移除内置 send_emoji。"""
        if not self.config.selector.filter_send_emoji:
            return {"modified_kwargs": kwargs}
        tools = kwargs.get("tool_definitions")
        if isinstance(tools, list):
            before_count = len(tools)
            kwargs["tool_definitions"] = [
                t for t in tools
                if not (
                    isinstance(t, dict)
                    and t.get("function", {}).get("name") == "send_emoji"
                )
            ]
            after_count = len(kwargs["tool_definitions"])
            self.ctx.logger.debug(
                f"[EmojiTextSelector] filter_send_emoji: "
                f"{before_count} → {after_count} tools"
            )
        return {"modified_kwargs": kwargs}


def create_plugin() -> EmojiTextSelectorPlugin:
    """插件工厂函数，由 SDK Runner 调用以创建插件实例。"""
    return EmojiTextSelectorPlugin()
