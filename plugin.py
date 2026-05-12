"""纯文本情绪标签匹配的表情包选择插件。

通过 ``@Tool("select_emoji")`` 注册一个 deferred tool 到 planner。
LLM 通过 ``tool_search`` 发现后调用，插件会：
1. 获取全量情绪标签
2. 调用文本 LLM 选出最匹配的 1-5 个标签
3. 按幂集降级匹配表情包
4. 发送选中的表情包
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal

from maibot_sdk import MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType
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


class EmojiTextSelectorConfig(PluginConfigBase):
    """表情包文本选择器配置。"""

    __ui_label__ = "表情包文本选择器"

    plugin: PluginSectionConfig = Field(
        default_factory=PluginSectionConfig,
    )
    max_emotion_tags: int = Field(
        default=80,
        description="传给 LLM 的最大情绪标签数量，0 表示不限制",
        json_schema_extra={"label": "最大情绪标签数"},
    )
    max_selected_tags: int = Field(
        default=5,
        description="LLM 最多选择的标签数",
        json_schema_extra={"label": "最多选择标签数"},
    )
    llm_model: Literal["emoji", "utils", "planner", "reply"] = Field(
        default="emoji",
        description="标签选择用的模型任务名",
        json_schema_extra={"label": "LLM 模型"},
    )


# ─── 纯逻辑函数 ─────────────────────────────────────────────────


def _parse_llm_tags(response_text: str, max_tags: int = 5) -> list[str]:
    """从 LLM 返回的 JSON 中解析标签列表，按优先级排列。解析失败返回空列表。"""
    text = (response_text or "").strip()
    if not text:
        return []

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
            if isinstance(data, dict) and "tags" in data:
                raw_tags = data["tags"]
                if isinstance(raw_tags, list):
                    tags = [str(t).strip() for t in raw_tags if str(t).strip()]
                    return tags[:max_tags]
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    return []


def _build_selection_prompt(emotions: list[str], context_hint: str = "") -> str:
    """构建发送给文本 LLM 的标签选择 prompt。"""
    tags_text = "、".join(emotions)
    prompt = (
        "你是一个表情包选择助手。根据上下文语气和情绪，从可用标签中选择最合适的标签。\n\n"
        f"可用情绪标签（{len(emotions)} 个）：\n{tags_text}\n\n"
        "请返回一个 JSON 对象，不要输出 JSON 之外的内容。\n"
        '格式：{"tags": ["标签1", "标签2"], "reason": "选择理由"}\n'
        "标签按重要性从高到低排列，最多选 5 个，最少选 1 个。"
    )
    if context_hint:
        prompt += f"\n\n当前上下文提示：{context_hint}"
    return prompt


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
    )
    async def handle_select_emoji(
        self,
        stream_id: str = "",
        emotion_hint: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """处理 select_emoji 工具调用。

        1. 获取全量情绪标签（纯文本，数据量小）
        2. 调用文本 LLM 选出最匹配的 1-5 个标签
        3. 按优先级逐个调用 get_by_description 查询（host 端相似度匹配，
           每次只传回一张表情包，避免全量 base64 传输）
        4. 全 miss 则随机兜底
        """
        del kwargs

        try:
            # 1. 获取全量情绪标签
            emotions: list[str] = await self.ctx.emoji.get_emotions()
            if not emotions:
                return {"success": False, "error": "表情包库中没有可用标签"}

            max_tags = self.config.max_emotion_tags
            if max_tags > 0 and len(emotions) > max_tags:
                emotions = emotions[:max_tags]

            # 2. 调用文本 LLM 选择标签
            prompt = _build_selection_prompt(emotions, emotion_hint)
            llm_result = await self.ctx.llm.generate(
                prompt=prompt,
                model=self.config.llm_model,
                temperature=0.7,
                max_tokens=256,
            )

            response_text = ""
            if isinstance(llm_result, dict):
                response_text = str(
                    llm_result.get("response") or llm_result.get("content") or ""
                ).strip()

            selected_tags = _parse_llm_tags(response_text, self.config.max_selected_tags)
            if not selected_tags:
                self.ctx.logger.warning(
                    f"[EmojiTextSelector] LLM 标签解析失败，降级为随机选择。"
                    f" LLM 返回: {response_text[:200]}"
                )

            # 3. 按优先级逐个查询表情包
            chosen: dict[str, str] | None = None
            matched_tag_info = "随机选择"

            for tag in selected_tags:
                result = await self.ctx.emoji.get_by_description(tag)
                if isinstance(result, dict) and result.get("success"):
                    emoji_data = result.get("emoji")
                    if emoji_data:
                        chosen = emoji_data
                        matched_tag_info = f"命中标签: {tag}"
                        break

            # 4. 未命中则随机兜底
            if chosen is None:
                self.ctx.logger.info("[EmojiTextSelector] 标签匹配失败，降级为随机选择")
                random_result = await self.ctx.emoji.get_random(count=1)
                random_emojis = (
                    random_result.get("emojis", [])
                    if isinstance(random_result, dict)
                    else []
                )
                if not random_emojis:
                    return {"success": False, "error": "表情包库为空"}
                chosen = random_emojis[0]

            # 5. 发送
            emoji_base64 = chosen.get("base64", "")
            if not emoji_base64:
                return {"success": False, "error": "选中表情包的 base64 数据为空"}

            send_result = await self.ctx.send.emoji(emoji_base64, stream_id)
            if not send_result:
                return {"success": False, "error": "发送表情包失败"}

            description = chosen.get("description", "")
            self.ctx.logger.info(
                f"[EmojiTextSelector] 表情包发送成功。"
                f" 描述: {description}, {matched_tag_info}"
            )

            return {
                "success": True,
                "content": f"表情包发送成功（{description}）",
                "description": description,
                "selected_tags": selected_tags,
                "matched_tag_info": matched_tag_info,
            }

        except Exception as exc:
            self.ctx.logger.error(
                f"[EmojiTextSelector] 工具执行异常: {exc}", exc_info=True
            )
            return {"success": False, "error": str(exc)}


def create_plugin() -> EmojiTextSelectorPlugin:
    """插件工厂函数，由 SDK Runner 调用以创建插件实例。"""
    return EmojiTextSelectorPlugin()
