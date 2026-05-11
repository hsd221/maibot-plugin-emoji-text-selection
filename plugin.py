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
import random
from itertools import combinations
from typing import Any, ClassVar

from maibot_sdk import MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType
from pydantic import Field


# ─── 配置模型 ───────────────────────────────────────────────────


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置节，对应 config.toml 中的 [plugin] 节。"""

    config_version: str = Field(
        default="1.0.0",
        description="配置版本号",
    )


class EmojiTextSelectorConfig(PluginConfigBase):
    """插件配置。"""

    plugin: PluginSectionConfig = Field(
        default_factory=PluginSectionConfig,
        description="插件基础配置",
    )
    max_emotion_tags: int = Field(
        default=80,
        description="传给 LLM 的最大情绪标签数量，0 表示不限制",
    )
    max_selected_tags: int = Field(
        default=5,
        description="LLM 最多选择的标签数",
    )
    llm_model: str = Field(
        default="",
        description="标签选择用的模型任务名，空字符串表示使用默认 text 模型",
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


def _split_tags(description: str) -> list[str]:
    """将逗号分隔的描述文本拆分为去重标签列表。"""
    if not description or not description.strip():
        return []
    items = [
        item.strip()
        for item in description.replace("，", ",").replace("、", ",").split(",")
    ]
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _emoji_has_all_tags(emoji_tags: list[str], target_tags: list[str]) -> bool:
    """表情包标签是否包含全部目标标签（精确匹配）。"""
    return all(tag in emoji_tags for tag in target_tags)


def _match_emojis_by_tags(
    emojis: list[dict[str, str]],
    selected_tags: list[str],
) -> list[dict[str, str]] | None:
    """按幂集降级匹配表情包。

    selected_tags 按优先级从高到低排列。
    优先匹配全部标签的交集，逐步降级到单标签匹配。
    """
    if not selected_tags or not emojis:
        return None

    emoji_tag_cache: list[tuple[dict[str, str], list[str]]] = [
        (emoji, _split_tags(emoji.get("description", "")))
        for emoji in emojis
    ]

    n = len(selected_tags)
    for size in range(n, 0, -1):
        subsets = _generate_tag_subsets(selected_tags, size)
        for tag_subset in subsets:
            matched = [
                emoji
                for emoji, tags in emoji_tag_cache
                if _emoji_has_all_tags(tags, tag_subset)
            ]
            if matched:
                return matched

    return None


def _generate_tag_subsets(
    selected_tags: list[str], size: int
) -> list[list[str]]:
    """生成指定大小的标签子集，优先保留高优先级标签。"""
    n = len(selected_tags)
    if size >= n:
        return [list(selected_tags)]

    subsets: list[list[str]] = []
    seen: set[str] = set()

    if size == 1:
        for tag in selected_tags:
            if tag not in seen:
                seen.add(tag)
                subsets.append([tag])
        return subsets

    # 包含第一个（最高优先级）标签的组合
    for combo in combinations(range(1, n), size - 1):
        indices = (0,) + combo
        subset = [selected_tags[i] for i in indices]
        key = ",".join(subset)
        if key not in seen:
            seen.add(key)
            subsets.append(subset)

    # 不包含第一个标签的组合
    for combo in combinations(range(1, n), size):
        subset = [selected_tags[i] for i in combo]
        key = ",".join(subset)
        if key not in seen:
            seen.add(key)
            subsets.append(subset)

    return subsets


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

        Host 自动注入 stream_id。插件从 Host 获取全量情绪标签和表情包列表，
        调用文本 LLM 选择标签，按幂集降级匹配后发送。
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

            # 3. 获取全量表情包
            all_emojis: list[dict[str, str]] = await self.ctx.emoji.get_all()
            if not all_emojis:
                return {"success": False, "error": "表情包库为空"}

            # 4. 匹配
            if selected_tags:
                matched = _match_emojis_by_tags(all_emojis, selected_tags)
            else:
                matched = None

            if not matched:
                # 降级：随机选一个
                chosen = random.choice(all_emojis)
                self.ctx.logger.info(
                    "[EmojiTextSelector] 标签匹配失败，使用随机表情包"
                )
            else:
                chosen = random.choice(matched)

            # 5. 发送
            emoji_base64 = chosen.get("base64", "")
            if not emoji_base64:
                return {"success": False, "error": "选中表情包的 base64 数据为空"}

            send_result = await self.ctx.send.emoji(emoji_base64, stream_id)
            if not send_result:
                return {"success": False, "error": "发送表情包失败"}

            description = chosen.get("description", "")
            matched_tag_info = (
                f"命中标签: {selected_tags}" if selected_tags else "随机选择"
            )
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
