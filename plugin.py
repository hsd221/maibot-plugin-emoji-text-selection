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
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

import numpy as np
from maibot_sdk import HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import HookMode, ToolParameterInfo, ToolParamType
from pydantic import Field

logger = logging.getLogger("emoji_text_selector")

# 向量缓存持久化文件名
_VECTOR_CACHE_FILE = "emoji_vector_cache.npz"
_VECTOR_CACHE_META_FILE = "emoji_vector_cache_meta.json"


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
        default=True,
        description="启用后从 planner 工具列表移除内置 send_emoji，避免 LLM 绕过本插件直接发送",
        json_schema_extra={"label": "过滤原生 send_emoji"},
    )
    tool_discovery: Literal["始终发现", "按需发现"] = Field(
        default="始终发现",
        description="始终发现：每次对话都提供 select_emoji 工具；按需发现：LLM 需要时通过 tool_search 自行搜索",
        json_schema_extra={"label": "工具发现模式"},
    )


class SemanticSectionConfig(PluginConfigBase):
    """语义向量匹配配置。"""

    __ui_label__ = "语义匹配"
    __ui_icon__ = "search"
    __ui_order__ = 2

    enabled: bool = Field(
        default=False,
        description="启用后优先使用 embedding 向量匹配选择表情包，失败时降级为文本 LLM 选择",
        json_schema_extra={"label": "启用语义匹配"},
    )
    refresh_interval_seconds: int = Field(
        default=300,
        description="向量缓存刷新间隔（秒）",
        json_schema_extra={"label": "缓存刷新间隔"},
    )
    similarity_threshold: float = Field(
        default=0.3,
        description="最低余弦相似度阈值，低于此值的表情包不会被选中",
        json_schema_extra={"label": "相似度阈值"},
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
    semantic: SemanticSectionConfig = Field(
        default_factory=SemanticSectionConfig,
    )


# ─── 提示词模板 ───────────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent
_SELECTION_PROMPT_PATH = _PROMPT_DIR / "select_emoji.prompt"

_FALLBACK_SELECTION_PROMPT = """\
阅读以下对话上下文和当前想表达的情感，从{emoji_count}个表情包描述中选择最匹配的一个：

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
    emotion_expression: str = "",
) -> str:
    """构建发给文本 LLM 的表情包选择 prompt。从 .prompt 文件加载模板。"""
    template = _load_prompt_template()
    try:
        return template.format(
            emoji_count=str(len(descriptions)),
            conversation_context=conversation_context,
            emotion_hint_block=(
                f"当前想表达的情感：{emotion_expression}"
                if emotion_expression else ""
            ),
            description_list="\n".join(
                f"{i+1}. {d}" for i, d in enumerate(descriptions)
            ),
        )
    except KeyError:
        logger.warning(
            "[EmojiTextSelector] 自定义 prompt 模板包含未知变量，回退到内置模板"
        )
        return _FALLBACK_SELECTION_PROMPT.format(
            emoji_count=str(len(descriptions)),
            conversation_context=conversation_context,
            emotion_hint_block=(
                f"当前想表达的情感：{emotion_expression}"
                if emotion_expression else ""
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
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    idx = int(raw)
                    if 1 <= idx <= max_count:
                        return idx
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            continue

    return None


# ─── Embedding 向量缓存 ────────────────────────────────────────


class EmojiEmbeddingCache:
    """表情包描述 embedding 向量缓存，使用 numpy 矩阵加速检索。"""

    def __init__(self) -> None:
        self._ids: np.ndarray = np.array([], dtype=np.int64)
        self._text_keys: Dict[int, str] = {}
        self._emotion_tags: Dict[int, str] = {}
        self._tag_to_desc: Dict[str, str] = {}
        self._matrix: Optional[np.ndarray] = None
        self._last_refresh_time: float = 0.0
        self._refreshing: bool = False

    @property
    def is_empty(self) -> bool:
        return len(self._ids) == 0

    @property
    def count(self) -> int:
        return len(self._ids)

    def needs_refresh(self, interval_seconds: int) -> bool:
        if self._refreshing:
            return False
        return (time.time() - self._last_refresh_time) > interval_seconds

    def rebuild(
        self,
        ids: List[int],
        text_keys: Dict[int, str],
        emotion_tags: Dict[int, str],
        matrix: np.ndarray,
    ) -> None:
        """用预处理好的数据重建缓存。"""
        if len(ids) == 0:
            self._ids = np.array([], dtype=np.int64)
            self._text_keys = {}
            self._emotion_tags = {}
            self._tag_to_desc = {}
            self._matrix = None
            return

        self._ids = np.array(ids, dtype=np.int64)
        self._text_keys = dict(text_keys)
        self._emotion_tags = dict(emotion_tags)
        # 预计算 tag → description 映射，同一 description 对应多个 tag 时只保留第一个
        tag_to_desc: Dict[str, str] = {}
        seen_descs: set[str] = set()
        for cid in ids:
            tag = emotion_tags.get(cid)
            desc = text_keys.get(cid)
            if tag and desc and desc not in seen_descs:
                seen_descs.add(desc)
                tag_to_desc[tag] = desc
        self._tag_to_desc = tag_to_desc
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self._matrix = (matrix / norms).astype(np.float32)

    def get_text_key(self, expr_id: int) -> Optional[str]:
        return self._text_keys.get(expr_id)

    def get_emotion_tag(self, expr_id: int) -> Optional[str]:
        return self._emotion_tags.get(expr_id)

    def get_tag_to_id_map(self) -> Dict[str, int]:
        """返回 tag → cache_id 的反向映射，用于跨刷新周期保持同一 tag 的稳定 id。"""
        return {v: int(k) for k, v in self._emotion_tags.items()}

    def get_tag_description_map(self) -> Dict[str, str]:
        """返回去重后的 tag → description 映射，结果在 rebuild 时预计算。"""
        return dict(self._tag_to_desc)

    def get_existing_entries(
        self, valid_ids: set[int], changed_ids: set[int]
    ) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[int, str]]:
        """提取未变更的已有条目，返回 (ids, matrix_rows, text_keys, emotion_tags)。"""
        if self._matrix is None or len(self._ids) == 0:
            return [], np.empty((0, 0), dtype=np.float32), {}, {}

        mask = np.array([
            (int(eid) in valid_ids and int(eid) not in changed_ids)
            for eid in self._ids
        ], dtype=bool)

        kept_ids = self._ids[mask].tolist()
        kept_matrix = self._matrix[mask]
        kept_text_keys = {eid: self._text_keys[eid] for eid in kept_ids if eid in self._text_keys}
        kept_emotion_tags = {eid: self._emotion_tags[eid] for eid in kept_ids if eid in self._emotion_tags}
        return kept_ids, kept_matrix, kept_text_keys, kept_emotion_tags

    def mark_refreshed(self) -> None:
        self._last_refresh_time = time.time()
        self._refreshing = False

    def set_refreshing(self) -> None:
        self._refreshing = True

    def search(
        self,
        query_vector: np.ndarray,
        threshold: float,
        max_count: int,
    ) -> List[Tuple[int, float]]:
        """向量检索，返回 (id, score) 列表。"""
        if self._matrix is None or len(self._ids) == 0:
            return []

        scores = self._matrix @ query_vector

        above_threshold = scores >= threshold
        if not above_threshold.any():
            return []

        filtered_scores = scores[above_threshold]
        filtered_ids = self._ids[above_threshold]

        if len(filtered_scores) <= max_count:
            top_indices = np.argsort(-filtered_scores)
        else:
            top_indices = np.argpartition(-filtered_scores, max_count)[:max_count]
            top_indices = top_indices[np.argsort(-filtered_scores[top_indices])]

        return [(int(filtered_ids[i]), float(filtered_scores[i])) for i in top_indices]

    def save_to_disk(self, cache_dir: Path) -> None:
        """将向量缓存持久化到磁盘。"""
        if self._matrix is None or len(self._ids) == 0:
            return

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_dir / _VECTOR_CACHE_FILE,
                ids=self._ids,
                matrix=self._matrix,
            )
            meta = {
                "text_keys": {str(k): v for k, v in self._text_keys.items()},
                "emotion_tags": {str(k): v for k, v in self._emotion_tags.items()},
            }
            (cache_dir / _VECTOR_CACHE_META_FILE).write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(f"向量缓存持久化失败: {exc}")

    def load_from_disk(self, cache_dir: Path) -> bool:
        """从磁盘加载向量缓存，成功返回 True。"""
        npz_path = cache_dir / _VECTOR_CACHE_FILE
        meta_path = cache_dir / _VECTOR_CACHE_META_FILE

        if not npz_path.exists() or not meta_path.exists():
            return False

        try:
            with np.load(npz_path) as data:
                self._ids = data["ids"].astype(np.int64)
                self._matrix = data["matrix"].astype(np.float32)

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._text_keys = {int(k): v for k, v in meta.get("text_keys", {}).items()}
            self._emotion_tags = {int(k): v for k, v in meta.get("emotion_tags", {}).items()}
            # 恢复预计算的 tag → description 映射
            tag_to_desc: Dict[str, str] = {}
            seen_descs: set[str] = set()
            for cid, tag in self._emotion_tags.items():
                desc = self._text_keys.get(cid)
                if desc and desc not in seen_descs:
                    seen_descs.add(desc)
                    tag_to_desc[tag] = desc
            self._tag_to_desc = tag_to_desc
            self._last_refresh_time = time.time()
            return True
        except Exception as exc:
            logger.warning(f"从磁盘加载向量缓存失败: {exc}")
            return False


# ─── 插件主类 ───────────────────────────────────────────────────


class EmojiTextSelectorPlugin(MaiBotPlugin):
    """表情包选择插件，支持语义向量匹配 + 文本 LLM 选择两级策略。"""

    config_model: ClassVar[type[PluginConfigBase] | None] = EmojiTextSelectorConfig

    def __init__(self) -> None:
        super().__init__()
        self._cache = EmojiEmbeddingCache()
        self._refresh_task: Optional[asyncio.Task] = None
        self._plugin_dir: Optional[Path] = None

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

    def get_components(self) -> list[dict[str, Any]]:
        components = super().get_components()
        for comp in components:
            meta = comp.get("metadata")
            if not isinstance(meta, dict):
                continue
            inner = meta.get("metadata")
            if isinstance(inner, dict) and "core_tool" in inner:
                meta["core_tool"] = inner["core_tool"]
        return components

    # ─── 生命周期 ────────────────────────────────────────────

    async def on_load(self) -> None:
        self._plugin_dir = Path(__file__).parent
        cache_dir = self._plugin_dir / ".cache"

        if self._cache.load_from_disk(cache_dir):
            logger.info(f"[EmojiTextSelector] 从磁盘恢复向量缓存成功，共 {self._cache.count} 条")

        self._refresh_task = asyncio.create_task(self._background_refresh_loop())
        logger.info("[EmojiTextSelector] 插件已加载")

    async def on_unload(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None

        if self._plugin_dir is not None:
            self._cache.save_to_disk(self._plugin_dir / ".cache")

        logger.info("[EmojiTextSelector] 插件已卸载")

    async def on_config_update(
        self, scope: str, config_data: dict[str, Any], version: str
    ) -> None:
        if scope == "self":
            self.set_plugin_config(config_data)

    # ─── 向量缓存刷新 ────────────────────────────────────────

    async def _background_refresh_loop(self) -> None:
        """后台定时刷新向量缓存。仅在语义匹配启用时运行。"""
        await asyncio.sleep(5)
        while True:
            try:
                if not self.config.semantic.enabled:
                    await asyncio.sleep(30)
                    continue
                interval = self.config.semantic.refresh_interval_seconds
                if self._cache.needs_refresh(interval):
                    await self._refresh_cache()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[EmojiTextSelector] 向量缓存刷新失败: {exc}")
            await asyncio.sleep(30)

    async def _refresh_cache(self) -> None:
        """从表情包库加载情绪标签和描述，增量计算 embedding 向量。"""
        self._cache.set_refreshing()
        refresh_start = time.time()
        try:
            try:
                emotions: list[str] = await self.ctx.emoji.get_emotions()
            except Exception as exc:
                logger.warning(f"[EmojiTextSelector] 获取情绪标签失败: {exc}")
                return

            if not emotions:
                return

            # 构建 tag → 旧 cache_id 的反向映射，确保同一 tag 跨刷新周期使用稳定 id
            old_tag_to_id: Dict[str, int] = self._cache.get_tag_to_id_map()
            next_id = max(old_tag_to_id.values()) + 1 if old_tag_to_id else 0

            # 并发获取每个标签的代表表情包
            semaphore = asyncio.Semaphore(10)

            async def _fetch_one(tag: str) -> tuple[str, Any]:
                async with semaphore:
                    return await self._do_fetch_one_emoji(tag)

            results = await asyncio.gather(*(_fetch_one(t) for t in emotions))

            # 按 description 去重，用 tag 分配稳定的 cache_id
            seen_descs: set[str] = set()
            valid_ids: set[int] = set()
            changed_ids: set[int] = set()
            texts_to_embed: List[Tuple[int, str]] = []
            id_to_tag: Dict[int, str] = {}

            for tag, emoji_dict in results:
                if not isinstance(emoji_dict, dict) or not emoji_dict.get("description"):
                    continue
                desc = str(emoji_dict.get("description", "")).strip()
                if not desc or desc in seen_descs:
                    continue
                seen_descs.add(desc)

                cache_id = old_tag_to_id.get(tag, next_id)
                if cache_id == next_id:
                    next_id += 1
                valid_ids.add(cache_id)
                id_to_tag[cache_id] = tag

                if self._cache.get_text_key(cache_id) != desc:
                    changed_ids.add(cache_id)
                    texts_to_embed.append((cache_id, desc))

            # 保留已有条目（排除变更的，其旧向量将在后面被新向量覆盖）
            kept_ids, kept_matrix, kept_text_keys, kept_emotion_tags = (
                self._cache.get_existing_entries(valid_ids, changed_ids)
            )

            # 分批计算新增/变更的 embedding
            BATCH_SIZE = 64
            new_ids: List[int] = []
            new_vectors: List[List[float]] = []
            new_text_keys: Dict[int, str] = {}
            new_emotion_tags: Dict[int, str] = {}

            if texts_to_embed:
                for batch_start in range(0, len(texts_to_embed), BATCH_SIZE):
                    batch_items = texts_to_embed[batch_start:batch_start + BATCH_SIZE]
                    batch_texts = [text_key for _, text_key in batch_items]

                    embed_result = None
                    for attempt in range(2):
                        try:
                            embed_result = await self.ctx.llm.embed(texts=batch_texts)
                            break
                        except Exception as exc:
                            if attempt == 0:
                                logger.warning(
                                    f"[EmojiTextSelector] embedding 调用失败（第1次），"
                                    f"10s 后重试: {exc}"
                                )
                                await asyncio.sleep(10)
                            else:
                                logger.error(
                                    f"[EmojiTextSelector] embedding 调用失败（第2次），"
                                    f"跳过当前批次 ({len(batch_items)} 条): {exc}"
                                )

                    if embed_result is None:
                        continue

                    if isinstance(embed_result, dict) and embed_result.get("success"):
                        emb_results = embed_result.get("results", [])
                        if len(emb_results) < len(batch_items):
                            dropped = len(batch_items) - len(emb_results)
                            logger.warning(
                                f"[EmojiTextSelector] embedding API 返回结果不足: "
                                f"请求 {len(batch_items)} 条，仅收到 {len(emb_results)} 条，"
                                f"{dropped} 条描述将回退到旧缓存"
                            )
                        for i, (cache_id, text_key) in enumerate(batch_items):
                            if i < len(emb_results):
                                vector = emb_results[i].get("embedding", [])
                                if vector:
                                    new_ids.append(cache_id)
                                    new_vectors.append(vector)
                                    new_text_keys[cache_id] = text_key
                                    new_emotion_tags[cache_id] = id_to_tag.get(cache_id, "")
                    else:
                        logger.warning(f"[EmojiTextSelector] 批量 embedding 失败: {embed_result}")

            # 合并：new 覆盖 kept 中的同 id 条目（embedding 成功时用新向量，失败时保留旧向量）
            new_id_set = set(new_ids)
            final_ids: List[int] = []
            final_text_keys: Dict[int, str] = {}
            final_emotion_tags: Dict[int, str] = {}
            matrix_rows: List[np.ndarray] = []

            for i, cid in enumerate(kept_ids):
                if cid in new_id_set:
                    continue
                final_ids.append(cid)
                final_text_keys[cid] = kept_text_keys.get(cid, "")
                final_emotion_tags[cid] = kept_emotion_tags.get(cid, "")
                if kept_matrix.size > 0:
                    matrix_rows.append(kept_matrix[i])

            for i, cid in enumerate(new_ids):
                final_ids.append(cid)
                final_text_keys[cid] = new_text_keys.get(cid, "")
                final_emotion_tags[cid] = new_emotion_tags.get(cid, "")
                matrix_rows.append(np.array(new_vectors[i], dtype=np.float32))

            if matrix_rows:
                all_matrix = np.vstack(matrix_rows)
            else:
                all_matrix = np.empty((0, 0), dtype=np.float32)

            self._cache.rebuild(final_ids, final_text_keys, final_emotion_tags, all_matrix)

            if self._plugin_dir is not None:
                self._cache.save_to_disk(self._plugin_dir / ".cache")

            refresh_elapsed_ms = (time.time() - refresh_start) * 1000
            logger.info(
                f"[EmojiTextSelector] 向量缓存刷新完成，共 {self._cache.count} 条表情包描述，"
                f"本次新增/更新 {len(new_ids)} 条，耗时 {refresh_elapsed_ms:.0f}ms"
            )
        finally:
            self._cache.mark_refreshed()

    async def _semantic_select(
        self,
        query_text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """语义向量匹配。返回 (matched_tag, matched_description) 或 (None, None)。"""
        if not query_text.strip():
            return None, None

        try:
            embed_result = await self.ctx.llm.embed(text=query_text)
            if not isinstance(embed_result, dict) or not embed_result.get("success"):
                logger.warning(f"[EmojiTextSelector] 查询 embedding 失败: {embed_result}")
                return None, None

            raw_vector = embed_result.get("embedding", [])
            if not raw_vector:
                return None, None

            query_vector = np.array(raw_vector, dtype=np.float32)
            q_norm = np.linalg.norm(query_vector)
            if q_norm < 1e-12:
                return None, None
            query_vector = query_vector / q_norm

            threshold = self.config.semantic.similarity_threshold
            top_matches = self._cache.search(query_vector, threshold, max_count=1)
            if not top_matches:
                logger.info("[EmojiTextSelector] 语义匹配未找到超过阈值的表情包")
                return None, None

            best_id, best_score = top_matches[0]
            matched_tag = self._cache.get_emotion_tag(best_id)
            matched_desc = self._cache.get_text_key(best_id)
            logger.info(
                f"[EmojiTextSelector] 语义匹配命中: tag={matched_tag}, "
                f"desc={matched_desc}, score={best_score:.3f}"
            )
            return matched_tag, matched_desc
        except Exception as exc:
            logger.error(f"[EmojiTextSelector] 语义匹配异常: {exc}")
            return None, None

    async def _fetch_conversation_context(self, stream_id: str) -> str:
        """获取并格式化最近的对话上下文，返回 planner 风格文本。失败返回空字符串。"""
        if not stream_id:
            return ""
        try:
            messages = await self.ctx.message.get_recent(stream_id, limit=30)
            if not messages:
                return ""
            return _build_conversation_context(messages)
        except Exception as exc:
            logger.debug(
                f"[EmojiTextSelector] 获取对话上下文异常: {exc}"
            )
            return ""

    async def _do_fetch_one_emoji(self, tag: str) -> Tuple[str, Any]:
        """通过 tag 获取单个表情包数据，异常时返回 (tag, None)。"""
        try:
            return tag, await self.ctx.emoji.get_by_description(tag)
        except Exception as exc:
            logger.debug(
                f"[EmojiTextSelector] get_by_description('{tag}') 异常: {exc}"
            )
            return tag, None

    # ─── Tool: select_emoji ──────────────────────────────────

    @Tool(
        name="select_emoji",
        description="根据当前对话情绪从表情包库中选择并发送合适的表情包。",
        parameters=[
            ToolParameterInfo(
                name="emotion_hint",
                param_type=ToolParamType.STRING,
                description="你想通过表情包表达的情感或态度，例如：'对刚才的玩笑表示开心和赞同'、'表达无奈和吐槽'、'给对方鼓励和安慰'",
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

        1. 如果启用语义匹配且缓存就绪，优先使用 embedding 向量匹配
        2. 向量匹配失败则降级为文本 LLM 选择
        3. 两级都失败则返回 error

        优化：缓存就绪时直接从缓存获取 tag→description 映射，
        仅在 LLM 选中后对目标 tag 调用一次 get_by_description 获取 base64 发送。
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

            # 2. 获取表情包描述列表
            # 缓存就绪时：直接从缓存取 tag→description 映射（零次 get_by_description）
            # 缓存为空时：降级为并发调用 get_by_description 获取全部
            cache_available = not self._cache.is_empty
            desc_to_emoji: dict[str, dict[str, str]] = {}
            desc_to_tag: dict[str, str] = {}
            ordered_descriptions: list[str] = []
            extra_context = ""

            if cache_available:
                # 从向量缓存直接获取 tag → description 映射，无需调用 get_by_description
                tag_desc_map = self._cache.get_tag_description_map()
                seen_descs: set[str] = set()
                for tag in emotions:
                    desc = tag_desc_map.get(tag)
                    if desc and desc not in seen_descs:
                        seen_descs.add(desc)
                        desc_to_tag[desc] = tag
                        ordered_descriptions.append(desc)

                extra_context = await self._fetch_conversation_context(stream_id)

                logger.debug(
                    f"[EmojiTextSelector] （缓存命中）{len(ordered_descriptions)} 个表情包描述"
                )
            else:
                # 缓存为空：并发为每个标签取回表情包（原有降级逻辑）
                semaphore = asyncio.Semaphore(10)

                async def _fetch_one(tag: str) -> tuple[str, Any]:
                    async with semaphore:
                        return await self._do_fetch_one_emoji(tag)

                async def _fetch_context() -> str:
                    return await self._fetch_conversation_context(stream_id)

                results, extra_context = await asyncio.gather(
                    asyncio.gather(*(_fetch_one(t) for t in emotions)),
                    _fetch_context(),
                )

                for tag, emoji_dict in results:
                    if not isinstance(emoji_dict, dict) or not emoji_dict.get("base64"):
                        continue
                    desc = str(emoji_dict.get("description", "")).strip()
                    if not desc or desc in desc_to_emoji:
                        continue
                    desc_to_emoji[desc] = emoji_dict
                    desc_to_tag[desc] = tag
                    ordered_descriptions.append(desc)

                logger.debug(
                    f"[EmojiTextSelector] （缓存未命中）去重后 {len(ordered_descriptions)} 个表情包描述"
                )

            if not ordered_descriptions:
                logger.error(
                    "[EmojiTextSelector] 未能获取任何表情包描述，放弃发送"
                )
                return {"success": False, "error": "未能获取任何表情包描述"}

            # ── 3. 语义向量匹配（优先） ──
            if self.config.semantic.enabled and not self._cache.is_empty:
                try:
                    query_text = emotion_hint.strip() if emotion_hint else ""
                    if not query_text and extra_context:
                        # 无 emotion_hint 时用对话上下文作为查询
                        query_text = extra_context[:500]

                    if query_text:
                        matched_tag, matched_desc = await self._semantic_select(
                            query_text
                        )
                        if matched_tag and matched_desc:
                            # 缓存就绪时需通过 get_by_description 获取 base64
                            emoji_result = await self.ctx.emoji.get_by_description(matched_tag)
                            emoji_base64 = ""
                            if isinstance(emoji_result, dict):
                                emoji_base64 = str(emoji_result.get("base64") or "")
                            if emoji_base64:
                                send_result = await self.ctx.send.emoji(
                                    emoji_base64, stream_id
                                )
                                if send_result:
                                    logger.info(
                                        f"[EmojiTextSelector] 语义匹配发送成功。"
                                        f" tag={matched_tag}, desc={matched_desc}"
                                    )
                                    return {
                                        "success": True,
                                        "content": f"表情包发送成功（{matched_desc}）",
                                        "description": matched_desc,
                                        "method": "semantic",
                                    }
                            logger.error(
                                "[EmojiTextSelector] 语义匹配命中但发送失败，"
                                "将降级到文本 LLM 选择"
                            )
                except Exception as exc:
                    logger.warning(
                        f"[EmojiTextSelector] 语义匹配失败，降级为文本 LLM 选择: {exc}"
                    )

            # ── 4. 文本 LLM 选择（降级） ──
            prompt = _build_selection_prompt(
                ordered_descriptions,
                conversation_context=extra_context,
                emotion_expression=emotion_hint,
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

            # 5. 解析 LLM 选择结果
            if selected_idx is None:
                logger.error(
                    f"[EmojiTextSelector] LLM 索引解析失败，放弃发送。"
                    f" LLM 返回: {response_text[:200]}"
                )
                return {"success": False, "error": "LLM 索引解析失败"}

            selected_desc = ordered_descriptions[selected_idx - 1]

            # 6. 获取选中表情包的 base64 并发送
            if cache_available:
                # 缓存就绪时：仅对选中的 tag 调用一次 get_by_description
                selected_tag = desc_to_tag.get(selected_desc)
                if not selected_tag:
                    logger.error(
                        f"[EmojiTextSelector] 选中描述[{selected_idx}]无对应标签，放弃发送"
                    )
                    return {"success": False, "error": "选中描述无对应标签"}
                try:
                    emoji_result = await self.ctx.emoji.get_by_description(selected_tag)
                except Exception as exc:
                    logger.error(
                        f"[EmojiTextSelector] get_by_description('{selected_tag}') 异常: {exc}"
                    )
                    return {"success": False, "error": f"获取表情包失败: {exc}"}

                if not isinstance(emoji_result, dict):
                    return {"success": False, "error": "获取表情包返回数据异常"}
                emoji_base64 = str(emoji_result.get("base64") or "")
                chosen_desc = str(emoji_result.get("description") or selected_desc).strip()
            else:
                # 缓存为空时：从已缓存的 emoji_dict 中取 base64
                chosen = desc_to_emoji.get(selected_desc)
                if not chosen or not chosen.get("base64"):
                    logger.error(
                        f"[EmojiTextSelector] 选中编号[{selected_idx}]无匹配表情包，放弃发送"
                    )
                    return {"success": False, "error": "选中编号无匹配表情包"}
                emoji_base64 = chosen.get("base64", "")
                chosen_desc = chosen.get("description", "")

            if not emoji_base64:
                return {"success": False, "error": "选中表情包的 base64 数据为空"}

            # 7. 发送
            send_result = await self.ctx.send.emoji(emoji_base64, stream_id)
            if not send_result:
                logger.error(
                    f"[EmojiTextSelector] 发送表情包失败。"
                    f" description={chosen_desc}"
                )
                return {"success": False, "error": "发送表情包失败"}

            logger.info(
                f"[EmojiTextSelector] 文本 LLM 发送成功。"
                f" 描述: {chosen_desc}, 命中描述[{selected_idx}]: {selected_desc}"
            )

            return {
                "success": True,
                "content": f"表情包发送成功（{chosen_desc}）",
                "description": chosen_desc,
                "selected_index": selected_idx,
                "method": "text_llm",
            }

        except Exception as exc:
            logger.error(
                f"[EmojiTextSelector] 工具执行异常: {exc}", exc_info=True
            )
            return {"success": False, "error": str(exc)}

    # ─── Hook: 从 planner 工具列表里移除 send_emoji ──────────────

    @HookHandler(
        "maisaka.planner.before_request",
        mode=HookMode.BLOCKING,
    )
    async def filter_send_emoji_tool(self, **kwargs: Any) -> dict[str, Any]:
        """根据配置决定是否从 planner 工具列表中移除 send_emoji / select_emoji。"""
        tools = kwargs.get("tool_definitions")
        if not isinstance(tools, list):
            return {"modified_kwargs": kwargs}

        names_to_remove: set[str] = set()

        try:
            filter_send = self.config.selector.filter_send_emoji
            discovery_mode = self.config.selector.tool_discovery
        except RuntimeError:
            logger.warning(
                "[EmojiTextSelector] 配置未注入，跳过工具过滤"
            )
            return {"modified_kwargs": kwargs}

        if filter_send:
            names_to_remove.add("send_emoji")

        if discovery_mode == "按需发现":
            names_to_remove.add("select_emoji")

        if not names_to_remove:
            return {"modified_kwargs": kwargs}

        before_count = len(tools)
        filtered_tools = [
            t for t in tools
            if not (
                isinstance(t, dict)
                and t.get("function", {}).get("name") in names_to_remove
            )
        ]
        after_count = len(filtered_tools)
        logger.info(
            f"[EmojiTextSelector] 工具过滤: {before_count} → {after_count}, "
            f"移除: {names_to_remove}, filter_send_emoji={filter_send}, "
            f"tool_discovery={discovery_mode}"
        )
        return {"modified_kwargs": {**kwargs, "tool_definitions": filtered_tools}}


def create_plugin() -> EmojiTextSelectorPlugin:
    """插件工厂函数，由 SDK Runner 调用以创建插件实例。"""
    return EmojiTextSelectorPlugin()
