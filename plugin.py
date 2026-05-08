from maibot_sdk import HookHandler, MaiBotPlugin, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType


class EmojiTextSelectionPlugin(MaiBotPlugin):
    """纯文本表情选择插件 —— 绕过内置 VLM 表情选择，用文本匹配发送表情。"""

    async def on_load(self) -> None:
        return None

    async def on_unload(self) -> None:
        return None

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        del scope, config_data, version

    @HookHandler("maisaka.planner.before_request")
    async def on_before_request(self, **kwargs):
        """在 Planner 请求前偷梁换柱：移除内置 send_emoji，注入 send_emoji_text。"""
        tools = kwargs.get("tool_definitions", [])
        # 移除内置 send_emoji 以及插件 @Tool 注册的 send_emoji_text（后者含 stream_id，会误导 LLM 回填字面值）
        tools = [t for t in tools if t.get("function", {}).get("name") not in ("send_emoji", "send_emoji_text")]
        # 注入纯文本版工具定义
        tools.append({
            "type": "function",
            "function": {
                "name": "send_emoji_text",
                "description": "选择一个合适的表情包并发送",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emotion": {
                            "type": "string",
                            "description": "想要表达的情绪标签，如开心、难过、震惊",
                        },
                    },
                    "required": ["emotion"],
                },
            },
        })
        return {"modified_kwargs": {"tool_definitions": tools}}

    @Tool(
        "send_emoji_text",
        brief_description="选择一个合适的表情包并发送",
        detailed_description="根据输入的情绪标签，从表情库中匹配最合适的表情包并发送。",
        parameters=[
            ToolParameterInfo(
                name="emotion",
                param_type=ToolParamType.STRING,
                description="想要表达的情绪标签，如开心、难过、震惊",
                required=True,
            ),
        ],
    )
    async def handle_send_emoji_text(self, emotion: str, stream_id: str, **kwargs):
        """处理 send_emoji_text 工具调用：文本匹配表情 → 发送。"""
        del kwargs

        # 1. 通过能力代理进行文本匹配表情
        result = await self.ctx.emoji.get_by_description(description=emotion)
        if not result or not result.get("success"):
            return {"success": False, "message": "表情匹配失败"}

        emoji_data = result.get("emoji")
        if not emoji_data or not emoji_data.get("base64"):
            return {"success": False, "message": "没有匹配的表情"}

        # 2. 发送表情
        send_result = await self.ctx.send.emoji(
            emoji_base64=emoji_data["base64"],
            stream_id=stream_id,
            sync_to_maisaka_history=True,
            maisaka_source_kind="guided_reply",
        )

        if send_result and send_result.get("success"):
            return {"success": True, "message": "表情发送成功"}
        return {"success": False, "message": "表情发送失败"}


def create_plugin():
    return EmojiTextSelectionPlugin()
