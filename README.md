# emoji_text_selection —— 纯文本表情选择插件

> 绕过内置 VLM 表情选择，用纯文本模型完成表情包选择与发送。

## 概述

Maisaka 内置的 `send_emoji` 需要视觉模型看图选表情。这个插件注册一个 `send_emoji_text` 工具，用 Levenshtein 文本匹配替代 VLM，完成同样的表情发送流程。纯文本模型也能用，无需额外配置视觉模型。

---

## 机制

### 为什么需要这个插件？

- 内置 `send_emoji` 工具在选择表情时，会把候选表情拼成一张网格图发给 VLM。如果没有视觉模型，没办法发表情。
- 插件的 `cap.emoji.get_by_description` 支持用文本描述匹配表情（基于编辑距离），但它无法替换内置选择逻辑——因为选择回调写死在 `send_emoji` 中。
- 所以插件走了一条**偷梁换柱**的路：Hook 拦截发给 LLM 的工具列表，把 `send_emoji` 换成 `send_emoji_text`，LLM 只看到文本参数，不会尝试看图，最终由插件 handler 完成匹配和发送。
- 最关键的原因是我不想花vlm的钱()
  
### 三个关键步骤（流程）

```
消息到达 -> Timing Gate -> 构建工具列表
                        |
                [Hook 拦截：移除 send_emoji，注入 send_emoji_text]
                        |
                Planner 发给 LLM 的工具定义（只含 emotion）
                        |
                LLM 调用 send_emoji_text(emotion="开心")
                        |
                路由到插件 handler
                        |
            +------------------------------------------+
            | 1. 解析 emotion 参数                     |
            | 2. 调 get_emoji_for_emotion()            |
            |    -> Levenshtein 匹配表情描述            |
            | 3. 读取表情文件 -> base64                 |
            | 4. 调 cap.send.emoji 发送                |
            +------------------------------------------+
```

### 为什么插件文件里有 @Tool 装饰器，但 Hook 又手动注入一个定义？

这是插件里最巧妙的点，也是最初出 bug 的根因。

- `@Tool("send_emoji_text", ...)` 是 SDK 的标准玩法：它会把工具注册到 MaiBot 的工具发现系统（`ToolRegistry`）。MaiBot 内部对**插件工具**会自动注入一个隐藏参数 `stream_id`。
- 如果直接把这个含 `stream_id` 的完整定义原样发给 LLM，LLM 可能回填字面值 `"stream_id": "当前聊天流ID"`（或者直接报错）。这会导致：LLM 以为调用了工具，但参数不对，表情发不出去。
- 因此 Hook 做了两件事：
  1. **同时过滤掉** `send_emoji` 和 `send_emoji_text`（避免重复定义）。
  2. **手动构造一个不含 `stream_id` 的纯净工具定义**塞回 `tool_definitions` 列表。
- 实际执行时，`ToolRegistry.invoke()` 会按 Provider 遍历，先走 `maisaka_builtin`，再走 `plugin_runtime`。插件工具名在 `plugin_runtime` 中，匹配到 `send_emoji_text` 后，框架通过 RPC 调用插件的 handler，此时会**自动注入**正确的 `stream_id`。LLM 从头到尾都不知道 `stream_id` 的存在。

简单总结：

> `@Tool` 负责让 MaiBot 知道"插件的 `send_emoji_text` 存在并且怎么调用"。  
> Hook 手动注入的定义负责"让 LLM 看到一个干净、不含内部参数的接口"。

如果理解不了，务必记住：**`@Tool` 只是注册入口，真正发给 LLM 的定义是 Hook 里手写的那份。**

---

## 文件结构

```
plugins/emoji_text_selection/
├── _manifest.json         # 插件元数据 (id, 版本, 依赖能力, 兼容范围)
├── plugin.py              # 插件主体 (Hook + Tool handler)
├── config.toml            # 插件配置 (目前仅 enable 开关)
├── i18n/
│   ├── zh-CN.json         # 中文翻译
│   └── en-US.json         # 英文翻译
└── README.md              # 本文件
```

---

## 安装与启用

1. 将整个 `emoji_text_selection` 目录放到 MaiBot 的 `plugins/` 下。
2. 确保 `config.toml` 中 `[plugin].enabled = true`。
3. 重启 MaiBot。

**模型配置（重要）**：  
`model_config.toml` 中 `emoji` 任务对应的模型可以是纯文本模型，不需要 VLM。例如：

```toml
[model_task_config.emoji]
model_list = ["your-text-model"]
```

---

## 依赖的能力

插件通过 MaiBot 的能力代理调用以下底层能力：

- `cap.emoji.get_by_description` -> `emoji_manager.get_emoji_for_emotion()`  
  用 Levenshtein 距离对表情的 `description` 做文本匹配。
- `cap.send.emoji` -> `send_service.emoji_to_stream()`  
  发送 base64 编码的表情图片到聊天流。

这些能力在 `_manifest.json` 的 `capabilities` 字段中显式声明。

---

## Hook 详解

```python
@HookHandler("maisaka.planner.before_request")
async def on_before_request(self, **kwargs):
    tools = kwargs.get("tool_definitions", [])
    # 移除内置 send_emoji 以及插件 @Tool 注册的 send_emoji_text（后者含 stream_id）
    tools = [t for t in tools
             if t.get("function", {}).get("name") not in ("send_emoji", "send_emoji_text")]
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
                        "description": "想要表达的情绪标签，如开心、难过、震惊"
                    }
                },
                "required": ["emotion"]
            }
        }
    })
    return {"modified_kwargs": {"tool_definitions": tools}}
```

- **过滤 `send_emoji`**：彻底禁用内置 VLM 表情发送。
- **过滤 `send_emoji_text`**：避免 LLM 看到含 `stream_id` 的定义。
- **手动注入纯净定义**：只暴露 `emotion` 一个参数。
- **返回值格式**：必须是 `{"modified_kwargs": {"tool_definitions": tools}}`，而不是直接返回 `{"tool_definitions": tools}`。框架通过 `modified_kwargs` 键来更新原参数。

Hook 点 `maisaka.planner.before_request` 在工具发现**之后**、发送给 LLM **之前**触发，因此可以安全修改 `tool_definitions`。

---

## Tool Handler 详解

```python
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
    del kwargs

    # 1. 文本匹配
    result = await self.ctx.emoji.get_by_description(description=emotion)
    if not result or not result.get("success"):
        return {"success": False, "message": "表情匹配失败"}

    emoji_data = result.get("emoji")
    if not emoji_data or not emoji_data.get("base64"):
        return {"success": False, "message": "没有匹配的表情"}

    # 2. 发送
    send_result = await self.ctx.send.emoji(
        emoji_base64=emoji_data["base64"],
        stream_id=stream_id,
        sync_to_maisaka_history=True,
        maisaka_source_kind="guided_reply",
    )

    if send_result and send_result.get("success"):
        return {"success": True, "message": "表情发送成功"}
    return {"success": False, "message": "表情发送失败"}
```

- **`stream_id` 由框架在 RPC 调用时自动注入**，Handler 签名必须保留该参数，但 LLM 完全看不到。
- **`@Tool` 的 `parameters` 列表中不包含 `stream_id`**，只声明 LLM 可见的参数。框架会根据签名中的 `stream_id` 参数名自动注入。
- Handler 本身通过 `@Tool` 的参数列表向框架声明 `emotion` 是必要参数，框架会做参数校验。
- 发送时 `sync_to_maisaka_history=True` 确保消息同步到历史，`guided_reply` 标记为引导回复。

---

## 常见问题排查

### 1. LLM 调用了 send_emoji_text，但表情没发出来

**现象**：日志显示模型输出了类似如下内容，但消息未发送：

```
<DSML|tool_calls>
  <DSML|invoke name="send_emoji_text">
    <DSML|parameter name="emotion">敷衍</DSML|parameter>
    <DSML|parameter name="stream_id">当前聊天流ID</DSML|parameter>
  </DSML|invoke>
</DSML|tool_calls>
```

**原因**：LLM 看到了包含 `stream_id` 参数的工具定义，并直接回填了字面值或占位符。

**解决**：确认 Hook 中**同时过滤** `send_emoji` 和 `send_emoji_text`，然后注入只含 `emotion` 的定义。如果 `send_emoji_text` 没被过滤，LLM 会收到两份定义（一份来自 `@Tool` 注册，一份来自 Hook 手动注入），其中一份包含 `stream_id`，LLM 可能选择回填。

**检查清单**：
- [ ] `on_before_request` 过滤条件包含 `"send_emoji_text"`
- [ ] 手动注入的定义中只含 `emotion` 参数
- [ ] 返回值使用了 `{"modified_kwargs": {"tool_definitions": tools}}` 格式

### 2. LLM 报了 "I cannot select emojis" 或类似错误

**原因**：`send_emoji` 没被过滤干净，LLM 看到了内置工具但没法调用（没有 VLM）。

**解决**：检查 Hook 中是否确实移除了函数名为 `send_emoji` 的工具。

### 3. 表情匹配失败（返回 "表情匹配失败"）

**原因**：输入的 `emotion` 和现有表情的 `description` 之间 Levenshtein 距离都太大，没有匹配项。

**解决**：
- 确保表情库中有对应描述的表情。可登录管理后台查看/编辑表情描述。
- 尝试使用更常见的情绪词（如"开心""哭"），避免生僻词。

### 4. 表情匹配不准，常选到错误的表情

这是**纯文本匹配方案的固有局限**。`get_emoji_for_emotion()` 基于编辑距离排序，不涉及语义理解，无法区分"微笑"和"假笑"这种语义相近但实质不同的标签。

如果对精度要求极高，建议配置视觉模型使用内置 `send_emoji`。

### 5. Planner 提示词里看不到 send_emoji_text

**原因**：Hook 返回值格式不对。必须是 `{"modified_kwargs": {"tool_definitions": tools}}`，直接返回 `{"tool_definitions": tools}` 不会被框架合并到 Planner 的参数中。

---

## Hook 调试技巧

在 `plugin.py` 的 `on_before_request` 开头加上日志可以观察 Hook 触发情况：

```python
import logging
logger = logging.getLogger(__name__)
# 在 Hook 中：
names = [t.get("function", {}).get("name") for t in tools]
logger.info(f"Hook triggered, tools before: {names}")
```

Handler 中也可以加日志验证 `stream_id` 是否为有效值：

```python
self.logger.info(f"Handler called with emotion={emotion!r}, stream_id={stream_id!r}")
```

---

## 与内置 VLM 方案对比

| 维度 | 内置 VLM (send_emoji) | 本插件 (send_emoji_text) |
|------|-----------------------|---------------------------|
| 选择方式 | VLM 看图选序号 | Levenshtein 文本匹配 |
| 精确度 | 高（语义理解图片） | 中（依赖标签覆盖度） |
| 模型要求 | 必须视觉模型 | 纯文本模型即可 |
| 表情打标签 | VLM 自动生成 | 依赖已有 description |
| 子代理 | 有（拼图+选图） | 无（直接匹配） |
| 速度 | 较慢（需拼图+调用VLM） | 快（仅文本匹配） |

---

## 兼容性

- MaiBot >= 1.0.0，< 1.99.99
- SDK >= 2.0.0，< 2.99.99
- 声明依赖能力：`emoji.get_by_description`、`send.emoji`

---

## 参考文档

- MaiBot 插件开发指南：https://github.com/Mai-with-u/maibot-plugin-sdk/blob/main/docs/guide.md
- 插件仓库提交规范：https://github.com/Mai-with-u/plugin-repo/blob/main/CONTRIBUTING.md
- 项目架构说明：`docs/plugin-emoji-text-selection.md`（与本 README 互补）

---

希望这份文档能帮下一个会话的 AI 或人类快速上手，少踩坑。
有问题欢迎提 Issue 或直接在插件代码中加注释。
