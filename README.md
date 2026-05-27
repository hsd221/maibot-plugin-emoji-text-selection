# Emoji Text Selector —— 智能表情包选择插件

让 MaiBot 在没有视觉模型的情况下，通过语义向量匹配或文本 LLM 完成表情包选择与发送。

---

## 这是什么？

MaiBot 内置的 `send_emoji` 工具通过 VLM 看图选表情，未配置视觉模型时会报错。

本插件提供两级选择策略：
1. **语义向量匹配**（可选）—— 将表情包描述和当前语境分别 embedding，余弦相似度匹配最贴合的表情包
2. **文本 LLM 选择** —— 将全量情绪标签 + 聊天上下文交给文本 LLM 选出最匹配的表情包

两级均失败则返回错误，主程序原生 `send_emoji`（VLM）可接管。

---

## 安装

### 方式一：从插件中心安装（推荐）

在 MaiBot WebUI 的插件市场中搜索 `emoji-text-selector`，点击安装。

### 方式二：手动安装

```bash
cd /path/to/MaiBot/plugins
git clone https://github.com/hsd221/maibot-plugin-emoji-text-selection.git emoji_text_selector
# 重启 MaiBot
```

---

## 配置

```toml
[plugin]
enabled = true
config_version = "1.0.0"

[selector]
max_emotion_tags = 50        # 传给 LLM 的最大情绪标签数量
max_selected_tags = 5
llm_model = "emoji"          # 文本选择用的模型任务名
filter_send_emoji = true     # 过滤原生 send_emoji，避免 LLM 绕过本插件
tool_discovery = "始终发现"

[semantic]
enabled = false              # 启用语义向量匹配（需配置 embedding 模型）
refresh_interval_seconds = 300
similarity_threshold = 0.3
```

---

## 工作原理

1. Planner 调用 `select_emoji` 工具，传入想表达的情感（`emotion_hint`）
2. **语义匹配优先**（若启用）：将情感描述 embedding，与预缓存的表情包描述向量做余弦相似度检索
3. **文本 LLM 降级**：语义匹配失败则走文本 LLM 从标签列表中选出最匹配的描述
4. 发送选中的表情包

---

## 和内置表情功能的区别

|  | 内置 VLM 表情 | 本插件（文本 LLM） | 本插件（语义匹配） |
|------|-------|------|------|
| 选择方式 | 视觉模型拼图选 | 文本标签匹配 | embedding 向量匹配 |
| 模型要求 | 必须 VLM | 纯文本模型 | embedding + 文本模型 |
| 精确度 | 高 | 中 | 中-高（依赖描述覆盖度） |
| 速度 | 慢（拼图 + VLM） | 快 | 最快（纯向量检索） |

---

## 致谢

- 语义向量匹配的 embedding 缓存与检索机制参考了 [CharTyr/MaiBot-Better-Expression](https://github.com/CharTyr/MaiBot-Better-Expression) 插件的设计，特此感谢。

---

## 兼容性

- MaiBot >= 1.0.0
- SDK >= 2.0.0

---

## 许可

GPL-3.0-or-later
