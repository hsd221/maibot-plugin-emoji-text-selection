# Emoji Text Selector —— 纯文本表情包选择插件

让 MaiBot 在没有视觉模型的情况下，通过纯文本 LLM 完成表情包的情绪标签匹配与发送。

---

## 这是什么？

MaiBot 内置的 `send_emoji` 工具通过 VLM 看图选表情，未配置视觉模型时会报错。

这个插件提供纯文本替代方案：将全量情绪标签列表 + 聊天上下文交给文本 LLM 选出最匹配的 1-5 个标签，再通过幂集降级匹配找到最合适的表情包发送。

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

安装后在 WebUI 的插件管理页面启用即可。

---

## 配置

```toml
[plugin]
config_version = "1.0.0"

# 传给 LLM 的最大情绪标签数量，0 表示不限制
max_emotion_tags = 80

# LLM 最多选择的标签数
max_selected_tags = 5

# 标签选择用的模型任务名，空字符串表示使用默认 text 模型
llm_model = ""
```

---

## 工作原理

1. Planner 通过 `tool_search` 发现 `select_emoji` 工具
2. LLM 调用工具时，插件获取全量情绪标签（如 "开心, 赞, 无语, 猫, 狗, ..."）
3. 调用文本 LLM 从标签列表中选出 1-5 个最匹配当前语境的标签
4. 插件按幂集降级匹配：全交集 → 2 标签组合 → 单标签 → 随机兜底
5. 发送选中的表情包

---

## 和内置表情功能的区别

|  | 内置 VLM 表情 | 本插件 |
|------|-------|------|
| 选择方式 | 视觉模型拼图选 | 文本标签匹配 |
| 模型要求 | 必须 VLM | 纯文本模型即可 |
| 精确度 | 高 | 中（依赖表情的描述覆盖度） |
| 速度 | 慢（拼图 + VLM） | 快（直接文本推理） |

---

## 兼容性

- MaiBot >= 1.0.0
- SDK >= 2.0.0

---

## 许可

GPL-3.0-or-later
