# emoji_text_selection —— 纯文本表情选择插件

> 让 Maisaka 在不依赖视觉模型的前提下，用纯文本模型完成表情包选择与发送。

---

## 这是什么？

MaiBot 内置的表情发送功能需要视觉模型（VLM）看图选表情。如果你只配了纯文本模型，表情功能会直接报错。

这个插件提供了一个替代方案：用文本描述来匹配表情（Levenshtein 编辑距离），完全不需要 VLM，纯文本模型就能用。

---

## 安装

### 方式一：从插件中心安装（推荐）

在 MaiBot WebUI 的插件市场中搜索 `emoji-text-selection`，点击安装。

### 方式二：手动安装

```bash
# 进入 MaiBot 插件目录
cd /path/to/MaiBot/plugins

# 克隆插件
git clone https://github.com/hsd221/maibot-plugin-emoji-text-selection.git

# 重启 MaiBot
```

安装后在 WebUI 的插件管理页面启用即可。

---

## 配置

插件本身无需额外配置，开箱即用。只需确保 `model_config.toml` 中 `emoji` 任务配置了可用的纯文本模型：

```toml
[model_task_config.emoji]
model_list = ["your-text-model"]
```

---

## 工作原理

1. 当 Maisaka 需要发送表情时，插件拦截发给 LLM 的工具列表
2. 把内置的 VLM 工具 `send_emoji` 替换成文本版的 `send_emoji_text`
3. LLM 只需要提供一个情绪标签（如"开心""难过"），不需要看图
4. 插件用这个标签在表情库中做文本匹配，找到最合适的表情并发送

---

## 和内置表情功能的区别

|  | 内置 VLM 表情 | 本插件 |
|------|-------|------|
| 选择方式 | 视觉模型看图选 | 文本标签匹配 |
| 模型要求 | 必须 VLM | 纯文本模型即可 |
| 精确度 | 高 | 中（依赖表情的描述覆盖度） |
| 速度 | 慢（需拼图+调 VLM） | 快 |

---

## 文件结构

```
emoji_text_selection/
├── _manifest.json    # 插件元数据
├── plugin.py         # 插件主体
├── config.toml       # 插件配置
├── README.md         # 本文件
├── LICENSE           # GPL-3.0
└── i18n/
    ├── zh-CN.json    # 中文翻译
    └── en-US.json    # 英文翻译
```

---

## 常见问题

### 表情匹配不太准？

这是纯文本匹配方案的固有局限。它基于编辑距离排序，不涉及语义理解，无法区分"微笑"和"假笑"这种细微差别。

**解决办法**：
- 在管理后台给表情补充更丰富的描述标签
- 如果对精度要求极高，建议配置视觉模型使用内置 `send_emoji`

### 表情发不出去？

1. 确认插件已在 WebUI 中启用
2. 确认 `model_config.toml` 中 `emoji` 任务配置了可用模型
3. 确认表情库中有描述标签的表情

---

## 兼容性

- MaiBot ≥ 1.0.0
- SDK ≥ 2.0.0

---

## 许可

GPL-3.0-or-later
