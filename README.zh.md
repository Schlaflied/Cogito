# Cogito — 认知自我建模

> *"我思故我在"——但你在想什么？*

**Cogito** 是一套从个人知识库和 git 历史中构建认知自我模型的方法论。

核心前提：你的 Obsidian vault（或任何有 git 版本控制的知识库）包含两种信号——你*改动了什么*（git diff）和你*写了什么*（笔记内容）。两者合在一起，是你思维的时间戳记录。这个 pipeline 提取这两层信号，进行嵌入和聚类，把你自己都没意识到的模式呈现出来。

数据留在你本地。开源的是方法。

[English →](README.md)

---

## 三个方向

Cogito 把自我建模拆成三个可以独立运行的问题。

### 方向 A — 注意力地图
*我在什么时期关注什么？我的注意力怎么随时间变化？*

```
extract_diffs.py       → data/diffs.jsonl        （带时间戳的 diff 记录）
embed_and_cluster.py   → data/clusters.jsonl      （嵌入 + KMeans 聚类）
visualise.py           → data/self_model_3d.html  （可交互的时间线可视化）
```

每个点是一次 commit。在空间中的位置 = 语义内容。X 轴 = 时间。颜色 = 主题 cluster。点的大小 = 输出强度（净改动行数）。

---

### 方向 B — 行为模式
*我什么时候最活跃？我的工作节律是什么样的？*

```
temporal_patterns.py   → data/temporal_report.html  （时间/星期/强度热力图）
```

使用 `diffs.jsonl` 里已有的 `hour`、`weekday`、`net_lines` 字段，无需额外 API 调用。可以看出不同类型的思维在什么时间发生——深夜的 commit 和早晨的 commit 有什么不同，高产日和维护日有什么区别。

---

### 方向 C — 身份结构
*什么概念在我所有领域里反复出现，而我自己并没有意识到？*

```
embed_notes.py            → data/notes_chunks.jsonl       （笔记内容，分块嵌入）
cross_domain_analysis.py  → data/cross_domain_map.html    （行为层 + 内容层合并地图）
                          → data/cross_domain_clusters.json（机器可读，供 AI 解读）
```

这是最难的方向。它合并了两层数据：
- **行为层**（`diffs.jsonl`）——你*做了什么*
- **认知层**（`notes_chunks.jsonl`）——你*写了什么、想了什么*

同时跨越多个域（创作写作、技术工作、法律文件、求职材料）的 cluster，是认知锚点的候选——那些你在所有地方都在做、但从来没有选择过的模式。

JSON 输出设计为交给有你背景信息的 AI 来解读。系统呈现模式，AI 解释它意味着什么。

---

## 核心研究问题

> 系统没有承认人是人。

在校时间 ≠ 能力。关键词 ≠ 真实水平。正式记录 ≠ 真正发生的事。每一个现有的分类系统都在优化它能测量的东西，而不是真实存在的东西。

Cogito 是对这个问题的一种回应：如果你能从一个人的行为痕迹中重建他的认知结构，你就有了不依赖机构类别的证据。那个被所有现有分类遗漏的人，仍然留下了记录——在 diff 里，在笔记里，在跨越他所有工作的模式里。

这也是为什么 git diff 比单纯读笔记更重要：

> 笔记是最终状态。diff 是一次决定。

你删掉一句话又重写，这是你思维变化的数据点。标准的 RAG 会丢掉这个。Cogito 保留它。

---

## 安装依赖

```bash
pip install gitpython openai umap-learn scikit-learn plotly numpy python-dotenv matplotlib networkx jieba
```

在 vault 根目录创建 `.env` 文件：
```
OPENAI_API_KEY=sk-...
```

编辑每个脚本顶部的 `VAULT_DIR`，指向你的 vault（必须是 git 仓库）。

---

## API 费用估算

| 步骤 | 数量 | 估算费用 |
|------|------|----------|
| 方向 A（diff） | ~4000 个 commit | ~$0.50–1.00 USD |
| 方向 C（笔记） | ~3000 个 chunk | ~$0.30–0.60 USD |

使用 `text-embedding-3-small`。内置断点续跑支持，中断后重启不会重复嵌入。

---

## 相关项目

- [Plot Ark](https://github.com/Schlaflied/Plot-Ark) — 同一问题的机构规模版本：面向学习系统的 xAPI 行为分析
- [career-ops](https://github.com/santifer/career-ops) — 开源求职自动化 pipeline

---

## 这不是什么

- 不是心理治疗工具
- 不是生产力追踪器
- 不是内省的替代品

它是一面比你的记忆更长的镜子。

---

## 许可

MIT — 用它，fork 它，在你自己的数据上跑。

---

*由一个搞不清楚自己是什么驱动的人建造，为了找到答案。*
