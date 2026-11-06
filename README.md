# VibMCP — 工业振动信号诊断 MCP Server

让任意 LLM Agent（Claude Code / Cursor / 任意 MCP 客户端）具备**可解释的轴承故障诊断能力**。

不是又一个 RAG 问答机器人：本项目把信号处理领域知识（包络解调、特征频率、证据融合）
封装成 MCP 工具集，LLM 负责编排与解释，物理引擎负责可审计的判定 —— 每个结论都能
溯源到包络谱上的具体频带。

## 亮点

- **物理证据融合诊断引擎**：谐波信噪比（harmonic prominence）+ 边带证据 + 峭度辅证
  加权融合，无需训练数据，输出置信度 + 完整证据链
- **自适应诊断 Harness**：置信度门控 → 共振带搜索 → 分段谱平均（Welch）→ 证据对比裁决，
  附完整决策轨迹（trace）。极端噪声（-10dB）下比单次诊断 **+5pp**，其余场景不劣化
- **自闭环评测**：内置合成故障信号发生器（ground truth 已知），一条命令出准确率曲线：
  90% @ +5dB / 85% @ 0dB / 75% @ -5dB / 48% @ -10dB
- **应用外壳**：单文件 HTML 诊断报告（证据链表格 + 波形/包络谱图 base64 内嵌，
  可直接发给任何人）

## 快速开始

```bash
pip install -e .
# 一键演示(合成信号, 4 类 + harness 对照 + HTML 报告)
python demo/run_demo.py
```

### 接入 Claude Code / Cursor

```bash
claude mcp add vibmcp -- <项目路径>/.venv/Scripts/python -m vibmcp.server
```

然后对 Agent 说：

> 用 vibmcp 加载 synth:BPFI 信号，跑 harness 诊断，解释证据链，最后生成报告

Agent 会自主执行：`load_signal → run_harness → 解读 trace → generate_report`，
产出 `output/report_*.html`。

## 工具清单（11 个）

| 工具 | 作用 |
|---|---|
| `list_signals` / `load_signal` | 数据获取（文件 / 内置合成信号） |
| `signal_stats` / `envelope_spectrum_tool` / `bearing_freqs` | 观察：时域指标、包络谱主峰、特征频率 |
| `diagnose_signal` | 单次物理证据融合诊断 |
| `run_harness` | 自适应诊断会话（门控/搜索/谱平均 + trace） |
| `compare_harness` / `run_evaluation` | 批量评测：baseline vs harness 对照、准确率曲线 |
| `synth_signal_tool` / `generate_report` | 合成信号生成 / HTML 报告 |

## 架构

```
MCP 客户端(LLM Agent)          Python 内核                 外壳
┌──────────────────┐   stdio   ┌──────────────────┐      ┌─────────────┐
│ 编排/解释/提问    │ ←──────→ │ server.py (11工具)│ ───→ │ HTML 报告    │
│ 按 docstring 决策 │          │ diagnose.py 引擎  │      │ (证据链+谱图)│
└──────────────────┘          │ harness.py 运行时 │      └─────────────┘
                              │ analysis/bearing  │
                              │ synth/loaders     │
                              └──────────────────┘
```

## 设计取舍（面试深挖点）

1. **包络解调前先共振带通**（2.5~3.5kHz）：轴承冲击激发结构共振，带外噪声会让
   包络谱被噪声包络淹没 —— Randall 教科书标准流程，实测从"故障谱线不可见"到"清晰可辨"。
2. **谐波信噪比而非绝对能量占比**：纯噪声的包络谱低频加权，绝对占比会让正常信号
   误报。每个谐波带与局部背景(±20Hz 窗中位数)对比，噪声底被同时归一。
3. **指数分布修正 1/ln2**：包络谱幅值服从指数分布，E[X]/median = 1.443。直接用
   中位数当背景会系统性低估 44%；分段谱平均后趋于 Gamma 分布，修正系数用
   Wilson-Hilferty 近似 `((1-1/9K))^-3` 解析求出。
4. **nfft=8192 而非全长 FFT**：实测细分辨率会把故障谱线经时基抖动/调制摊成裙边，
   污染背景窗的分布假设，prominence 反而失真 —— 频率分辨率 ≠ 检测灵敏度。
5. **Harness 升级用"谱平均"而非"投票"**：短段单票准确率不足 50% 时投票放大噪声；
   平均证据（Welch 思想）才是统计上正确的聚合方式。升级换带要求显著优势
   （switch_margin），防多候选取最大值的选择偏差。
6. **Harness 门控含矛盾检测**："结论 normal 但峭度>4"是弱故障被噪声淹没的信号，
   必须复核 —— 漏检的代价远大于多算几步。

## 评测

```
SNR +5dB: baseline 90%  harness 88%   (升级率 50%)
SNR  0dB: baseline 85%  harness 85%   (升级率 48%)
SNR -5dB: baseline 75%  harness 75%   (升级率 45%)
SNR -10dB: baseline 48% harness 52%   (升级率 42%)  ← +5pp
```

复现：`compare_harness(n_per_class=8, snr_db=...)` 或 server 的 `run_evaluation`。

## 技术栈

Python 3.10+ · MCP SDK 2.x (stdio) · numpy/scipy (信号处理) · matplotlib (出图)
· 零训练数据、零 GPU 依赖

## 已知边界

- 诊断预设几何参数为 CWRU/MFPT/PU 三公开数据集的轴承，可用 `Z/d/D` 自定义
- 合成信号为单共振载波模型（3kHz），真实结构的共振搜索请依赖 harness 的频带扫描
- 置信度门控在 +5dB 下会多触发一次升级（-2.5pp 的单例翻错），取舍见 trace
