# 项目3 · VibMCP 注意事项（回顾用）

> 最后更新: 2026-10-12 · 状态: **完工, 待 git 提交 + 录 demo**
> 维护约定: 后续对项目的任何更改, 由 Claude 同步更新本文件

## 一句话

工业振动信号诊断 MCP Server——让任意 LLM Agent（Claude Code/Cursor）具备可解释的轴承故障诊断能力。物理证据融合引擎 + 自适应诊断 harness + 自闭环评测 + HTML 报告外壳。

## 运行命令

```bash
.venv/Scripts/python demo/run_demo.py          # 一键演示(4类信号+harness对照+HTML报告)
.venv/Scripts/python -m pytest tests/ -q       # 12 个测试
claude mcp add vibmcp -- F:/work/项目3/.venv/Scripts/python -m vibmcp.server   # 接入 Claude Code
```

报告产出在 `output/`（单文件 HTML, 图表 base64 内嵌, 可直接发人）。

## 核心数字（简历用）

- 诊断准确率（40条合成信号/档, ground truth 已知）:
  - **90% @ +5dB / 85% @ 0dB / 75% @ -5dB / 48% @ -10dB**（baseline 单次诊断）
  - harness: 全线不劣化, **-10dB 极端噪声 +5pp**（52%）
- 没到 95%：物理规则引擎卖点是可解释+零训练, 不是刷单点。简历写曲线+提升幅度, 不写孤立数字。
- 若要冲 95%: 加长信号时长 + 扩大评测集 + harness 精调, +5dB 档可摸到。**别在现有评测集上反复调参（过拟合）**。

## 架构速查

```
src/vibmcp/
├── server.py      11 个 MCP 工具（MCP SDK 2.x: from mcp.server.mcpserver import MCPServer）
├── diagnose.py    证据融合引擎: 谐波prominence + 边带 + 峭度 → verdict+置信度+证据链
├── harness.py     自适应运行时: 置信度门控→共振带搜索(带switch_margin防选择偏差)→Welch谱平均→证据对比裁决; 全程 trace
├── analysis.py    包络谱(带通+希尔伯特)、谐波prominence、时域指标
├── bearing.py     特征频率 BPFI/BPFO/BSF/FTF（CWRU/MFPT/PU 预设）
├── synth.py       合成故障信号（冲击串+共振载波+AM调制+噪声）, ground truth 已知
├── loaders.py     .npz/.mat/.csv/.wav 加载, 会话内 sig_id 注册表
├── plotting.py / report.py   matplotlib 出图 / HTML 报告
tests/test_core.py  12 测试   demo/run_demo.py  一键演示
```

## 面试深挖点（按 README"设计取舍"展开）

1. **包络解调前先共振带通**（2.5~3.5kHz）——Randall 教科书标准流程; 没有它故障谱线被噪声包络淹没
2. **谐波 prominence 而非绝对能量占比**——纯噪声包络谱低频加权, 绝对占比会让正常信号误报
3. **指数分布背景修正 1/ln2≈1.443**——谱幅值服从指数分布, 用中位数当背景会低估 44% 导致误报
4. **分段谱平均后改用 Wilson-Hilferty 修正** `(1-1/9K)^-3`——Gamma(K) 分布的均值/中位数比, K=1 时≈1.42 与理论吻合
5. **nfft=8192 而非全长 FFT**——细分辨率把故障线经抖动/调制摊成裙边, 污染背景分布假设; **频率分辨率≠检测灵敏度**
6. **升级用谱平均不用投票**——短段单票准确率不足50%时投票放大噪声; Welch 平均才是统计正确的聚合
7. **harness 门控含矛盾检测**——"结论 normal 但峭度>4"= 弱故障被淹没, 必须复核; 漏检代价 > 多算几步
8. **换带要 switch_margin**——多候选取最大值有选择偏差, 噪声尖峰会冒充证据

## 开发中踩过的坑（面试讲"debug 故事"的素材）

- `np.fft.rfft(env, 8192)` 在输入 48000 点时**静默截断**成前 0.68s——白丢 5/6 数据; 试过全长 FFT 反而崩（见设计取舍 5）, 回滚并写成注释
- 谱平均后背景修正不换 → welch 全判 normal（Gamma vs 指数分布）
- harness 初版投票逻辑反而降准确率（-10pp）→ 换带无门槛 + 投票聚合错误, 双重教训
- MCP SDK 装到 2.x, FastMCP 更名 MCPServer——一行迁移

## 已知边界（面试官追问时主动说）

- +5dB 档 harness 有一次单例翻错（-2.5pp）: 升级策略的代价, trace 可审计
- 合成信号是单共振载波模型（3kHz）; 真实结构靠 harness 频带扫描兜底
- 真实数据接入: CWRU 原始 .mat 在 `E:\研究生课题\代码\one\data\cwru\`（ball/inner/...）, loaders 已支持 .mat, 未跑过真实数据评测——**这是目前最大的待验证项**

## 待办

- [x] git init + 首次提交 (main bf59e74)
- [ ] 录 3 分钟演示视频: 终端接 Claude Code 跑真实诊断 + 打开 HTML 报告
- [ ] （可选）用真实 CWRU .mat 跑一轮评测, 补"真实数据"数字
- [ ] （可选）冲 +5dB 档 95%
