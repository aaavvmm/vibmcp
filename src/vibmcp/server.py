# -*- coding: utf-8 -*-
"""
VibMCP MCP Server —— 9 个工具, stdio 传输
=========================================
工具设计面向 Agent 编排:
  - 中间工具返回 JSON 文本摘要, 供 LLM 推理并决定下一步
  - 会话状态(已加载信号)驻留服务端, LLM 只传轻量 sig_id
  - generate_report 是"应用外壳"落点: 产出可直接打开的 HTML 报告

接入 Claude Code:  claude mcp add vibmcp -- F:/work/项目3/.venv/Scripts/python -m vibmcp.server
"""
import json
import os

from mcp.server.mcpserver import MCPServer

from . import loaders, synth
from .analysis import envelope_spectrum, time_stats, top_peaks
from .bearing import PRESETS, char_freqs
from .diagnose import diagnose
from .plotting import spectrum_plot, waveform_envelope
from .report import save as save_report

mcp = MCPServer(
    "vibmcp",
    instructions=(
        "工业振动信号诊断工具集。典型流程: load_signal → diagnose → generate_report。"
        "对不确定的样本可先用 signal_stats/envelope_spectrum 观察再下结论。"
    ),
)

OUT_DIR = os.environ.get("VIBMCP_OUT", "output")


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=str)


# ---------- 数据获取 ----------

@mcp.tool()
def list_signals() -> str:
    """列出本会话已加载的信号及 data/ 目录下的可用文件(含内置合成信号)。"""
    data_dir = os.environ.get("VIBMCP_DATA", "data")
    files = []
    if os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir)
                 if f.lower().endswith((".npz", ".mat", ".csv", ".wav"))]
    return _j({"session": loaders.list_ids(),
               "data_dir": os.path.abspath(data_dir),
               "files": files,
               "builtin_synth": ["synth:None", "synth:BPFI", "synth:BPFO", "synth:BSF", "synth:FTF"],
               "hint": "synth:XXX 为内置合成信号(ground truth 已知, 参数 fs=12k, fr=29.95, snr=0dB)"})


@mcp.tool()
def load_signal(source: str, fs: float | None = None, fr: float | None = None) -> str:
    """加载信号到会话。source 可为: 文件路径(.npz/.mat/.csv/.wav)、data/ 下文件名、
    或内置合成信号 'synth:BPFI' 等(自动带 ground truth label)。返回 sig_id 与元数据。"""
    if source.startswith("synth:"):
        fault = source.split(":", 1)[1]
        fault = None if fault in ("None", "normal", "") else fault
        x, meta = synth.synth_signal(fault, fs=fs or 12000.0, fr=fr or 29.95)
        label = meta["fault"]
        sid = loaders.register(x, meta, label=label)
        return _j({"sig_id": sid, "meta": meta, "ground_truth": label})
    path = source if os.path.isabs(source) or os.path.sep in source \
        else os.path.join(os.environ.get("VIBMCP_DATA", "data"), source)
    d = loaders.load_file(path, fs=fs, fr=fr)
    sid = loaders.register(d["x"], d["meta"])
    return _j({"sig_id": sid, "meta": d["meta"]})


# ---------- 观察与探索 ----------

@mcp.tool()
def signal_stats(sig_id: str) -> str:
    """时域条件指标(RMS/峭度/峰值因子/脉冲因子等, 全部无量纲)。峭度>4 提示冲击成分。"""
    rec = loaders.get(sig_id)
    return _j({"sig_id": sig_id, "stats": time_stats(rec["x"]), "meta": rec["meta"]})


@mcp.tool()
def envelope_spectrum_tool(sig_id: str, k: int = 10) -> str:
    """希尔伯特包络谱前 k 个主峰。用于初步判断调制频率成分, 保存谱图 PNG。"""
    rec = loaders.get(sig_id)
    fs = rec["meta"]["fs"]
    freqs, sp = envelope_spectrum(rec["x"], fs)
    peaks = top_peaks(freqs, sp, k=k)
    png = os.path.join(OUT_DIR, f"env_{sig_id}.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    import base64
    with open(png, "wb") as f:
        f.write(base64.b64decode(spectrum_plot(rec["x"], fs, rec["meta"].get("fr") or 1.0)))
    return _j({"sig_id": sig_id, "top_peaks_hz": peaks, "plot": os.path.abspath(png),
               "hint": "峰若聚在 fr 的倍数附近, 对照 bearing_freqs 的特征频率判断故障类型"})


@mcp.tool()
def bearing_freqs(machine: str = "cwru", fr: float = 29.95,
                  Z: int | None = None, d: float | None = None,
                  D: float | None = None) -> str:
    """计算轴承特征频率(BPFI/BPFO/BSF/FTF)。可查预设(cwru/mfpt/pu)或自定义几何 Z/d/D。"""
    cf = char_freqs(machine=machine, fr=fr, Z=Z, d=d, D=D)
    geo = cf.pop("_geo")
    return _j({"presets": {k: v["desc"] for k, v in PRESETS.items()},
               "geometry": geo,
               "char_freqs_hz": {k: round(cf[k]["coef"] * fr, 2) for k in cf}})


# ---------- 诊断与报告 ----------

@mcp.tool()
def diagnose_signal(sig_id: str, machine: str = "cwru",
                    Z: int | None = None, d: float | None = None,
                    D: float | None = None) -> str:
    """主诊断: 包络谱证据融合 → 故障类型 + 置信度 + 完整证据链(可解释)。"""
    rec = loaders.get(sig_id)
    fr = rec["meta"].get("fr")
    if not fr:
        return _j({"error": "元数据缺转频 fr (Hz), 请先 load_signal 时传入或重新加载",
                   "hint": "fr = 转速rpm/60; 不确定时可用 envelope_spectrum_tool 的主峰估计"})
    r = diagnose(rec["x"], rec["meta"]["fs"], fr, machine=machine, Z=Z, d=d, D=D)
    loaders.LAST_DIAG[sig_id] = r
    gt = rec.get("label")
    if gt:
        r["ground_truth"] = gt
        r["correct"] = (gt == r["verdict"])
    return _j(r)


@mcp.tool()
def run_harness(sig_id: str, machine: str = "cwru") -> str:
    """自适应诊断 harness(推荐): 置信度门控 → 共振带搜索 → 多段投票,
    附完整决策轨迹 trace。低置信/强噪声场景比单次诊断显著更稳。"""
    from .harness import DiagnosticHarness
    rec = loaders.get(sig_id)
    fr = rec["meta"].get("fr")
    if not fr:
        return _j({"error": "元数据缺转频 fr (Hz), 请先 load_signal 时传入"})
    r = DiagnosticHarness().run(rec["x"], rec["meta"]["fs"], fr, machine=machine)
    gt = rec.get("label")
    if gt:
        r["ground_truth"] = gt
        r["correct"] = (gt == r["verdict"])
    return _j(r)


@mcp.tool()
def compare_harness(n_per_class: int = 8, snr_db: float = -5.0,
                    machine: str = "cwru") -> str:
    """对照评测: 单次诊断 vs 自适应harness 在同一批合成信号上的准确率。
    输出提升幅度(pp) —— harness 策略收益的量化证据。"""
    from .harness import DiagnosticHarness
    base_ok = h_ok = n = 0
    h = DiagnosticHarness()
    for fault in (None, "BPFI", "BPFO", "BSF", "FTF"):
        for seed in range(n_per_class):
            x, m = synth.synth_signal(fault, machine=machine, snr_db=snr_db, seed=seed)
            truth = m["fault"]
            base_ok += diagnose(x, m["fs"], m["fr"], machine=machine)["verdict"] == truth
            h_ok += h.run(x, m["fs"], m["fr"], machine=machine)["verdict"] == truth
            n += 1
    return _j({"snr_db": snr_db, "n_total": n,
               "baseline_acc": round(base_ok / n, 3),
               "harness_acc": round(h_ok / n, 3),
               "gain_pp": round((h_ok - base_ok) / n * 100, 1),
               "note": "harness 每步决策细节见 run_harness 返回的 trace 字段"})


@mcp.tool()
def generate_report(sig_id: str, machine: str = "cwru") -> str:
    """生成单文件 HTML 诊断报告(证据链+谱图内嵌), 返回路径。需已 diagnose 或会自动诊断。"""
    rec = loaders.get(sig_id)
    fr = rec["meta"].get("fr") or 29.95
    # harness 结论缺证据链明细(其结构不同), 有则优先用完整诊断结果渲染
    r = loaders.LAST_DIAG.get(sig_id)
    if not r or not r.get("evidence"):
        r = diagnose(rec["x"], rec["meta"]["fs"], fr, machine=machine)
    png_wave = waveform_envelope(rec["x"], rec["meta"]["fs"])
    png_spec = spectrum_plot(rec["x"], rec["meta"]["fs"], fr, machine, verdict=r["verdict"])
    path = save_report(r, png_wave, png_spec, rec["meta"], OUT_DIR)
    return _j({"report": os.path.abspath(path), "verdict": r["verdict"],
               "confidence": r["confidence"]})


@mcp.tool()
def synth_signal_tool(fault: str | None = None, snr_db: float = 0.0,
                      fr: float = 29.95, seed: int = 0) -> str:
    """生成合成故障信号并加载到会话(测试/演示/评测用, ground truth 已知)。
    fault ∈ None/BPFI/BPFO/BSF/FTF。"""
    x, meta = synth.synth_signal(None if fault in (None, "normal") else fault,
                                 fr=fr, snr_db=snr_db, seed=seed)
    sid = loaders.register(x, meta, label=meta["fault"])
    return _j({"sig_id": sid, "meta": meta})


# ---------- 批量评测(出数字) ----------

@mcp.tool()
def run_evaluation(n_per_class: int = 5, snr_db: float = 0.0,
                   machine: str = "cwru") -> str:
    """批量合成信号诊断评测: 每类(含 normal) n 条, 输出混淆计数与准确率。
    用于验证诊断引擎、生成简历指标数字。"""
    from collections import Counter
    confusion: dict[str, Counter] = {}
    n_ok = n_all = 0
    for fault in (None, "BPFI", "BPFO", "BSF", "FTF"):
        for seed in range(n_per_class):
            x, meta = synth.synth_signal(fault, machine=machine, snr_db=snr_db, seed=seed)
            fr = meta["fr"]
            r = diagnose(x, meta["fs"], fr, machine=machine)
            truth = meta["fault"]
            confusion.setdefault(truth, Counter())[r["verdict"]] += 1
            n_all += 1
            n_ok += truth == r["verdict"]
    return _j({"accuracy": round(n_ok / n_all, 3), "n_total": n_all,
               "snr_db": snr_db, "machine": machine, "confusion": {k: dict(v) for k, v in confusion.items()}})


def main():
    mcp.run()


if __name__ == "__main__":
    main()
