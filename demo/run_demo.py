# -*- coding: utf-8 -*-
"""
一键演示: 不经过 MCP, 直接调用工具函数跑完整流程
=================================================
① 4 类合成信号(正常/内圈/外圈/滚动体)逐一诊断: 基线 vs harness 对照
② 打印 harness 决策轨迹(可审计)
③ 生成 HTML 报告到 output/
MCP 全流程演示(接入 Claude Code/Cursor)见 README。
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from vibmcp import synth
from vibmcp.diagnose import diagnose
from vibmcp.harness import DiagnosticHarness
from vibmcp.plotting import spectrum_plot, waveform_envelope
from vibmcp.report import save as save_report

OUT = os.path.join(os.path.dirname(__file__), "..", "output")


def main():
    print("=" * 66)
    print("VibMCP 一键演示 —— 合成信号诊断 (ground truth 已知, snr=0dB)")
    print("=" * 66)
    h = DiagnosticHarness()
    n_base = n_h = n_all = 0
    reports = []
    for fault in (None, "BPFI", "BPFO", "BSF"):
        x, meta = synth.synth_signal(fault, seed=42, snr_db=0)
        base = diagnose(x, meta["fs"], meta["fr"], machine=meta["machine"])
        hr = h.run(x, meta["fs"], meta["fr"], machine=meta["machine"])
        gt = meta["fault"]
        n_base += base["verdict"] == gt
        n_h += hr["verdict"] == gt
        n_all += 1
        mark = lambda v: "✓" if v == gt else "✗"
        print(f"\n真值 {gt or 'normal'}: 基线 {base['verdict']}({base['confidence']:.0%}) {mark(base['verdict'])}"
              f"  →  harness {hr['verdict']}({hr['confidence']:.0%}) {mark(hr['verdict'])}"
              f"  [{hr['steps']}步, 升级={hr['escalated']}]")
        if hr["escalated"]:
            for st in hr["trace"]:
                print(f"    · {st['step']}: {st['action']}")
                if st.get("decision"):
                    print(f"      ↳ {st['decision']}")
        p = save_report(base, waveform_envelope(x, meta["fs"]),
                        spectrum_plot(x, meta["fs"], meta["fr"], verdict=base["verdict"]),
                        meta, OUT)
        reports.append(p)
    print("-" * 66)
    print(f"基线 {n_base}/{n_all}  harness {n_h}/{n_all}  (更多档位对照: server.compare_harness)")
    print("HTML 报告:")
    for p in reports:
        print("  ", os.path.abspath(p))


if __name__ == "__main__":
    main()
