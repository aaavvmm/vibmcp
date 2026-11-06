"""
诊断 Harness —— LLM 之外的运行时层
==================================
harness = 包住模型的编排循环: 置信度门控、自适应策略搜索、分段谱平均、
预算控制、完整决策轨迹(trace)。本模块把"一次诊断"升级成"一次会自我
检查、会换策略、会重试的诊断会话":

  ┌─ step1 时域观察 ──── 峭度/统计 → 决定是否值得深入
  ├─ step2 基线诊断 ──── 默认共振带 (2.5~3.5kHz)
  │    └─ 置信度 ≥ 门控 → 提前退出(省预算)
  ├─ step3 共振带搜索 ── 扫描候选频带, 显著优于默认带才换(防选择偏差)
  ├─ step4 分段谱平均 ── Welch 思想: 平均 K 段包络谱, 噪声底 ~1/√K 收紧
  │    (低信噪比下"平均证据"优于"投票" —— 单段准确率不足时投票放大噪声)
  └─ 输出 verdict + 完整 trace(每步决策依据可审计)

设计原则: 与 LLM 无关 —— MCP 场景下 LLM Agent 按工具 docstring 扮演
同样的角色; 本模块证明该策略可被确定性实现并量化收益(对比见
server.run_evaluation 的 baseline/harness 对照)。
"""

import time

import numpy as np

from .analysis import envelope_spectrum, family_prominence, time_stats
from .bearing import FAULTS, char_freqs
from .diagnose import diagnose, diagnose_from_spectrum

# 候选共振带(Hz): 覆盖常见轴承结构共振区(末带上限避开 Nyquist)
BAND_GRID = [
    (1000.0, 2000.0),
    (1500.0, 2500.0),
    (2000.0, 3000.0),
    (2500.0, 3500.0),
    (3000.0, 4500.0),
    (4500.0, 5800.0),
]


class DiagnosticHarness:
    """置信度门控 + 共振带搜索 + 多段投票的诊断会话编排器。"""

    def __init__(self, conf_gate: float = 0.55, n_avg: int = 4, switch_margin: float = 0.5):
        self.conf_gate = conf_gate  # 低于此置信度触发升级策略
        self.n_avg = n_avg  # 分段谱平均段数(Welch 思想)
        self.switch_margin = switch_margin  # 换共振带所需的最小优势(防选择偏差)

    def run(self, x: np.ndarray, fs: float, fr: float, machine: str = "cwru") -> dict:
        """执行诊断会话。返回 {verdict, confidence, trace, steps, escalated}。"""
        t0 = time.perf_counter()
        trace: list[dict] = []

        # step1 时域观察
        ts = time_stats(x)
        trace.append(
            {
                "step": "time_stats",
                "action": "时域条件指标",
                "finding": {"kurtosis": ts["kurtosis"], "rms": ts["rms"]},
                "decision": "峭度>4 提示冲击成分, 进入诊断"
                if ts["kurtosis"] > 4
                else "峭度不高, 仍做基线诊断(防弱冲击漏检)",
            }
        )

        # step2 基线诊断(默认共振带)
        base = diagnose(x, fs, fr, machine=machine)
        trace.append(
            {
                "step": "baseline_diagnose",
                "action": "默认共振带(2.5~3.5kHz)包络诊断",
                "finding": {
                    "verdict": base["verdict"],
                    "confidence": base["confidence"],
                    "runner_up": base["runner_up"],
                },
            }
        )
        # 门控: 低置信必须升级; "正常"结论 + 高峭度是矛盾信号(噪声可能淹没
        # 了弱故障), 也必须复核 —— 漏检比多花几步计算的代价大得多
        contradicts = base["verdict"] == "normal" and ts["kurtosis"] > 4
        if base["confidence"] >= self.conf_gate and not contradicts:
            trace.append(
                {
                    "step": "exit",
                    "action": "置信度达标, 提前退出",
                    "decision": f"共 {len(trace)} 步, 未动用升级策略",
                }
            )
            return self._wrap(base["verdict"], base["confidence"], trace, t0, escalated=False)
        if contradicts:
            trace.append(
                {
                    "step": "gate",
                    "action": "结论与峭度矛盾",
                    "finding": {"verdict": base["verdict"], "kurtosis": ts["kurtosis"]},
                    "decision": "正常结论但峭度>4, 疑似弱故障被噪声淹没, 升级复核",
                }
            )

        # step3 共振带搜索 —— 只在候选带显著优于默认带时才换带
        # (在多个噪声候选里取最大值存在选择偏差, 无显著优势不动摇基线)
        best, _ = self._resonance_search(x, fs, fr, machine, trace)
        if best["confidence"] >= self.conf_gate:
            return self._wrap(best["verdict"], best["confidence"], trace, t0, escalated=True)

        # step4 分段谱平均(Welch 思想): 故障谱线在各段持续存在, 噪声底随段数
        # 按 1/√K 收紧 —— 低信噪比下"平均证据"才是统计上正确的聚合方式
        # (投票要求单票准确率>50%, 短段往往达不到, 投票反而放大噪声)。
        avg = self._welch_diagnose(x, fs, fr, machine, best.get("_band"), trace)
        # 最终裁决: 谱平均结果与全信号结果谁的证据更强听谁的(不盲目翻盘)
        best_ev = max(_evidence.get("family_score", 0) for _evidence in best["evidence"].values())
        avg_ev = max(_evidence.get("family_score", 0) for _evidence in avg["evidence"].values())
        if avg["verdict"] != best["verdict"] and avg_ev < best_ev * 1.2:
            trace.append(
                {
                    "step": "final_decision",
                    "action": "对比证据强度",
                    "finding": {"avg": avg["verdict"], "full": best["verdict"]},
                    "decision": f"谱平均证据不足 1.2 倍, 保留全信号结论 {best['verdict']}",
                }
            )
            return self._wrap(best["verdict"], best["confidence"], trace, t0, escalated=True)
        return self._wrap(avg["verdict"], avg["confidence"], trace, t0, escalated=True)

    # ---------- 升级策略 ----------

    def _resonance_search(self, x, fs, fr, machine, trace) -> tuple[dict, dict]:
        """扫描候选频带: 每个带算四类故障谐波族 prominence, 取最强带的完整诊断。"""
        results: list[dict] = []
        for band in BAND_GRID:
            freqs, sp = envelope_spectrum(x, fs, bandpass=band)
            cf = char_freqs(machine=machine, fr=fr)
            best_score, best_fault = 0.0, None
            for f in FAULTS:
                s = family_prominence(freqs, sp, cf[f]["coef"], fr)["score"]
                if s > best_score:
                    best_score, best_fault = s, f
            results.append({"band": band, "score": best_score, "fault": best_fault})
        top = max(results, key=lambda r: r["score"])
        default = next(r for r in results if r["band"] == (2500.0, 3500.0))
        if top["band"] != default["band"] and top["score"] < default["score"] + self.switch_margin:
            top = default
            decision = (
                f"候选带优势不足 switch_margin={self.switch_margin}, "
                f"保持默认带 (防噪声候选冒充证据)"
            )
        else:
            decision = f"选定 {list(top['band'])} (证据 {top['score']:.2f}), 重诊断"
        trace.append(
            {
                "step": "resonance_search",
                "action": f"扫描 {len(BAND_GRID)} 个候选共振带",
                "finding": {
                    "per_band": [
                        {
                            "band": list(r["band"]),
                            "evidence": round(r["score"], 2),
                            "suggests": r["fault"],
                        }
                        for r in results
                    ]
                },
                "decision": decision,
            }
        )
        band = top["band"]
        r = diagnose(x, fs, fr, machine=machine, band=band)  # type: ignore[arg-type]
        r["_band"] = band
        return r, top

    def _welch_diagnose(self, x, fs, fr, machine, band, trace) -> dict:
        """分段包络谱平均后重诊断。谱平均使噪声底方差 ~ 1/K, 故障谱线不动。"""
        n = self.n_avg
        seg_len = len(x) // n
        freqs: np.ndarray | None = None
        sp_avg: np.ndarray | None = None
        for i in range(n):
            f_i, sp_i = envelope_spectrum(x[i * seg_len : (i + 1) * seg_len], fs, bandpass=band)
            freqs, sp_avg = f_i, (sp_i if sp_avg is None else sp_avg + sp_i)
        from .analysis import gamma_bg_corr

        assert freqs is not None and sp_avg is not None  # n ≥ 1 必然填充
        r = diagnose_from_spectrum(freqs, sp_avg / n, fr, machine=machine, bg_corr=gamma_bg_corr(n))
        r["_band"] = band
        trace.append(
            {
                "step": "welch_avg",
                "action": f"{n} 段包络谱平均后重诊断",
                "finding": {
                    "verdict": r["verdict"],
                    "confidence": r["confidence"],
                    "evidence": {k: v.get("family_score") for k, v in r["evidence"].items()},
                },
                "decision": f"噪声底收紧 ~1/√{n}, 期望故障谱线相对更突出",
            }
        )
        return r

    @staticmethod
    def _wrap(verdict, confidence, trace, t0, escalated) -> dict:
        return {
            "verdict": verdict,
            "confidence": confidence,
            "trace": trace,
            "steps": len(trace),
            "escalated": escalated,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }
