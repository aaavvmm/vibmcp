"""
证据融合诊断引擎
================
不训练模型 —— 用物理规则做可解释诊断:
  1. 包络谱各特征频率谐波族 prominence (谐波带能量/局部背景, BPFI/BPFO/BSF/FTF, k=1..3)
  2. 内圈故障辅证: BPFI 谐波 ±fr 边带 prominence (内圈随轴旋转, 调制产生边带)
  3. 时域辅证: 峭度 (冲击型故障显著 >4)
  4. 证据加权融合 → 最高分故障类型; 全部低于正常阈值 → normal
  5. 置信度由证据强度与领先幅度组合而成

输出保留完整证据链(各谐波 prominence 明细), 让 LLM 能向用户解释"为什么"。
这是和黑盒分类器的核心差异 —— 每个结论都能溯源到谱上具体频带。
"""

import numpy as np

from .analysis import envelope_spectrum, family_prominence, sideband_prominence, time_stats
from .bearing import FAULTS, char_freqs

# prominence 分数阈值(Σ max(p_k-1,0), 经合成信号标定):
# 1/ln2 背景修正后正常信号 <0.15, 明确故障时主嫌 >0.3 —— 中间地带为可疑
SCORE_NORMAL = 0.2  # 低于此值 → 无故障证据
SCORE_STRONG = 1.5  # 高于此值 → 强证据
W_FAMILY = 1.0  # 谐波族 prominence 权重
W_SIDEBAND = 0.5  # 边带证据权重(仅对 BPFI 生效)
W_KURTOSIS = 0.3  # 峭度权重(时域辅证)


def diagnose(
    x: np.ndarray,
    fs: float,
    fr: float,
    machine: str = "cwru",
    nfft: int = 8192,
    band: tuple | None = (2500.0, 3500.0),
    Z: int | None = None,
    d: float | None = None,
    D: float | None = None,
) -> dict:
    """主诊断入口。返回含证据链与置信度的结构化结论。

    band: 包络解调前的共振带通(Hz), 默认 2.5~3.5kHz;
    共振频率未知时可用 harness 的共振带搜索确定。传 None 跳过带通。
    """
    freqs, sp = envelope_spectrum(x, fs, nfft, bandpass=band)
    ts = time_stats(x)
    return diagnose_from_spectrum(freqs, sp, fr, machine=machine, time_stats_=ts, Z=Z, d=d, D=D)


def diagnose_from_spectrum(
    freqs: np.ndarray,
    sp: np.ndarray,
    fr: float,
    machine: str = "cwru",
    time_stats_: dict | None = None,
    bg_corr: float | str = 1.4427,
    Z: int | None = None,
    d: float | None = None,
    D: float | None = None,
) -> dict:
    """从(可能已分段平均的)包络谱直接出诊断 —— harness 的 Welch 平均路径复用此入口。"""
    cf = char_freqs(machine=machine, fr=fr, Z=Z, d=d, D=D)
    ts = time_stats_ or {"kurtosis": 0.0}

    scores, evidence = {}, {}
    for fault in FAULTS:
        coef = cf[fault]["coef"]
        fam = family_prominence(freqs, sp, coef, fr, bg_corr=bg_corr)
        s = W_FAMILY * fam["score"]
        detail = {
            "family_score": round(fam["score"], 3),
            "prominence": fam["prom"],
            "char_freq_hz": round(coef * fr, 2),
        }
        if fault == "BPFI":
            sb = sideband_prominence(freqs, sp, coef, fr, bg_corr=bg_corr)
            s += W_SIDEBAND * sb["score"]
            detail["sideband_score"] = round(sb["score"], 3)
        if ts["kurtosis"] > 4:
            s += W_KURTOSIS * min((ts["kurtosis"] - 4) / 4, 1.0)
        scores[fault] = s
        evidence[fault] = detail

    top2 = sorted(scores.items(), key=lambda kv: -kv[1])
    best, second = top2[0], top2[1]

    if best[1] < SCORE_NORMAL:
        verdict, confidence = "normal", round(min(0.95, 1 - best[1] / SCORE_NORMAL), 2)
    else:
        verdict = best[0]
        margin = (best[1] - second[1]) / (best[1] + 1e-9)
        strength = min(best[1] / SCORE_STRONG, 1.0)
        confidence = round(min(0.99, 0.5 * strength + 0.5 * margin), 2)

    return {
        "verdict": verdict,  # normal / BPFI / BPFO / BSF / FTF
        "confidence": confidence,
        "runner_up": {"fault": second[0], "score": round(second[1], 3)},
        "char_freqs": {
            f: {"hz": round(cf[f]["coef"] * fr, 2), "coef_x_fr": round(cf[f]["coef"], 3)}
            for f in FAULTS
        },
        "evidence": evidence,
        "time_stats": ts,
        "geometry": cf["_geo"],
        "shaft_freq_hz": fr,
        "notes": _notes(verdict, best[1], ts),
    }


def _notes(verdict: str, score: float, ts: dict) -> list[str]:
    """生成给 LLM 的人话解释(证据链摘要)。"""
    out = []
    if verdict == "normal":
        out.append(f"各特征频率谐波族 prominence 均低于正常阈值 {SCORE_NORMAL}, 未见故障冲击调制。")
    else:
        label = {"BPFI": "内圈", "BPFO": "外圈", "BSF": "滚动体", "FTF": "保持架"}[verdict]
        lv = "强" if score >= SCORE_STRONG else "弱"
        out.append(
            f"包络谱在{label}故障特征频率谐波族检测到{lv}证据 (prominence 得分 {score:.2f})。"
        )
    if ts["kurtosis"] > 4:
        out.append(f"峭度 {ts['kurtosis']} (>4) 表明波形存在显著冲击成分, 与调制型故障一致。")
    elif verdict != "normal":
        out.append(f"峭度 {ts['kurtosis']} 不高, 建议复核信号长度与信噪比。")
    out.append("诊断基于包络谱物理证据融合, 无需训练数据; 结论可由 evidence 字段逐谐波溯源。")
    return out
