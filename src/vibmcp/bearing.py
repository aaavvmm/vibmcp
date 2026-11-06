# -*- coding: utf-8 -*-
"""
轴承几何 → 特征频率
===================
滚动轴承缺陷冲击重复频率(Healths et al. 经典公式):
  BPFO = (Z/2)·(1 - r·cosα)·fr   外圈
  BPFI = (Z/2)·(1 + r·cosα)·fr   内圈
  BSF  = (D/(2d))·(1 - (r·cosα)²)·fr   滚动体
  FTF  = (1/2)·(1 - r·cosα)·fr   保持架
其中 r = d/D, fr 为转轴转频(Hz)。

预设参数来自公开数据集官方文档:
  CWRU: SKF6205-2RS JEM 驱动端 (Z=9, d=7.94mm, D=39.03mm)
  MFPT: NiceBearing (Z=8, d=5.969mm, D=31.623mm)
  PU:   LFK 6208 (Z=8, d=6.75mm, D=29.05mm)  近似值
"""
import math

# ---- 预设几何 (Z滚动体数, d滚动体直径mm, D节径mm, α接触角deg) ----
PRESETS = {
    "cwru": {"Z": 9, "d": 7.94,  "D": 39.03,  "alpha": 0.0, "desc": "CWRU SKF6205-2RS JEM 驱动端"},
    "mfpt": {"Z": 8, "d": 5.969, "D": 31.623, "alpha": 0.0, "desc": "MFPT NiceBearing"},
    "pu":   {"Z": 8, "d": 6.75,  "D": 29.05,  "alpha": 0.0, "desc": "PU LFK 6208 (近似)"},
}

FAULTS = ("BPFI", "BPFO", "BSF", "FTF")


def char_freqs(machine: str | None = None, fr: float = 1.0,
               Z: int | None = None, d: float | None = None,
               D: float | None = None, alpha: float | None = None) -> dict:
    """计算四个特征频率(单位与 fr 一致, fr=1 时返回系数)。

    优先用显式几何参数, 否则查预设。返回 {名称: {hz, coef, label}} + 几何回显。
    """
    if Z and d and D:
        geo = {"Z": Z, "d": d, "D": D, "alpha": alpha or 0.0, "desc": "自定义几何"}
    else:
        if machine not in PRESETS:
            raise ValueError(f"未知预设 '{machine}', 可选: {list(PRESETS)} 或提供 Z/d/D")
        geo = dict(PRESETS[machine])
    r = geo["d"] / geo["D"]
    ca = math.cos(math.radians(geo["alpha"]))
    return {
        "BPFO": {"coef": geo["Z"] / 2 * (1 - r * ca), "label": "外圈故障频率"},
        "BPFI": {"coef": geo["Z"] / 2 * (1 + r * ca), "label": "内圈故障频率"},
        "BSF":  {"coef": geo["D"] / (2 * geo["d"]) * (1 - (r * ca) ** 2), "label": "滚动体故障频率"},
        "FTF":  {"coef": 0.5 * (1 - r * ca), "label": "保持架故障频率"},
        "_geo": geo,
        "_fr": fr,
    }
