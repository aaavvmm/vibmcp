# -*- coding: utf-8 -*-
"""
合成故障轴承信号发生器
======================
用途: ① 项目自闭环演示(无需真实数据集) ② 测试的 ground truth ③ 评测集批量生成。

信号模型(工程常用简化模型):
  x(t) = Σ_n  A·(1 + β·sin(2π·fr·t_n)) · e^{-(t-t_n)/τ} · sin(2π·f_r(t-t_n))  + noise
  - 冲击周期 1/f_fault (BPFI/BPFO/BSF/FTF 之一), normal 无冲击
  - 每次冲击激发轴承共振载波 f_r(默认 3kHz), 指数衰减 τ
  - 内圈故障随轴转负载调制 β=0.6; 外圈不调制(载荷区固定); 滚动体轻度调制
  - 加高斯噪声 SNR 可控
"""
import numpy as np

from .bearing import char_freqs

RESONANCE_HZ = 3000.0   # 轴承共振载波
TAU = 0.004             # 冲击衰减时间常数(s)


def synth_signal(fault: str | None, fs: float = 12000.0, fr: float = 29.95,
                 duration: float = 4.0, snr_db: float = 0.0,
                 machine: str = "cwru", seed: int = 0) -> tuple[np.ndarray, dict]:
    """生成一段合成振动信号。fault ∈ {None,'BPFI','BPFO','BSF','FTF'}。

    返回 (波形, 元数据)。SNR 以冲击信号功率为基准。
    """
    rng = np.random.default_rng(seed)
    cf = char_freqs(machine=machine, fr=fr)
    if fault and fault not in ("BPFI", "BPFO", "BSF", "FTF"):
        raise ValueError(f"fault 须为 BPFI/BPFO/BSF/FTF/None, got '{fault}'")

    n = int(fs * duration)
    t = np.arange(n) / fs
    x = np.zeros(n)

    if fault:
        f_fault = cf[fault]["coef"] * fr
        # 冲击到达时刻(带 ±1.5% 抖动, 模拟实际转频波动)
        period = 1.0 / f_fault
        n_imp = int(duration * f_fault * 1.2)
        beta = {"BPFI": 0.6, "BPFO": 0.0, "BSF": 0.25, "FTF": 0.1}[fault]
        for k in range(n_imp):
            t0 = k * period * (1 + rng.normal(0, 0.015))
            idx0 = int(t0 * fs)
            m = int(6 * TAU * fs)          # 冲击持续时间窗
            if idx0 + m >= n:
                break
            tt = np.arange(m) / fs
            amp = 1.0 + beta * np.sin(2 * np.pi * fr * t0)
            burst = amp * np.exp(-tt / TAU) * np.sin(2 * np.pi * RESONANCE_HZ * tt)
            x[idx0: idx0 + m] += burst

    # 加噪到指定 SNR
    psig = np.mean(x ** 2) + 1e-12
    pnoise = psig / (10 ** (snr_db / 10))
    x = x + rng.normal(0, np.sqrt(pnoise), n)

    meta = {"fs": fs, "fr": fr, "fault": fault or "normal", "machine": machine,
            "duration_s": duration, "snr_db": snr_db, "source": "synthetic",
            "seed": seed}
    return x, meta
