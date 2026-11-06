# -*- coding: utf-8 -*-
"""matplotlib 出图: 波形+包络 / 包络谱(标注特征频率)。Agg 后端, 供报告嵌入。"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import envelope_spectrum
from .bearing import char_freqs, FAULTS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"BPFI": "#e74c3c", "BPFO": "#3498db", "BSF": "#2ecc71", "FTF": "#9b59b6"}


def waveform_envelope(x: np.ndarray, fs: float, t_max: float = 0.4) -> str:
    """前 t_max 秒波形 + 希尔伯特包络。返回 PNG 字节。"""
    from scipy.signal import hilbert
    n = min(len(x), int(fs * t_max))
    t = np.arange(n) / fs
    env = np.abs(hilbert(x[:n]))
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=110)
    ax.plot(t, x[:n], lw=0.5, color="#555", label="波形")
    ax.plot(t, env, lw=1.2, color="#e67e22", label="包络")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("幅值")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("时域波形与包络", fontsize=10)
    fig.tight_layout()
    return _to_png(fig)


def spectrum_plot(x: np.ndarray, fs: float, fr: float, machine: str = "cwru",
                  nfft: int = 8192, fmax: float = 600.0, verdict: str | None = None) -> str:
    """包络谱 + 四个特征频率谐波族标注(按嫌疑度着色)。返回 PNG 字节。"""
    freqs, sp = envelope_spectrum(x, fs, nfft)
    cf = char_freqs(machine=machine, fr=fr)
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110)
    m = freqs <= fmax
    ax.plot(freqs[m], sp[m], lw=0.7, color="#34495e")

    for fault in FAULTS:
        coef = cf[fault]["coef"]
        color = COLORS[fault]
        lw = 1.6 if fault == verdict else 0.9
        a = 1.0 if fault == verdict else 0.45
        for k in range(1, 4):
            fc = coef * fr * k
            if fc <= fmax:
                ax.axvline(fc, color=color, lw=lw, alpha=a)
        ax.axvline(np.nan, color=color, lw=lw, label=f"{fault} = {coef:.2f}·fr")

    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("包络谱幅值")
    ax.set_title("包络谱与特征频率标注", fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    fig.tight_layout()
    return _to_png(fig)


def _to_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
