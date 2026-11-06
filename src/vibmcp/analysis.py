"""
信号分析: 包络谱 + 时域条件指标 + 物理特征
==========================================
包络解调是轴承诊断的标准手段: 希尔伯特变换提取包络 → 去均值 → FFT。
故障冲击会在包络谱上表现为特征频率及其谐波处的离散谱线。

时域条件指标全部无量纲化, 对载荷/转速变化稳健:
  RMS / 峭度 / 偏度 / 峰值因子 / 脉冲因子 / 波形因子 / 裕度因子 / 峰峰值
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

FS_DEFAULT = 12000.0
FMAX_DEFAULT = 1000.0  # 包络谱分析带宽(Hz)
HARMS = 3  # 谐波族阶数


def envelope_spectrum(
    x: np.ndarray, fs: float, nfft: int = 8192, bandpass: tuple | None = (2500.0, 3500.0)
) -> tuple[np.ndarray, np.ndarray]:
    """希尔伯特包络谱。返回 (freqs, 幅值谱)。

    nfft=8192 是实测最优的工程折中: 更细的分辨率会把故障谱线经时基抖动/调制
    摊成"裙边", 污染背景窗的指数分布假设, prominence 反而失真 (细分辨率
    ≠高检测灵敏度, 见 README"设计取舍")。
    bandpass: 包络解调前的共振带通(Hz)。轴承冲击激发结构共振(本工具集默认
    2.5~3.5kHz), 先带通再解调可滤掉带外噪声、显著提升故障谱线信噪比 ——
    这是经典包络分析的标准步骤(Randall《Vibration based Condition Monitoring》)。
    传 None 跳过(原始包络)。
    """
    xr = x - x.mean()
    if bandpass:
        from scipy.signal import butter, sosfiltfilt

        bandpass = (max(bandpass[0], 1.0), min(bandpass[1], fs / 2 - 1.0))  # 防 Nyquist 越界
        sos = butter(4, bandpass, btype="bandpass", fs=fs, output="sos")
        xr = sosfiltfilt(sos, xr)
    env = np.abs(hilbert(xr))
    env = env - env.mean()
    sp = np.abs(np.fft.rfft(env, nfft))
    freqs = np.arange(len(sp)) * fs / nfft
    return freqs, sp


def top_peaks(freqs: np.ndarray, sp: np.ndarray, k: int = 12, fmin: float = 5.0) -> list[dict]:
    """包络谱前 k 个峰(忽略直流附近与频谱边缘)。返回 [{hz, amp}] 降序。"""
    lo = int(fmin / (freqs[1] - freqs[0])) if len(freqs) > 1 else 0
    seg = sp[lo : len(sp) // 2]  # 只看 Nyquist 内的物理频段
    order = np.argsort(seg)[::-1][:k]
    return [{"hz": round(float(freqs[lo + i]), 2), "amp": round(float(seg[i]), 4)} for i in order]


def family_energy(freqs: np.ndarray, sp: np.ndarray, f_center: float, tol_hz: float) -> float:
    """f_center ± tol_hz 频带能量占比(归一化到全谱)。"""
    total = sp.sum() + 1e-12
    lo = np.searchsorted(freqs, max(0.0, f_center - tol_hz))
    hi = np.searchsorted(freqs, f_center + tol_hz)
    return float(sp[lo:hi].sum() / total)


def gamma_bg_corr(k: int) -> float:
    """K 段谱平均后幅值谱 ~ Gamma(K,1): mean/median 比的 Wilson-Hilferty 近似。

    K=1 → 1.424 (理论指数分布 1/ln2=1.443); K=4 → 1.088; K→∞ → 1。
    供分段谱平均路径计算背景修正系数, 避免小样本 mean/median 估计的噪声。
    """
    return float((1 - 1 / (9 * k)) ** -3)


def family_prominence(
    freqs: np.ndarray,
    sp: np.ndarray,
    coef: float,
    fr: float,
    harms: int = HARMS,
    tol_frac: float = 0.02,
    bg_corr: float | str = 1.4427,
) -> dict:
    """谐波信噪比(harmonic prominence): 各谐波带能量 / 局部背景期望能量。

    背景 = 谐波两侧 ±W 窗口内(排除谐波段)的谱中位数 × 波段bin数。
    纯噪声/慢变漂移被局部基线吸收 → prominence≈1; 真实故障谱线是尖峰 → ≫1。
    相比绝对能量占比, 该度量对包络谱整体形状(低频加权、噪声底)不敏感。

    返回 {"score": Σ max(p_k - 1, 0), "prom": [k=1..3 的 prominence]}。
    """
    df = freqs[1] - freqs[0]
    prom = []
    for k in range(1, harms + 1):
        fc = coef * fr * k
        tol = max(2 * df, tol_frac * fc)
        lo, hi = np.searchsorted(freqs, fc - tol), np.searchsorted(freqs, fc + tol)
        if lo >= hi:
            prom.append(0.0)
            continue
        e_band = sp[lo:hi].sum()
        w = max(6 * tol, 20.0)  # 背景窗口半宽
        wlo, whi = np.searchsorted(freqs, max(fc - w, 2.0)), np.searchsorted(freqs, fc + w)
        mask = np.ones(whi - wlo, dtype=bool)
        mask[max(0, lo - wlo) : hi - wlo] = False  # 排除谐波段本身
        seg = sp[wlo:whi][mask]
        # 背景期望 = 中位数 × 分布修正系数:
        #   原始谱线服从指数分布, E[X]/median = 1/ln2 ≈ 1.443, 直接用中位数会
        #   低估背景 44% → 正常信号误报; 分段平均后趋于 Gamma 分布, 比值变小,
        #   此时用 bg_corr="auto"(局段 mean/median, 封顶 1/ln2)自适应。
        bg_med = np.median(seg) if len(seg) else 0.0
        corr_f = float(bg_corr)
        if bg_corr == "auto":
            corr_f = min(float(np.mean(seg)) / (bg_med + 1e-12), 1.4427) if len(seg) else 1.0
        e_bg = bg_med * corr_f * (hi - lo) + 1e-12
        prom.append(float(e_band / e_bg))
    return {"score": sum(max(p - 1.0, 0.0) for p in prom), "prom": [round(p, 2) for p in prom]}


def sideband_prominence(
    freqs: np.ndarray,
    sp: np.ndarray,
    coef: float,
    fr: float,
    harms: int = HARMS,
    bg_corr: float | str = 1.4427,
) -> dict:
    """内圈故障判别器: BPFI 谐波 ±fr 边带的 prominence 之和。

    内圈随轴旋转周期性进出载荷区 → 包络谱出现 fc ± k·fr 边带;
    外圈固定不动无边带 —— 这是区分内圈/外圈的关键物理证据。
    """
    df = freqs[1] - freqs[0]
    total = 0.0
    proms = []
    for k in range(1, harms + 1):
        fc = coef * fr * k
        for s in (+1, -1):
            fsb = fc + s * fr
            tol = max(2 * df, 0.02 * fr)
            lo, hi = np.searchsorted(freqs, fsb - tol), np.searchsorted(freqs, fsb + tol)
            if lo >= hi:
                continue
            e_band = sp[lo:hi].sum()
            w = max(6 * tol, 20.0)
            wlo, whi = np.searchsorted(freqs, max(fsb - w, 2.0)), np.searchsorted(freqs, fsb + w)
            mask = np.ones(whi - wlo, dtype=bool)
            mask[max(0, lo - wlo) : hi - wlo] = False
            seg = sp[wlo:whi][mask]
            # 同 family_prominence: 中位数 × 分布修正系数(默认 1/ln2, 可 auto)
            bg_med = np.median(seg) if len(seg) else 0.0
            corr_f = float(bg_corr)
            if bg_corr == "auto":
                corr_f = min(float(np.mean(seg)) / (bg_med + 1e-12), 1.4427) if len(seg) else 1.0
            bg = (bg_med * corr_f * (hi - lo)) + 1e-12
            p = e_band / bg
            proms.append(p)
            total += max(p - 1.0, 0.0)
    return {"score": total, "prom": [round(p, 2) for p in proms]}


def time_stats(x: np.ndarray) -> dict:
    """时域条件指标(全部无量纲)。"""
    xr = x - x.mean()
    rms = float(np.sqrt(np.mean(xr**2))) + 1e-12
    pk = float(np.abs(xr).max())
    ma = float(np.mean(np.abs(xr))) + 1e-12
    sq = float(np.sqrt(np.mean(xr**2)))
    return {
        "rms": round(rms, 6),
        "kurtosis": round(float(kurtosis(xr, fisher=False)), 3),  # 正常≈3, 冲击故障显著升高
        "skewness": round(float(skew(xr)), 3),
        "crest_factor": round(pk / rms, 3),  # 峰值因子
        "impulse_factor": round(pk / ma, 3),  # 脉冲因子
        "shape_factor": round(sq / ma, 3),  # 波形因子
        "margin_factor": round(pk / (np.mean(np.sqrt(np.abs(xr))) ** 2) + 1e-12, 3),
        "peak_to_peak": round(float(xr.max() - xr.min()), 6),
    }
