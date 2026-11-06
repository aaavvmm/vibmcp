# -*- coding: utf-8 -*-
"""核心功能测试: 特征频率公式 / 合成信号 ground truth / 诊断引擎准确率。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from vibmcp import synth
from vibmcp.analysis import envelope_spectrum, time_stats
from vibmcp.bearing import char_freqs
from vibmcp.diagnose import diagnose
from vibmcp.loaders import load_file, register, get


# ---------- 特征频率 ----------

def test_char_freqs_cwru():
    # CWRU 6205: BPFO=3.585, BPFI=5.415 (文献值)
    cf = char_freqs(machine="cwru")
    assert abs(cf["BPFO"]["coef"] - 3.585) < 0.01
    assert abs(cf["BPFI"]["coef"] - 5.415) < 0.01


def test_char_freqs_custom_geometry():
    cf = char_freqs(Z=8, d=6.0, D=30.0, fr=50.0)
    assert abs(cf["BPFI"]["coef"] * 50 - (4 * (1 + 0.2) * 50)) < 1e-6


# ---------- 合成信号 ----------

def test_synth_normal_vs_fault():
    x0, m0 = synth.synth_signal(None, seed=1)
    x1, m1 = synth.synth_signal("BPFI", seed=1)
    # 故障信号峭度应显著高于正常(冲击成分)
    assert time_stats(x1)["kurtosis"] > time_stats(x0)["kurtosis"]
    assert m1["fault"] == "BPFI"


# ---------- 诊断引擎 ----------

@pytest.mark.parametrize("fault", ["BPFI", "BPFO", "BSF", "FTF"])
def test_diagnose_accuracy(fault):
    # 高信噪比下每类 3 个种子全对(引擎定位: 物理证据充足时零误诊)
    for seed in range(3):
        x, m = synth.synth_signal(fault, seed=seed, snr_db=5)
        r = diagnose(x, m["fs"], m["fr"], machine=m["machine"])
        assert r["verdict"] == fault, f"seed={seed}: {r['verdict']} != {fault}"


def test_diagnose_low_snr_not_worse_than_chance():
    # 低信噪比(0dB)下整体准确率应显著高于随机(1/5): 允许个别弱信号漏检
    n_ok = n = 0
    for fault in ["BPFI", "BPFO", "BSF", "FTF"]:
        for seed in range(4):
            x, m = synth.synth_signal(fault, seed=seed, snr_db=0)
            r = diagnose(x, m["fs"], m["fr"], machine=m["machine"])
            n_ok += r["verdict"] == fault
            n += 1
    assert n_ok / n >= 0.7, f"0dB 准确率 {n_ok}/{n} 过低"


def test_diagnose_normal():
    x, m = synth.synth_signal(None, seed=0)
    r = diagnose(x, m["fs"], m["fr"], machine=m["machine"])
    assert r["verdict"] == "normal"


# ---------- 加载与注册表 ----------

def test_register_and_get():
    x = np.random.default_rng(0).normal(size=1000)
    sid = register(x, {"fs": 12000.0, "fr": 29.95})
    rec = get(sid)
    assert len(rec["x"]) == 1000


def test_get_missing_raises():
    with pytest.raises(KeyError):
        get("nope")


def test_load_npz_roundtrip(tmp_path):
    x, m = synth.synth_signal("BPFO", seed=7)
    p = tmp_path / "t.npz"
    np.savez(p, x=x, fs=m["fs"], fr=m["fr"])
    d = load_file(str(p))
    assert d["meta"]["fs"] == m["fs"]
    assert abs(d["meta"]["fr"] - m["fr"]) < 1e-9
