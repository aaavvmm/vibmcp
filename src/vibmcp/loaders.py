"""
信号加载与会话注册表
====================
支持格式: .npz (键 x/fs/fr) / .mat (CWRU 风格, 自动选 DE 通道) / .csv / .wav (单通道)。
MCP 会话内加载的信号存入内存注册表, 后续工具用 sig_id 引用 —— 这是
面向 Agent 编排的关键设计: 中间状态驻留服务端, LLM 只传轻量句柄。
"""

import json
import os
import wave

import numpy as np

# 会话内信号注册表: sig_id → {"x": 波形, "meta": 元数据, "label": 可选真值}
_REGISTRY: dict = {}
_COUNTER = [0]
# 供 Agent 查看的历史信息(报告用)
LAST_DIAG: dict = {}


def register(x: np.ndarray, meta: dict, label: str | None = None) -> str:
    """注册一个信号, 返回 sig_id (s001, s002 ...)。"""
    _COUNTER[0] += 1
    sid = f"s{_COUNTER[0]:03d}"
    _REGISTRY[sid] = {"x": np.asarray(x, dtype=np.float64), "meta": meta, "label": label}
    return sid


def get(sid: str) -> dict:
    if sid not in _REGISTRY:
        raise KeyError(f"sig_id '{sid}' 不存在, 先调用 load_signal。已加载: {list(_REGISTRY)}")
    return _REGISTRY[sid]


def list_ids() -> list[dict]:
    return [
        {"sig_id": sid, **{k: v["meta"].get(k) for k in ("fs", "fr", "fault", "source")}}
        for sid, v in _REGISTRY.items()
    ]


def _read_mat(path: str) -> tuple[np.ndarray, float | None]:
    """CWRU .mat: 变量名如 X105_DE_time, 取 DE(驱动端)通道, 无 DE 取 FE。"""
    from scipy.io import loadmat

    md = loadmat(path)
    keys = [k for k in md if not k.startswith("__")]
    de = [k for k in keys if "_DE_time" in k]
    fe = [k for k in keys if "_FE_time" in k]
    if de:
        return np.ravel(md[de[0]]), None
    if fe:
        return np.ravel(md[fe[0]]), None
    # 退化: 取第一个数值数组
    for k in keys:
        arr = np.asarray(md[k])
        if arr.dtype.kind in "fc" and arr.ndim >= 1:
            return np.ravel(arr).astype(np.float64), None
    raise ValueError(f"{path}: 未找到可用信号通道, 变量: {keys}")


def load_file(path: str, fs: float | None = None, fr: float | None = None) -> dict:
    """从文件加载信号。返回 {x, meta}; fs/fr 可显式覆盖。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    meta = {"source": path, "fs": fs, "fr": fr}

    if ext == ".npz":
        d = np.load(path, allow_pickle=False)
        x = np.ravel(d["x"]).astype(np.float64)
        meta["fs"] = fs or float(d["fs"]) if "fs" in d else fs
        if "fr" in d:
            meta["fr"] = fr or float(d["fr"])
    elif ext == ".mat":
        x, fs_mat = _read_mat(path)
        meta["fs"] = fs or fs_mat or 12000.0  # CWRU DE 默认 12kHz
    elif ext == ".csv":
        import csv

        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        col = 0
        # 跳过表头(首行非数字则丢弃)与可能的索引列
        start = 1 if rows and not _is_num(rows[0][0]) else 0
        if rows and start and len(rows[0]) > 1 and not _is_num(rows[0][1]):
            col = 1
        x = np.array([float(r[col]) for r in rows[start:] if r])
    elif ext == ".wav":
        with wave.open(path, "rb") as w:
            meta["fs"] = fs or w.getframerate()
            raw = w.readframes(w.getnframes())
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    else:
        raise ValueError(f"不支持的格式 '{ext}' (支持 .npz/.mat/.csv/.wav)")

    meta["fs"] = float(meta["fs"] or 12000.0)
    meta["fr"] = float(meta["fr"]) if meta.get("fr") else None  # type: ignore[arg-type]
    meta["duration_s"] = round(len(x) / float(meta["fs"]), 3)  # type: ignore[arg-type]
    meta["n"] = len(x)
    return {"x": x, "meta": meta}


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def load_json_list(json_str: str, fs: float, fr: float | None = None) -> dict:
    """从 JSON 数组文本加载(允许 Agent 直接内联传递短信号)。"""
    x = np.asarray(json.loads(json_str), dtype=np.float64)
    meta = {
        "source": "inline",
        "fs": float(fs),
        "fr": float(fr) if fr else None,
        "n": len(x),
        "duration_s": round(len(x) / fs, 3),
    }
    return {"x": x, "meta": meta}
