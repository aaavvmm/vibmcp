"""
HTML 诊断报告(应用外壳)
========================
把诊断证据链渲染成单文件 HTML(图片 base64 内嵌, 无外部依赖, 可直接发给别人)。
这是"既能演示又有深度"的落地物: 打开报告, 每条结论都有谱图和数字支撑。
"""

import datetime
import html
import os

FAULT_CN = {
    "BPFI": "内圈故障",
    "BPFO": "外圈故障",
    "BSF": "滚动体故障",
    "FTF": "保持架故障",
    "normal": "正常",
}

_CSS = """
body{font-family:"Microsoft YaHei",-apple-system,sans-serif;max-width:900px;margin:24px auto;
     padding:0 20px;color:#2c3e50;line-height:1.6}
h1{font-size:22px;border-bottom:2px solid #3498db;padding-bottom:8px}
.verdict{font-size:34px;font-weight:700;margin:16px 0;
     padding:16px 20px;border-radius:8px}
.verdict.bad{background:#fdecea;color:#c0392b;border-left:6px solid #e74c3c}
.verdict.ok{background:#eafaf1;color:#1e8449;border-left:6px solid #2ecc71}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th,td{border:1px solid #d5dbdb;padding:6px 10px;text-align:left}
th{background:#f4f6f7}
.note{background:#fef9e7;border-left:4px solid #f39c12;
     padding:8px 14px;margin:10px 0;font-size:13px}
img{max-width:100%;border:1px solid #eee;border-radius:4px;margin:8px 0}
.meta{color:#7f8c8d;font-size:12px}
.evidence td:first-child{font-weight:600}
.harm{color:#7f8c8d;font-size:11px}
"""


def render(result: dict, png_wave: str, png_spec: str, sig_meta: dict) -> str:
    """诊断结果 → HTML 字符串。result 为 diagnose() 输出。"""
    verdict = result["verdict"]
    is_bad = verdict != "normal"
    ev_rows = []
    for fault, ev in result["evidence"].items():
        proms = " ".join(
            f"<span class='harm'>k{k + 1}:{v}</span>" for k, v in enumerate(ev["prominence"])
        )
        sb = f"<td>{ev.get('sideband_score', '—')}</td>" if fault == "BPFI" else "<td>—</td>"
        ev_rows.append(
            f"<tr><td>{fault}<br><span class='harm'>{ev['char_freq_hz']} Hz</span></td>"
            f"<td>{ev['family_score']:.2f}<br>{proms}</td>{sb}</tr>"
        )
    ts_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in result["time_stats"].items())
    notes = "".join(f"<div class='note'>📌 {html.escape(n)}</div>" for n in result["notes"])
    geo = result["geometry"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>振动诊断报告 · {FAULT_CN[verdict]}</title><style>{_CSS}</style></head><body>
<h1>轴承振动诊断报告</h1>
<p class="meta">生成时间 {now} | 几何: {html.escape(str(geo.get("desc", "")))} |
Z={geo.get("Z")} d={geo.get("d")}mm D={geo.get("D")}mm | 转频 {result["shaft_freq_hz"]} Hz |
信号: {html.escape(str(sig_meta.get("source", "")))}
({sig_meta.get("duration_s", "?")}s @ {sig_meta.get("fs")}Hz)</p>

<div class="verdict {"bad" if is_bad else "ok"}">
{FAULT_CN[verdict]} <span style="font-size:18px">置信度 {result["confidence"]:.0%}</span>
</div>
<p>次优假设: {result["runner_up"]["fault"]} (得分 {result["runner_up"]["score"]})</p>

<h2>证据链</h2>
<table class="evidence">
<tr><th>故障类型 (特征频率)</th><th>谐波族能量占比 (k=1..3)</th><th>边带能量</th></tr>
{"".join(ev_rows)}</table>
{notes}

<h2>图谱</h2>
<img src="data:image/png;base64,{png_wave}" alt="波形">
<img src="data:image/png;base64,{png_spec}" alt="包络谱">

<h2>时域指标</h2>
<table>{ts_rows}</table>

<p class="meta">VibMCP · 物理证据融合诊断, 结论可由证据链逐谐波溯源</p>
</body></html>"""


def save(result: dict, png_wave: str, png_spec: str, sig_meta: dict, out_dir: str) -> str:
    """渲染并写盘, 返回 HTML 路径。文件名含结论与微秒时间戳避免覆盖。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"report_{result['verdict']}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}.html"
    )
    doc = render(result, png_wave, png_spec, sig_meta)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
