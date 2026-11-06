# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与语义化版本。

## [0.2.0] - 2026-11-06

### Added
- CI 工作流: ruff lint/format + mypy + pytest(核心算法层覆盖率门槛 80%)
- 结构化工具调用日志(stderr, JSON 行) —— stdout 是 MCP JSON-RPC 通道不得污染
- ADR 决策记录 ×3(证据融合/谱平均/FFT 取舍)
- harness/loaders 测试补齐(0%→覆盖), 测试 18 个, 核心层覆盖率 89%
- Dependabot / Makefile / MIT LICENSE

## [0.1.0] - 2026-10-12

### Added
- 11 个 MCP 工具(MCP SDK 2.x), 物理证据融合诊断引擎, 自适应诊断 harness
- 合成信号自闭环评测: 90/85/75/48% @ +5/0/-5/-10dB, harness 极端噪声 +5pp
- 单文件 HTML 诊断报告
