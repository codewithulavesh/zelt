# Changelog

All notable changes to **zelt** (zelt-ai on PyPI) are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Multi-model ensemble early exit with cross-model consensus
- MATH-500 and AIME pretrained decider checkpoints
- OpenAI-compatible `/v1/chat/completions` server adapter
- WebSocket streaming support for real-time dashboards

---

## [0.2.1] — 2026-10-14

### Added
- `zelt.from_ollama(model)` — one-liner factory to connect to local Ollama server
- `zelt.from_vllm(model, url)` — one-liner factory to connect to vLLM endpoint
- `zelt.token_budget(max_tokens)` — budget-based early-stop wrapper
- `zelt.risk_report()` — certified Clopper-Pearson risk audit without model inference
- `py.typed` marker (PEP 561) — full mypy / pyright support
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`
- `conftest.py` with shared pytest fixtures
- GitHub Actions publish-to-PyPI workflow (`publish.yml`)
- `python -m zelt` full entrypoint

### Changed
- All `print()` calls in `api.py` replaced with `logging` (respects log level)
- CLI `verbose` flag properly propagates to logging level

### Fixed
- `OllamaAdapter` timeout hardcoded at 120s — now configurable
- `BatchResult.speedup` divide-by-zero guard on empty batch

---

## [0.2.0] — 2026-10-13

### Added
- Full production REST microservice (`zelt serve`) via FastAPI + Uvicorn
- Web Studio dashboard (`zelt ui`)
- `zelt compare`, `zelt eval`, `zelt batch` CLI commands
- `BatchResult.export_prometheus()` Prometheus metrics endpoint
- CO2 and cost savings calculators
- `ZeltStoppingCriteria` HuggingFace adapter
- Ollama and vLLM adapters
- v2 feature set: 18 signals
- Adaptive probe interval and consensus-k stopping

### Changed
- Package renamed from `earlystop` to `zelt-ai` on PyPI
- Bundled decider retrained on 1,319 real GSM8K trajectories (tau=0.50)

---

## [0.1.0] — 2026-10-01

### Added
- Initial public release
- Core `EarlyStopper` with KV-cache probe
- `StopDecider` with Clopper-Pearson calibration
- MIT License

---

[Unreleased]: https://github.com/codewithulavesh/zelt/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/codewithulavesh/zelt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/codewithulavesh/zelt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/codewithulavesh/zelt/releases/tag/v0.1.0
