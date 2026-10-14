# Contributing to zelt

Thank you for your interest in contributing to **zelt** — calibrated, risk-controlled early exit for reasoning LLMs!

---

## Ways to Contribute

- 🐛 **Bug reports** — Open an issue with a minimal reproducible example
- 💡 **Feature requests** — Describe the use case and expected behavior
- 🔬 **Research contributions** — New datasets, baselines, or model evaluations
- 🛠️ **Code contributions** — Bug fixes, new adapters, feature implementations
- 📖 **Documentation** — Improve examples, clarify the math, fix typos

---

## Development Setup

```bash
# Clone the repo
git clone https://github.com/codewithulavesh/zelt.git
cd zelt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in editable mode with dev dependencies
pip install -e ".[full]"
pip install pytest pytest-cov ruff mypy

# Run tests
pytest tests/ -v --cov=zelt --cov-report=term-missing

# Run linting
ruff check zelt/
mypy zelt/ --ignore-missing-imports
```

---

## Testing

All contributions must include tests:

- **Unit tests**: `tests/` directory, using `pytest`
- **Coverage**: New code should maintain ≥ 85% coverage
- **No GPU required**: Tests use the `FallbackEngine` — no model weights needed

```bash
pytest tests/ -v --tb=short
```

---

## Code Style

- Format: `ruff format zelt/`
- Lint: `ruff check zelt/`
- Type hints: All public functions must have type annotations
- Docstrings: NumPy docstring style

---

## Research Contributions

If you have:
- New benchmark results on MATH-500, AIME, or other reasoning datasets
- Alternative decider architectures (e.g., LightGBM, neural probes)
- Cross-model experiments (Qwen, Llama, Mistral)

Please open an issue to discuss before opening a PR. We can work together to integrate it into the paper.

---

## Submitting a Pull Request

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit with descriptive messages following [Conventional Commits](https://www.conventionalcommits.org/)
4. Push and open a PR against `main`
5. Ensure all CI checks pass

---

## Code of Conduct

Be respectful and constructive. Discrimination, harassment, or hostility of any kind will not be tolerated.

---

## License

By contributing, you agree your contributions will be licensed under the [MIT License](LICENSE).
