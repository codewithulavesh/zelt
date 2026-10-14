# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | ✅ Active support  |
| 0.1.x   | ❌ No longer supported |

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, send an email to **ulaveshk2@gmail.com** with:

1. **Subject**: `[SECURITY] zelt — <short description>`
2. **Description**: What the vulnerability is and its potential impact
3. **Reproduction steps**: Minimal example to reproduce it
4. **Suggested fix** (optional but appreciated)

You can expect:
- **Acknowledgement** within 48 hours
- **Status update** within 7 days
- **Patch release** within 30 days for confirmed vulnerabilities

## Scope

This library runs **100% locally** — it does not make outbound network calls unless you explicitly configure an Ollama or vLLM backend URL. Security concerns most relevant to this project include:

- **Deserialization**: `StopDecider.load()` uses `joblib.load()`. Only load `.joblib` files from trusted sources.
- **FastAPI server**: When running `zelt serve`, restrict access to trusted networks. Do not expose to the public internet without authentication middleware.
- **Prompt injection**: If passing user-controlled prompts to the model, apply appropriate sanitization at the application layer.

## Acknowledgements

We thank security researchers who responsibly disclose vulnerabilities. Contributors who report valid security issues will be credited in the CHANGELOG (unless they prefer to remain anonymous).
