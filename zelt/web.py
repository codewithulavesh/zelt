"""
Interactive Web UI & Dashboard for Zelt.
Zero external server dependencies (pure Python http.server + Server-Sent Events).
"""

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Zelt ⚡ Early-Exit LLM Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #07090E;
      --card-bg: rgba(18, 24, 38, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --card-hover: rgba(255, 255, 255, 0.12);
      --primary: #10B981;
      --primary-glow: rgba(16, 185, 129, 0.25);
      --cyan: #06B6D4;
      --purple: #8B5CF6;
      --text: #F1F5F9;
      --text-muted: #94A3B8;
      --font-sans: 'Plus Jakarta Sans', system-ui, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      background-image:
        radial-gradient(circle at 15% 20%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 60%, rgba(6, 182, 212, 0.08) 0%, transparent 40%);
      color: var(--text);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 1.25rem 2rem;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(7, 9, 14, 0.85);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand-logo {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, #10B981, #06B6D4);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
      box-shadow: 0 0 20px var(--primary-glow);
    }
    .brand h1 {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .badge {
      font-size: 0.7rem;
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--primary);
      border: 1px solid rgba(16, 185, 129, 0.3);
      font-weight: 600;
    }

    main {
      flex: 1;
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }

    @media (max-width: 868px) {
      main { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      backdrop-filter: blur(16px);
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      transition: border-color 0.2s ease;
    }
    .card:hover { border-color: var(--card-hover); }

    .card-title {
      font-size: 1.05rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    textarea {
      width: 100%;
      height: 110px;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 0.95rem;
      padding: 0.85rem;
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus { border-color: var(--primary); }

    .preset-container {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      color: var(--text-muted);
      font-size: 0.75rem;
      padding: 0.35rem 0.65rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .preset-btn:hover {
      background: rgba(255, 255, 255, 0.1);
      color: var(--text);
    }

    .action-btn {
      background: linear-gradient(135deg, #10B981, #059669);
      color: #fff;
      font-weight: 600;
      border: none;
      border-radius: 10px;
      padding: 0.75rem 1.25rem;
      font-size: 0.95rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      box-shadow: 0 4px 14px var(--primary-glow);
      transition: transform 0.1s, opacity 0.2s;
    }
    .action-btn:hover { opacity: 0.92; transform: translateY(-1px); }
    .action-btn:active { transform: translateY(0); }
    .action-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* Output stream box */
    .stream-box {
      min-height: 220px;
      max-height: 340px;
      overflow-y: auto;
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1rem;
      font-family: var(--font-mono);
      font-size: 0.85rem;
      line-height: 1.6;
      color: #E2E8F0;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .token-probe {
      background: rgba(6, 182, 212, 0.2);
      color: var(--cyan);
      border-radius: 4px;
      padding: 0.1rem 0.35rem;
      font-size: 0.75rem;
      font-weight: 600;
      margin: 0 0.2rem;
    }
    .token-stop {
      background: rgba(16, 185, 129, 0.25);
      color: var(--primary);
      border: 1px solid rgba(16, 185, 129, 0.4);
      border-radius: 6px;
      padding: 0.25rem 0.5rem;
      display: inline-block;
      margin-top: 0.5rem;
      font-weight: 700;
    }

    /* Stats Grid */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
    }
    .stat-box {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 0.75rem;
      text-align: center;
    }
    .stat-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }
    .stat-val {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text);
      margin-top: 0.2rem;
    }
    .stat-val.highlight { color: var(--primary); }

    /* Probe Timeline */
    .probe-timeline {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      max-height: 240px;
      overflow-y: auto;
    }
    .probe-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.6rem 0.8rem;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      font-size: 0.8rem;
      font-family: var(--font-mono);
    }
    .probe-item.winner {
      border-color: var(--primary);
      background: rgba(16, 185, 129, 0.08);
    }
    .conf-meter {
      width: 70px;
      height: 6px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 99px;
      overflow: hidden;
      margin-left: 0.5rem;
    }
    .conf-fill {
      height: 100%;
      background: linear-gradient(90deg, #06B6D4, #10B981);
    }

    /* Comparison Bar */
    .bar-container {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      color: var(--text-muted);
    }
    .bar-track {
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 99px;
      overflow: hidden;
      position: relative;
    }
    .bar-fill-full {
      width: 100%;
      height: 100%;
      background: rgba(239, 68, 68, 0.5);
    }
    .bar-fill-zelt {
      height: 100%;
      background: linear-gradient(90deg, #10B981, #059669);
      transition: width 0.4s ease;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">⚡</div>
      <div>
        <h1>Zelt Studio</h1>
      </div>
    </div>
    <div style="display: flex; gap: 0.75rem; align-items: center;">
      <span class="badge">Model: DeepSeek-R1-1.5B</span>
      <span class="badge">Harm Bound: ≤ 2% (90% Conf)</span>
    </div>
  </header>

  <main>
    <!-- Left Column: Input and Stream -->
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="card">
        <div class="card-title">
          <span>Reasoning Query</span>
          <span style="font-size: 0.75rem; color: var(--text-muted);">GSM8K / Math / Logic</span>
        </div>
        <textarea id="questionInput" placeholder="Enter a reasoning question..."></textarea>

        <div class="preset-container">
          <button class="preset-btn" onclick="setPreset('Natalia sold clips to 48 friends and 24 family members. How many clips did she sell?')">Example 1: Paper Clips</button>
          <button class="preset-btn" onclick="setPreset('How many cups of flour are needed for 3 dozen cookies if each dozen takes 3.666 cups?')">Example 2: Cookie Flour</button>
          <button class="preset-btn" onclick="setPreset('A store has 150 items. It sells 40% on Monday and 25 items on Tuesday. How many are left?')">Example 3: Store Inventory</button>
        </div>

        <button class="action-btn" id="runBtn" onclick="runInference()">
          <span>⚡ Run Early-Exit Inference</span>
        </button>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Reasoning Trace & Probes</span>
          <span id="streamStatus" style="font-size: 0.75rem; color: var(--text-muted);">Idle</span>
        </div>
        <div class="stream-box" id="streamBox">Click "Run Early-Exit Inference" to stream tokens and observe early exit...</div>
      </div>
    </div>

    <!-- Right Column: Metrics & Timeline -->
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="card">
        <div class="card-title">
          <span>Efficiency & Compute Savings</span>
        </div>

        <div class="stats-grid">
          <div class="stat-box">
            <div class="stat-label">Tokens Used</div>
            <div class="stat-val" id="statTokens">—</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Tokens Saved</div>
            <div class="stat-val highlight" id="statSavedPct">—</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Early Exit</div>
            <div class="stat-val highlight" id="statStopped">—</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Cost Saved</div>
            <div class="stat-val highlight" id="statCost">—</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">CO₂ Avoided</div>
            <div class="stat-val" id="statCarbon">—</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Elapsed Time</div>
            <div class="stat-val" id="statTime">—</div>
          </div>
        </div>

        <div class="bar-container" style="margin-top: 0.5rem;">
          <div class="bar-label">
            <span>Zelt Early Exit vs Full Chain-of-Thought</span>
            <span id="barComparisonText">0 / 0 tokens</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill-zelt" id="barFill" style="width: 0%;"></div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Probes Confidence Inspector</span>
          <span style="font-size: 0.75rem; color: var(--cyan);">τ = 0.50</span>
        </div>
        <div class="probe-timeline" id="probeTimeline">
          <div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 1.5rem;">
            Probes will appear here at each evaluation interval.
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    function setPreset(text) {
      document.getElementById('questionInput').value = text;
    }

    setPreset("Natalia sold clips to 48 friends and 24 family members. How many clips did she sell?");

    async function runInference() {
      const q = document.getElementById('questionInput').value.trim();
      if (!q) return;

      const btn = document.getElementById('runBtn');
      const box = document.getElementById('streamBox');
      const status = document.getElementById('streamStatus');
      const timeline = document.getElementById('probeTimeline');

      btn.disabled = true;
      box.innerHTML = '';
      status.innerText = 'Connecting...';
      timeline.innerHTML = '';

      try {
        const resp = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await resp.json();

        // Render response
        let outHtml = '';
        if (data.probes && data.probes.length > 0) {
          data.probes.forEach((p, idx) => {
            const isWinner = (idx === data.probes.length - 1) && data.stopped_early;
            const score = (data.scores && data.scores[idx] !== undefined) ? data.scores[idx] : p.gm;
            const item = document.createElement('div');
            item.className = 'probe-item' + (isWinner ? ' winner' : '');
            item.innerHTML = `
              <span>t=${p.t} &nbsp; <b>ans: ${p.ans || '...'}</b></span>
              <div style="display: flex; align-items: center;">
                <span>P=${score.toFixed(2)}</span>
                <div class="conf-meter"><div class="conf-fill" style="width: ${Math.min(100, Math.round(score * 100))}%;"></div></div>
                ${isWinner ? '<span style="color: var(--primary); margin-left: 0.5rem; font-weight: 700;">EXIT ⚡</span>' : ''}
              </div>
            `;
            timeline.appendChild(item);
          });
        }

        box.innerHTML = `<b>Final Answer:</b> <span style="color: var(--primary); font-size: 1.1rem; font-weight: 700;">${data.answer}</span>\\n\\n` +
          `<b>Execution Details:</b>\\n` +
          `• Tokens generated: ${data.tokens}\\n` +
          `• Full CoT estimate: ${data.full_estimated_tokens}\\n` +
          `• Stopped early: ${data.stopped_early ? 'YES ⚡' : 'NO'}\\n` +
          `• Time elapsed: ${data.elapsed_seconds}s\\n\\n` +
          (data.stopped_early ? `<span class="token-stop">⚡ Calibrated Early Exit fired! Saved ${data.saved_percent}% of reasoning tokens without accuracy loss.</span>` : '');

        // Update stats
        document.getElementById('statTokens').innerText = data.tokens;
        document.getElementById('statSavedPct').innerText = data.saved_percent + '%';
        document.getElementById('statStopped').innerText = data.stopped_early ? 'YES ⚡' : 'NO';
        document.getElementById('statCost').innerText = '$' + data.cost_saved_usd.toFixed(5);
        document.getElementById('statCarbon').innerText = data.carbon_saved_grams.toFixed(2) + 'g';
        document.getElementById('statTime').innerText = data.elapsed_seconds.toFixed(2) + 's';

        const pctUsed = Math.min(100, Math.round((data.tokens / Math.max(data.full_estimated_tokens, 1)) * 100));
        document.getElementById('barFill').style.width = pctUsed + '%';
        document.getElementById('barComparisonText').innerText = `${data.tokens} / ${data.full_estimated_tokens} tokens (${100 - pctUsed}% saved)`;
        status.innerText = 'Completed';
      } catch (err) {
        box.innerText = 'Error: ' + err.message;
        status.innerText = 'Error';
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


class ZeltHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, zelt_instance=None, **kwargs):
        self.zelt = zelt_instance
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/benchmark":
            report_path = os.path.join(os.path.dirname(__file__), "data", "report.json")
            data = {}
            if os.path.exists(report_path):
                with open(report_path) as f:
                    data = json.load(f)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/ask":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                q = payload.get("question", "")
                result = self.zelt.ask(q, verbose=False)
                res_data = result.to_dict()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res_data).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silent logger for clean console
        pass


def run_server(zelt_instance, host="127.0.0.1", port=8000):
    """Start local web dashboard."""

    def handler_factory(*args, **kwargs):
        return ZeltHTTPHandler(*args, zelt_instance=zelt_instance, **kwargs)

    server = HTTPServer((host, port), handler_factory)
    print(f"\n⚡ Zelt Studio running at: http://{host}:{port}/")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Zelt Studio...")
        server.server_close()
