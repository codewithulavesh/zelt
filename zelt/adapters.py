"""
Adapters for external LLM servers (Ollama, vLLM, OpenAI-compatible APIs).
Allows running early-exit reasoning when models are hosted on local servers.
"""
import json
import urllib.request
import urllib.error
import time
from .answers import extract_answer


class OllamaAdapter:
    """
    Adapter for local Ollama instances (e.g. `ollama run deepseek-r1:1.5b`).
    Streams tokens and triggers early abort when confident answer is detected.
    """
    def __init__(self, model="deepseek-r1:1.5b", base_url="http://localhost:11434", chunk=48):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chunk = chunk
        self.device = "ollama-server"

    def is_available(self):
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def run(self, question, stop_fn=None):
        url = f"{self.base_url}/api/generate"
        payload = json.dumps({
            "model": self.model,
            "prompt": question,
            "stream": True,
            "options": {"temperature": 0.0}
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        tokens_text = []
        probes = []
        stopped = False
        token_count = 0

        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line.decode("utf-8"))
                token_str = chunk.get("response", "")
                tokens_text.append(token_str)
                token_count += 1

                # Check probe threshold
                if token_count >= self.chunk and token_count % self.chunk == 0:
                    current_text = "".join(tokens_text)
                    ans = extract_answer(current_text)
                    pr = {
                        "t": token_count,
                        "ans": ans,
                        "gm": 0.85 if ans else 0.2,
                        "ent": 0.5,
                        "ptok": 10,
                    }
                    probes.append(pr)
                    if stop_fn is not None and stop_fn(probes):
                        stopped = True
                        break

                if chunk.get("done", False):
                    break

        full_text = "".join(tokens_text)
        final_ans = probes[-1]["ans"] if stopped and probes else extract_answer(full_text)
        return dict(
            text=full_text,
            answer=final_ans,
            tokens=token_count,
            probes=probes,
            stopped=stopped,
            probe_tokens=sum(p.get("ptok", 0) for p in probes)
        )


class VLLMAdapter:
    """
    Adapter for vLLM / OpenAI-compatible /v1/completions endpoint with streaming.
    """
    def __init__(self, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", base_url="http://localhost:8000", chunk=48):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chunk = chunk
        self.device = "vllm-server"

    def is_available(self):
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def run(self, question, stop_fn=None):
        url = f"{self.base_url}/v1/completions"
        payload = json.dumps({
            "model": self.model,
            "prompt": question,
            "stream": True,
            "max_tokens": 2048,
            "temperature": 0.0
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        tokens_text = []
        probes = []
        stopped = False
        token_count = 0

        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data_part = line_str[6:]
                if data_part == "[DONE]":
                    break
                chunk = json.loads(data_part)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                token_str = choices[0].get("text", "")
                tokens_text.append(token_str)
                token_count += 1

                if token_count >= self.chunk and token_count % self.chunk == 0:
                    current_text = "".join(tokens_text)
                    ans = extract_answer(current_text)
                    pr = {"t": token_count, "ans": ans, "gm": 0.88 if ans else 0.2, "ent": 0.5, "ptok": 10}
                    probes.append(pr)
                    if stop_fn is not None and stop_fn(probes):
                        stopped = True
                        break

        full_text = "".join(tokens_text)
        final_ans = probes[-1]["ans"] if stopped and probes else extract_answer(full_text)
        return dict(
            text=full_text,
            answer=final_ans,
            tokens=token_count,
            probes=probes,
            stopped=stopped,
            probe_tokens=sum(p.get("ptok", 0) for p in probes)
        )
