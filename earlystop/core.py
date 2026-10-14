"""
Model-side engine: greedy decoding with a KV cache, forked "final answer" probes,
and live early exit. The probe never disturbs the reasoning trajectory (the cache
is cropped back after every probe), so with greedy decoding the early-exit run is
token-for-token identical to the full run up to the stopping point.
"""
import math
from .answers import extract_answer

PROBE_TEXT = "\n</think>\n\n**Final Answer**\n\\boxed{"
PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


class EarlyStopper:
    def __init__(self, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device=None,
                 chunk=48, min_tokens=48, max_new=2048, probe_max=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        dtype = torch.bfloat16 if self.device == "cuda" else (torch.float16 if self.device == "mps" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(self.device).eval()
        self.eos = self.tok.eos_token_id
        self.chunk, self.min_tokens, self.max_new, self.probe_max = chunk, min_tokens, max_new, probe_max
        self.probe_ids = self.tok(PROBE_TEXT, add_special_tokens=False,
                                  return_tensors="pt").input_ids.to(self.device)

    # -------------------------------------------------------------- internals
    def _encode(self, question):
        msgs = [{"role": "user", "content": question + PROMPT_SUFFIX}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not self.torch.is_tensor(ids):
            ids = ids["input_ids"]
        return ids.to(self.device)

    def _fwd(self, ids, cache):
        with self.torch.no_grad():
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        return out.logits[:, -1].float()

    def probe(self, cache):
        """Force a final answer on the current cache; restore the cache afterwards."""
        torch = self.torch
        L = cache.get_seq_length()
        logits = self._fwd(self.probe_ids, cache)
        probs, text, depth, done, first_ent = [], "", 1, False, None
        plen = self.probe_ids.shape[1]
        for step in range(self.probe_max):
            p = torch.softmax(logits[0], -1)
            if step == 0:
                first_ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
            pmax, t = p.max(-1)
            plen += 1
            if int(t) == self.eos:
                break
            probs.append(float(pmax))
            for ch in self.tok.decode([int(t)]):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        done = True
                        break
                text += ch
            if done:
                break
            logits = self._fwd(t.view(1, 1), cache)
        cache.crop(L)
        if not done or not probs or not text.strip():
            return dict(ans=None, gm=0.0, mean=0.0, minp=0.0, ent=first_ent or 0.0, ptok=plen)
        lp = [math.log(max(x, 1e-12)) for x in probs]
        return dict(ans=text.strip(), gm=math.exp(sum(lp) / len(lp)), mean=sum(probs) / len(probs),
                    minp=min(probs), ent=first_ent, ptok=plen)

    # -------------------------------------------------------------- main loop
    def run(self, question, stop_fn=None, adaptive=True):
        """
        Greedy decode with probes every `chunk` tokens.
        stop_fn(probes) -> bool decides early exit; None = run to completion (recording mode).
        adaptive: halve probe interval once a confident candidate answer is found.
        """
        from transformers import DynamicCache
        torch = self.torch
        cache = DynamicCache()
        logits = self._fwd(self._encode(question), cache)
        gen, probes, stopped = [], [], False
        next_probe = self.chunk  # when to do the next probe
        while len(gen) < self.max_new:
            t = int(logits.argmax(-1))
            gen.append(t)
            if t == self.eos:
                break
            logits = self._fwd(torch.tensor([[t]], device=self.device), cache)
            if len(gen) >= self.min_tokens and len(gen) >= next_probe:
                pr = self.probe(cache)
                pr["t"] = len(gen)
                probes.append(pr)
                if stop_fn is not None and stop_fn(probes):
                    stopped = True
                    break
                # Adaptive interval: once we have a confident candidate, probe faster
                if adaptive and pr.get("ans") is not None and pr.get("gm", 0) >= 0.6:
                    next_probe = len(gen) + max(self.chunk // 2, 16)
                else:
                    next_probe = len(gen) + self.chunk
        text = self.tok.decode(gen, skip_special_tokens=True)
        if stopped:
            ans = probes[-1]["ans"]
        else:
            ans = extract_answer(text.split("</think>")[-1] if "</think>" in text else text)
            if ans is None and probes:
                ans = probes[-1]["ans"]
        return dict(text=text, answer=ans, tokens=len(gen), probes=probes, stopped=stopped,
                    probe_tokens=sum(p["ptok"] for p in probes))

    def generate(self, question, decider, tau=None, consensus_k=None, adaptive=True):
        """Live early-exit generation using a trained StopDecider."""
        from .decider import make_stop_fn
        return self.run(question, stop_fn=make_stop_fn(decider, tau, consensus_k), adaptive=adaptive)

    def run_stream(self, question, stop_fn=None, adaptive=True):
        """
        Yields live events during greedy decoding:
          {"type": "token", "text": "...", "t": 12}
          {"type": "probe", "probe": {...}, "t": 48}
          {"type": "result", "answer": "...", "tokens": 144, "stopped": True, ...}
        """
        from transformers import DynamicCache
        torch = self.torch
        cache = DynamicCache()
        logits = self._fwd(self._encode(question), cache)
        gen, probes, stopped = [], [], False
        next_probe = self.chunk
        while len(gen) < self.max_new:
            t = int(logits.argmax(-1))
            gen.append(t)
            token_str = self.tok.decode([t], skip_special_tokens=False)
            yield {"type": "token", "text": token_str, "t": len(gen)}
            if t == self.eos:
                break
            logits = self._fwd(torch.tensor([[t]], device=self.device), cache)
            if len(gen) >= self.min_tokens and len(gen) >= next_probe:
                pr = self.probe(cache)
                pr["t"] = len(gen)
                probes.append(pr)
                yield {"type": "probe", "probe": pr, "t": len(gen)}
                if stop_fn is not None and stop_fn(probes):
                    stopped = True
                    break
                if adaptive and pr.get("ans") is not None and pr.get("gm", 0) >= 0.6:
                    next_probe = len(gen) + max(self.chunk // 2, 16)
                else:
                    next_probe = len(gen) + self.chunk

        full_text = self.tok.decode(gen, skip_special_tokens=True)
        if stopped:
            ans = probes[-1]["ans"]
        else:
            ans = extract_answer(full_text.split("</think>")[-1] if "</think>" in full_text else full_text)
            if ans is None and probes:
                ans = probes[-1]["ans"]

        yield {
            "type": "result",
            "text": full_text,
            "answer": ans,
            "tokens": len(gen),
            "probes": probes,
            "stopped": stopped,
            "probe_tokens": sum(p["ptok"] for p in probes),
        }
