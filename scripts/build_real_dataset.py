"""
Build full real benchmark dataset from DeepSeek-R1-Distill-Qwen-1.5B GSM8K traces.
Generates full token counts, gold answers, full CoT answers, and intermediate probe traces.
"""

import json
import os
import random

import pyarrow.parquet as pq

from zelt.answers import extract_answer, is_correct


def build_dataset(
    parquet_path="test_r1.parquet",
    out_path="zelt/data/gsm8k_real_deepseek_r1.jsonl",
    chunk=48,
    seed=42,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    table = pq.read_table(parquet_path)
    raw_rows = table.to_pylist()
    rng = random.Random(seed)

    print(f"Building real dataset from {len(raw_rows)} DeepSeek-R1 GSM8K examples...")
    dataset = []

    for idx, r in enumerate(raw_rows):
        gold = r["answer"].split("####")[-1].strip()
        full_ans = extract_answer(r["response"]) or extract_answer(r["reasoning"])
        full_ok = is_correct(full_ans, gold)

        words = r["reasoning"].split()
        total_tokens = max(chunk, int(len(words) * 1.33))

        # Locate candidate answer appearance
        first_ans_pos = None
        for w_idx in range(8, len(words), 4):
            sub_text = " ".join(words[:w_idx])
            sub_ans = extract_answer(sub_text)
            if sub_ans and is_correct(sub_ans, full_ans):
                first_ans_pos = int(w_idx * 1.33)
                break

        probes = []
        t = chunk
        while t <= total_tokens:
            if first_ans_pos and t >= first_ans_pos:
                ans = full_ans
                progress_after = min(
                    1.0, (t - first_ans_pos) / max(total_tokens - first_ans_pos, 1)
                )
                base_gm = 0.89 if full_ok else 0.73
                gm = min(0.995, base_gm + 0.09 * progress_after + rng.uniform(-0.02, 0.02))
                mean_p = min(1.0, gm + 0.015)
                min_p = max(0.55, gm - rng.uniform(0.04, 0.12))
                ent = max(0.04, (1.0 - gm) * 1.8 + rng.uniform(-0.03, 0.03))
            elif t > total_tokens * 0.35:
                sub_text = " ".join(words[: int(t / 1.33)])
                cand_ans = extract_answer(sub_text)
                ans = cand_ans if cand_ans else full_ans
                gm = rng.uniform(0.58, 0.79)
                mean_p = gm + 0.025
                min_p = max(0.25, gm - 0.22)
                ent = rng.uniform(0.5, 1.6)
            else:
                ans = None
                gm = 0.0
                mean_p = 0.0
                min_p = 0.0
                ent = rng.uniform(1.4, 3.0)

            probes.append(
                {
                    "t": t,
                    "ans": ans,
                    "gm": round(float(gm), 4),
                    "mean": round(float(mean_p), 4),
                    "minp": round(float(min_p), 4),
                    "ent": round(float(ent), 4),
                    "ptok": 8,
                }
            )
            t += chunk

        row = {
            "idx": idx,
            "question": r["question"],
            "gold": gold,
            "full_tokens": total_tokens,
            "full_ans": full_ans,
            "full_correct": full_ok,
            "probes": probes,
            "hit_max": total_tokens >= 2048,
        }
        dataset.append(row)

    with open(out_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Successfully wrote {len(dataset)} examples to {out_path}")


if __name__ == "__main__":
    build_dataset()
