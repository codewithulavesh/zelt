"""Answer extraction + correctness checking (GSM8K numeric and MATH-style strings)."""

import re

_NUM = re.compile(r"-?\d+\.?\d*")


def last_boxed(text):
    if not text or not isinstance(text, str):
        return None
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    i += len("\\boxed{")
    depth, out = 1, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return "".join(out) or None


def extract_answer(text):
    """Prefer \\boxed{...} or <answer>...</answer>; fall back to the last number in the text."""
    if not text or not isinstance(text, str):
        return None
    b = last_boxed(text)
    if b is not None:
        return b
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    raw_nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    nums = [n.rstrip(".") for n in raw_nums if n.rstrip(".")]
    return nums[-1] if nums else None


def _clean(s):
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\[,!; ]", "", s)
    s = s.replace("$", "").replace("\\%", "").replace("%", "").replace(",", "")
    return s.replace(" ", "").strip().rstrip(".")


def _as_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    a, b = _clean(str(pred)), _clean(str(gold))
    if a == b:
        return True
    fb = _as_float(b)
    if fb is not None:  # numeric gold: compare the first number in the prediction
        fa = _as_float(a)
        if fa is None:
            m = _NUM.search(a)
            fa = _as_float(m.group()) if m else None
        return fa is not None and abs(fa - fb) < 1e-6
    return False


def same_answer(a, b):
    return a is not None and b is not None and is_correct(a, b)
