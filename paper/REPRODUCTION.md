# Reproduction README
# Zelt: Calibrated, Risk-Controlled Early Exit for Reasoning LLMs

## Minimal reproduction (paper Table 1 + Table 2)

```bash
pip install zelt           # installs bundled data + pretrained decider

python - <<'EOF'
from zelt.decider import load_rows, fit_and_report
import os, zelt

data = os.path.join(os.path.dirname(zelt.__file__), 'data',
                    'gsm8k_real_deepseek_r1.jsonl')
rows = load_rows(data)
fit_and_report(rows, 'results', alpha=0.02, delta=0.1, seed=0, verbose=True)
EOF
```

Expected output (matches paper):
```
method               tau/B     acc    dAcc   tokens  reduce%  w/probe%  stop%  harm%
full_cot                -    0.782  +0.000    750.7      0.0             0.0    0.0
fixed_budget          1632   0.767  -0.015    507.7     32.4      21.5  15.1    1.5
confidence            0.79   0.782  +0.000    430.6     42.6      33.4  40.2    0.0
learned               0.50   0.782  +0.000    465.9     37.9      27.9  37.8    0.0
learned_naive_0.9     0.90   0.782  +0.000    466.4     37.9      27.9  37.2    0.0
```

## Regenerate paper figures

```bash
python paper/generate_figures.py
# Writes: paper/figures/token_distribution.png
#         paper/figures/comparison_bars.png
```

## Compile paper (requires LaTeX)

```bash
brew install basictex   # macOS
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Environment

```
Python  3.12.14
numpy   2.5.4
scipy   1.18.1
scikit-learn 1.9.1
joblib  1.6.0
matplotlib 3.11.2
```

All results are deterministic given seed=0.

## Hardware

Calibration and simulation run on CPU (no GPU required).
Trajectory collection (not re-runnable from this repo without the model)
used DeepSeek-R1-Distill-Qwen-1.5B loaded via HuggingFace Transformers.
