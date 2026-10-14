import os
import tempfile

import pytest

from zelt.decider import StopDecider, clopper_pearson_ucb, cp_upper
from zelt.synthetic import make_synthetic


def test_clopper_pearson_bound():
    # 0 harmful exits out of 331 at 90% confidence
    ucb_zero = cp_upper(0, 331, delta=0.10)
    assert ucb_zero < 0.01  # approximately 0.00694
    assert ucb_zero > 0.005

    # k = n returns 1.0
    assert cp_upper(10, 10, delta=0.10) == 1.0

    # Alias check
    assert clopper_pearson_ucb(0, 331, 0.10) == ucb_zero


def test_stop_decider_fit_and_save():
    data = make_synthetic(n=50, seed=42)
    dec = StopDecider(kind="logreg")
    dec.fit(data)

    probes = data[0]["probes"]
    scores = dec.score(probes)
    assert len(scores) == len(probes)
    assert all(0.0 <= s <= 1.0 for s in scores)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "decider.joblib")
        dec.save(path)
        assert os.path.exists(path)

        loaded = StopDecider.load(path)
        loaded_scores = loaded.score(probes)
        assert list(scores) == pytest.approx(list(loaded_scores))
