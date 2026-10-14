import pytest

from zelt.decider import FEATURES, featurize


def test_featurize_empty():
    feats = featurize([])
    assert feats.shape == (0, len(FEATURES))


def test_featurize_single_probe():
    probes = [
        {"t": 48, "ans": "42", "gm": 0.85, "mean": 0.88, "minp": 0.70, "ent": 0.40, "ptok": 8}
    ]
    feats = featurize(probes)
    assert feats.shape == (1, len(FEATURES))
    assert feats[0, FEATURES.index("gm")] == pytest.approx(0.85)
    assert feats[0, FEATURES.index("stable")] == 0.0  # first probe has no previous probe
    assert feats[0, FEATURES.index("run")] == 0.0


def test_featurize_consecutive_probes():
    probes = [
        {"t": 48, "ans": "42", "gm": 0.70, "mean": 0.75, "minp": 0.50, "ent": 1.2, "ptok": 8},
        {"t": 96, "ans": "42", "gm": 0.90, "mean": 0.92, "minp": 0.80, "ent": 0.3, "ptok": 8},
        {"t": 144, "ans": "42", "gm": 0.98, "mean": 0.99, "minp": 0.95, "ent": 0.05, "ptok": 8},
    ]
    feats = featurize(probes)
    assert feats.shape == (3, len(FEATURES))

    # At step 2, answer was identical to step 1
    assert feats[1, FEATURES.index("stable")] == 1.0
    assert feats[1, FEATURES.index("run")] == 1.0
    # At step 3, consecutive repetition run is 2
    assert feats[2, FEATURES.index("run")] == 2.0
    # First difference in gm: 0.90 - 0.70 = 0.20
    assert feats[1, FEATURES.index("dgm")] == pytest.approx(0.20)
