"""Forward from zelt.decider for backward compatibility."""

from zelt.decider import (
    FEATURES,
    StopDecider,
    calibration_metrics,
    choose_tau,
    clopper_pearson_ucb,
    cp_upper,
    evaluate_policy,
    featurize,
    fit_and_report,
    make_stop_fn,
    simulate,
    summarize,
)

__all__ = [
    "FEATURES",
    "StopDecider",
    "calibration_metrics",
    "choose_tau",
    "clopper_pearson_ucb",
    "cp_upper",
    "evaluate_policy",
    "featurize",
    "fit_and_report",
    "make_stop_fn",
    "simulate",
    "summarize",
]
