"""Checks on the parts that decide whether the numbers mean anything: the two-day
cut-off, the label, the feature sets, the splits and the baseline.

    python3 -m pytest -q tests
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import models, config                                                  # noqa: E402

TAB = pd.read_csv(os.path.join(config.PROC, "events.csv.gz"), index_col=0)
RAW = os.path.join(config.DATA, "Collision Avoidance Challenge - Dataset", "raw_data", "raw_data_2015-2019.gz")


def test_table_counts_match_the_paper():
    assert len(TAB) == 10460 and TAB.high_risk.sum() == 216
    assert TAB.esa_test.sum() == 2167 and TAB.loc[TAB.esa_test == 1, "high_risk"].sum() == 150
    assert (TAB.esa_test == 0).sum() == 8293


def test_no_feature_comes_from_inside_the_last_two_days():
    assert (TAB.time_to_tca >= config.CUTOFF_DAYS).all()
    assert (TAB.last_ttca < config.FINAL_WITHIN_DAYS).all()
    raw = pd.read_csv(RAW)
    rng = np.random.default_rng(0)
    for ev in rng.choice(TAB.index.values, 40, replace=False):
        e = raw[raw.event_id == ev].sort_values("time_to_tca", ascending=False)
        use = e[e.time_to_tca >= config.CUTOFF_DAYS]
        row = TAB.loc[ev]
        assert np.isclose(row.risk, use.risk.iloc[-1])                  # latest usable CDM
        assert np.isclose(row.final_risk, e.risk.iloc[-1])               # label from the last CDM
        assert row.n_cdms_used == len(use)
        assert np.isclose(row.risk_first, use.risk.iloc[0]) and np.isclose(row.risk_max, use.risk.max())
        assert np.isclose(row.miss_distance_mean, use.miss_distance.mean())


def test_label_columns_never_reach_the_features():
    for fs in ("latest", "history", "no_risk"):
        num, cat = models.feature_lists(fs, TAB)
        for bad in ("final_risk", "high_risk", "esa_test", "lrp_high", "mission_id", "last_ttca", "n_cdms_total"):
            assert bad not in num + cat
        assert cat == ["c_object_type"]
    num, _ = models.feature_lists("no_risk", TAB)
    assert not any(models.base_name(c) in models.RISK_COLS for c in num)
    assert "mahalanobis_distance" in num                               # kept on purpose
    assert len(models.feature_lists("latest", TAB)[0]) == 100
    assert len(models.feature_lists("history", TAB)[0]) == 166


def test_naive_forecast_reproduces_the_paper():
    te = TAB[TAB.esa_test == 1]
    f2 = fbeta_score(te.high_risk, (te.risk >= -6).astype(int), beta=2)
    assert abs(f2 - 0.739) < 0.001                                     # Uriot et al. 2021, Table 3
    f2_all = fbeta_score(TAB.high_risk, (TAB.risk >= -6).astype(int), beta=2)
    assert abs(f2_all - 0.600) < 0.001                                 # their 505-split mean was 0.5997


def test_splits_do_not_leak_events():
    for split in ("cv", "mission"):
        folds = models.make_folds(TAB, split)
        for tr, te in folds:
            assert len(np.intersect1d(tr, te)) == 0
            assert len(tr) + len(te) == len(TAB)
        if split == "mission":
            for tr, te in folds:
                assert not set(TAB.mission_id.values[tr]) & set(TAB.mission_id.values[te])
    cv = models.make_folds(TAB, "cv", "history")
    assert len(cv) == models.N_FOLDS * models.N_REPEATS_CV
    assert len(models.make_folds(TAB, "cv", "latest")) == models.N_FOLDS
    assert [list(te) for _, te in models.make_folds(TAB, "cv", "latest")] == [list(te) for _, te in cv[:5]]
    for r in range(models.N_REPEATS_CV):                               # each repeat covers every event once
        tests = np.concatenate([te for _, te in cv[r * 5:(r + 1) * 5]])
        assert len(tests) == len(TAB) and len(np.unique(tests)) == len(TAB)
    tr, te = models.make_folds(TAB, "official")[0]
    assert len(te) == 2167 and TAB.high_risk.values[te].sum() == 150


def test_threshold_helpers():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    s = np.array([.1, .2, .3, .9, .8, .35, .7, .05, .5, .6])
    thr = models.best_f2_threshold(y, s)
    assert fbeta_score(y, (s >= thr).astype(int), beta=2) == 1.0
    assert models.f2_top_k(y, s, 3) == 1.0
    assert models.f2_top_k(y, s, 1) < 1.0


def test_esa_mse_hr():
    final = np.array([-5.0, -30.0, -4.0]); pred = np.array([-5.5, -30.0, -7.0])
    # only the two high-risk events count; the -7 prediction is clipped to -6.001
    assert np.isclose(models.esa_mse_hr(final, pred), ((0.5) ** 2 + (2.001) ** 2) / 2)
