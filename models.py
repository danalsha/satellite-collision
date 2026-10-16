"""Step 2 — models, splits, baselines, controls, uncertainty, manifest.

    python3 models.py <featureset> <split>

featureset:
    latest    the 100 numbers + object type of the LATEST usable CDM (>= 2 days out)
    history   latest + summaries of the whole series of usable CDMs (count, span,
              first/mean/min/max/std/change/slope of risk, miss distance, ...)
    no_risk   history WITHOUT the three risk outputs the CDM carries (risk,
              max_risk_estimate, max_risk_scaling) and their histories: can the
              geometry and the uncertainties alone tell high-risk events apart?
split:
    official  ESA's competition split: train on the 8,293 eligible events ESA
              released for training, test on its 2,167 test events (150 high risk).
              Bootstrap intervals on the test set; the network over 3 seeds.
    cv        stratified 5-fold over all 10,460 eligible events (2.07 % high risk);
              for the history features repeated 5 times with different shuffles
              = 25 folds, so that the difference to the naive forecast can be
              measured fold by fold
    mission   5-fold with whole satellites (mission_id) held out

Every model is scored by PR-AUC (average precision), ROC-AUC, F2 at a
threshold, precision and recall at that threshold, and the Brier score where
the model outputs a probability.  The threshold is chosen on the TRAINING
partition only: a 3-fold cross-validated prediction inside the training
partition, and the threshold that maximises F2 on it.  The test partition is
never used to pick anything.

Models: logistic regression, decision tree, RBF SVM, Keras dense network,
histogram gradient boosting, and the paper's framing "hgb_delta": gradient
boosting that predicts the CHANGE in risk from the latest usable CDM to the
final one, so that predicted final risk = latest risk + predicted change.
Baselines: the latest risk (ESA's "LRP" naive
forecast, decision at -6 as ESA defines it, and with a tuned threshold), and a
constant score.  Control: labels permuted 3 times per fold (official split and
the first five folds of each CV split); logistic regression and gradient
boosting are both fitted to every permutation and the 6 runs are averaged.
Each result file carries a manifest with the
SHA-256 of the code and data, the configuration, package versions and seed.
"""
import os
import sys
import json
import time
import hashlib
import platform
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, GroupKFold, cross_val_predict,
                                     train_test_split)
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, fbeta_score)

import config
from config import HERE, DATA, PROC, OUT, SEED, HIGH_RISK, CATEGORICAL

N_BOOT = 200
NN_SEEDS = 3
N_CTRL_OFFICIAL, N_CTRL_FOLD = 3, 3
N_FOLDS = 5
N_REPEATS_CV = 5         # history features: the stratified split is repeated 5 times = 25 folds
                         # (paired comparisons with LRP); the other feature sets use the first repeat
INNER_FOLDS = 3
RISK_COLS = ["risk", "max_risk_estimate", "max_risk_scaling"]
NOT_FEATURES = ["mission_id", "n_cdms_total", "first_ttca", "last_ttca", "final_risk",
                "high_risk", "esa_test", "lrp_high"]
HIST_SUFFIX = ("_first", "_mean", "_min", "_max", "_std", "_change", "_slope")
HIST_EXTRA = ["n_cdms_used", "span_days", "risk_max_jump"]


# ------------------------------------------------------------------ features
def base_name(c):
    """risk_slope -> risk, c_sigma_t_first -> c_sigma_t, risk_max_jump -> risk."""
    if c == "risk_max_jump":
        return "risk"
    for suf in HIST_SUFFIX:
        if c.endswith(suf):
            return c[: -len(suf)]
    return c


def feature_lists(fs, tab):
    cols = [c for c in tab.columns if c not in NOT_FEATURES]
    hist = [c for c in cols if c.endswith(HIST_SUFFIX) or c in HIST_EXTRA]
    if fs == "latest":
        cols = [c for c in cols if c not in hist]
    elif fs == "no_risk":
        cols = [c for c in cols if base_name(c) not in RISK_COLS]
    elif fs != "history":
        raise SystemExit(f"unknown feature set {fs}")
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in cat]
    return num, cat


def skewed_columns(X, num):
    """Non-negative, heavily right-skewed columns get a log1p before scaling.
    Decided on the training partition, from the features alone (no labels)."""
    out = []
    for c in num:
        x = X[c].dropna()
        if len(x) > 10 and x.min() >= 0 and x.skew() > 2:
            out.append(c)
    return out


def preprocessor(num, cat, log_cols):
    lin = [c for c in num if c not in log_cols]
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]
    parts = [("num", Pipeline(steps), lin),
             ("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))] + steps), log_cols)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat))
    return ColumnTransformer(parts, sparse_threshold=0)


# ------------------------------------------------------------------ the network
class KerasNet(BaseEstimator, ClassifierMixin):
    """Dense network in the course's TensorFlow style, with the two-stage protocol:
    hold out 15 % of the training rows to find the best epoch by early stopping,
    then retrain on all the training rows for that many epochs."""

    def __init__(self, units=(64, 32), lr=1e-3, seed=SEED, max_epochs=150, patience=10, two_stage=True):
        self.units, self.lr = units, lr
        self.seed, self.max_epochs, self.patience, self.two_stage = seed, max_epochs, patience, two_stage

    def _build(self, n_in):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Input
        try:                                    # bit-for-bit repeatable training on CPU
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        tf.keras.utils.set_random_seed(self.seed)
        m = Sequential([Input(shape=(n_in,))]
                       + [Dense(units=u, activation="relu") for u in self.units]
                       + [Dense(units=1, activation="sigmoid")])
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss="binary_crossentropy")
        return m

    def fit(self, X, y):
        import tensorflow as tf
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
        self.classes_ = np.array([0, 1])
        pos = y.mean(); cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.seed)
        m = self._build(X.shape[1])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        h = m.fit(Xa, ya, validation_data=(Xb, yb), epochs=self.max_epochs, batch_size=128,
                  class_weight=cw, callbacks=[es], verbose=0)
        self.best_epoch_ = int(np.argmin(h.history["val_loss"])) + 1
        if self.two_stage:                      # retrain on every training row for that many epochs
            self.model_ = self._build(X.shape[1])
            self.model_.fit(X, y, epochs=self.best_epoch_, batch_size=128, class_weight=cw, verbose=0)
        else:                                   # inner CV for the threshold: keep the early-stopped net
            self.model_ = m
        tf.keras.backend.clear_session()
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, np.float32), verbose=0).ravel()
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_models(seed=SEED):
    """Hyperparameters chosen in CS465_Project.ipynb (section 5.2) by 3-fold cross-validation
    on the training 80 % of a stratified split, scored by average precision:
      logreg C 0.01 (of 0.01, 0.1, 1, 10); tree depth 6, min 100 rows per leaf (of 3/6/10 x 5/25/100);
      network (64, 32) units, Adam 1e-3 (of (32), (64,32), (128,64) x 1e-3, 1e-2);
      SVM C 0.3, gamma 0.001 (of 0.3/1/3 x scale/0.001/0.01);
      boosting learning rate 0.05, 7 leaves (of 0.05/0.1 x 7/15/31), 150 trees, L2 1."""
    return {
        "logreg": LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000),
        "tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, class_weight="balanced", random_state=seed),
        "svm": SVC(kernel="rbf", C=0.3, gamma=0.001, class_weight="balanced", random_state=seed),
        "nn": KerasNet(seed=seed),
        "hgb": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, max_leaf_nodes=7,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              class_weight="balanced", random_state=seed),
    }


def score_of(pipe, X):
    est = pipe[-1]
    if hasattr(est, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def best_f2_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f2 = 5 * p[:-1] * r[:-1] / np.maximum(4 * p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f2))])


def f2_top_k(y, s, k):
    """F2 when exactly the k highest-scoring events are flagged (ties broken by order)."""
    order = np.argsort(-np.asarray(s), kind="stable")
    pred = np.zeros(len(y), int); pred[order[:k]] = 1
    return float(fbeta_score(y, pred, beta=2))


def metrics(y, s, thr, proba, n_flag=None):
    y = np.asarray(y); s = np.asarray(s); pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out = dict(ap=float(average_precision_score(y, s)), roc=float(roc_auc_score(y, s)),
               f2=float(fbeta_score(y, pred, beta=2)), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
               tp=tp, fp=fp, fn=fn, threshold=float(thr), n=int(len(y)), n_pos=int(y.sum()))
    if proba:
        out["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1)))
    if n_flag is not None:
        out["f2_at_lrp_rate"] = f2_top_k(y, s, n_flag)   # same number of alerts as the naive forecast raises
        out["n_flag_lrp"] = int(n_flag)
    return out


def fit_and_score(name, est, pre, Xtr, ytr, Xte, yte, rng, n_flag=None):
    """Fit on the training partition, choose the F2 threshold by inner CV on the
    training partition, score the test partition."""
    pipe = Pipeline([("pre", clone(pre)), ("clf", clone(est))])
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=int(rng.integers(1 << 31)))
    method = "predict_proba" if hasattr(est, "predict_proba") else "decision_function"
    inner_pipe = clone(pipe)
    if name == "nn":
        inner_pipe.set_params(clf__two_stage=False)
    oof = cross_val_predict(inner_pipe, Xtr, ytr, cv=inner, method=method)
    oof = oof[:, 1] if oof.ndim == 2 else oof
    thr = best_f2_threshold(ytr, oof)
    pipe.fit(Xtr, ytr)
    s = score_of(pipe, Xte)
    m = metrics(yte, s, thr, proba=(method == "predict_proba"), n_flag=n_flag)
    m["train_inner_f2"] = float(fbeta_score(ytr, (oof >= thr).astype(int), beta=2))
    m["train_inner_ap"] = float(average_precision_score(ytr, oof))
    if name == "nn":
        m["best_epoch"] = int(pipe[-1].best_epoch_)
    return m, s, pipe


def esa_mse_hr(final_true, final_pred):
    """ESA's regression metric: mean squared error of the predicted final log-risk over the
    TRUE high-risk events, with predictions clipped at -6.001 as the paper does."""
    hi = final_true >= HIGH_RISK
    p = np.maximum(np.asarray(final_pred, float), HIGH_RISK - 0.001)
    return float(np.mean((final_true[hi] - p[hi]) ** 2))


def delta_model(pre, Xtr, Xte, risk_tr, risk_te, final_tr, final_te, yte, rng, n_flag, seed=SEED):
    """The paper's framing (Uriot et al. 2022, section 6): learn the CHANGE in risk between
    the latest usable CDM and the final CDM, h = r - r_-2, with gradient boosting, and
    predict final risk = latest risk + h.  Decided at -6 like the naive forecast, and with
    a threshold tuned by inner CV."""
    # As in the paper, the target h is passed through a quantile transformer (to a uniform
    # distribution) before fitting: most events have h = 0 (risk -30 before and after) or a
    # large negative h (the risk collapses to -30), and a plain squared-error fit on h
    # predicts a large negative change for nearly every event (checked: it then flags no
    # event at all on the official test set).
    reg = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(learning_rate=0.1, max_iter=150, max_leaf_nodes=15,
                                                min_samples_leaf=20, l2_regularization=1.0, random_state=seed),
        transformer=QuantileTransformer(n_quantiles=200, output_distribution="uniform", random_state=seed))
    pipe = Pipeline([("pre", clone(pre)), ("reg", reg)])
    h_tr = np.asarray(final_tr, float) - np.asarray(risk_tr, float)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=int(rng.integers(1 << 31)))
    ytr = (np.asarray(final_tr) >= HIGH_RISK).astype(int)
    oof = np.asarray(risk_tr, float) + cross_val_predict(pipe, Xtr, h_tr, cv=inner.split(Xtr, ytr))
    pipe.fit(Xtr, h_tr)
    pred = np.asarray(risk_te, float) + pipe.predict(Xte)
    out = {"hgb_delta": metrics(yte, pred, HIGH_RISK, proba=False, n_flag=n_flag),
           "hgb_delta_tuned": metrics(yte, pred, best_f2_threshold(ytr, oof), proba=False, n_flag=n_flag)}
    out["hgb_delta"]["mse_hr"] = esa_mse_hr(np.asarray(final_te, float), pred)
    out["hgb_delta"]["train_inner_f2"] = float(fbeta_score(ytr, (oof >= HIGH_RISK).astype(int), beta=2))
    return out, pred


def baselines(risk_tr, ytr, risk_te, yte):
    """risk_* = the latest usable CDM's risk, taken from the event table whether or
    not 'risk' is in the feature set (the no_risk set still has a baseline)."""
    out, scores = {}, {}
    # ESA's naive forecast: the latest risk, decided at -6
    s_tr, s_te = np.asarray(risk_tr, float), np.asarray(risk_te, float)
    n_flag = int((s_te >= HIGH_RISK).sum())
    out["lrp"] = metrics(yte, s_te, HIGH_RISK, proba=False, n_flag=n_flag)
    scores["lrp"] = s_te
    out["lrp_tuned"] = metrics(yte, s_te, best_f2_threshold(ytr, s_tr), proba=False, n_flag=n_flag)
    scores["lrp_tuned"] = s_te
    # constant score: every event gets the training base rate
    c = np.full(len(yte), ytr.mean())
    out["constant"] = metrics(yte, c, c[0], proba=True, n_flag=n_flag)
    scores["constant"] = c
    return out, scores, n_flag


def bootstrap(y, s, thr, rng, n=N_BOOT):
    y = np.asarray(y); s = np.asarray(s); idx = np.arange(len(y)); rows = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0:
            continue
        pred = (s[b] >= thr).astype(int)
        rows.append((average_precision_score(y[b], s[b]), roc_auc_score(y[b], s[b]), fbeta_score(y[b], pred, beta=2)))
    a = np.array(rows)
    lo, hi = np.percentile(a, [2.5, 97.5], axis=0)
    return {k: [float(l), float(h)] for k, l, h in zip(["ap", "roc", "f2"], lo, hi)}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def manifest(fs, split, seed):
    import sklearn, tensorflow
    raw = os.path.join(DATA, "Collision Avoidance Challenge - Dataset", "raw_data", "raw_data_2015-2019.gz")
    return dict(featureset=fs, split=split, seed=seed, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                code={f: sha(os.path.join(HERE, f)) for f in ("config.py", "build_table.py", "models.py")},
                data={"raw_data_2015-2019.gz": sha(raw), "events.csv.gz": sha(os.path.join(PROC, "events.csv.gz"))},
                config={k: getattr(config, k) for k in ("HIGH_RISK", "CUTOFF_DAYS", "FINAL_WITHIN_DAYS")},
                settings=dict(N_BOOT=N_BOOT, NN_SEEDS=NN_SEEDS, N_FOLDS=N_FOLDS, N_REPEATS_CV=N_REPEATS_CV, INNER_FOLDS=INNER_FOLDS,
                              N_CTRL_OFFICIAL=N_CTRL_OFFICIAL, N_CTRL_FOLD=N_CTRL_FOLD),
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              sklearn=sklearn.__version__, tensorflow=tensorflow.__version__,
                              platform=platform.platform()))


# ------------------------------------------------------------------ main
def load():
    tab = pd.read_csv(os.path.join(PROC, "events.csv.gz"), index_col=0)
    return tab


def make_folds(tab, split, fs="history"):
    y = tab["high_risk"].values
    if split == "official":
        return [(np.where(tab.esa_test.values == 0)[0], np.where(tab.esa_test.values == 1)[0])]
    if split == "cv":
        reps = N_REPEATS_CV if fs == "history" else 1
        return list(RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=reps, random_state=SEED).split(tab, y))
    if split == "mission":
        return list(GroupKFold(N_FOLDS).split(tab, y, groups=tab.mission_id.values))
    raise SystemExit(f"unknown split {split}")


def tag(fs, split, k, nn_seed=None, control=False):
    t = f"{fs}__{split}_{k}"
    if control:
        return f"{t}_control"
    return t if nn_seed is None else f"{t}_nnseed{nn_seed}"


def run_fold(fs, split, k, nn_seed=None, control=False):
    """One fold: baselines and all models; or only the network with another seed
    (--nn-seed); or only the shuffled-label controls (--control)."""
    tab = load()
    num, cat = feature_lists(fs, tab)
    X_all = tab[num + cat]; y_all = tab["high_risk"].values
    tr, te = make_folds(tab, split, fs)[k]
    rng = np.random.default_rng(SEED + 1000 * k + (0 if nn_seed is None else nn_seed))
    t0 = time.time()
    Xtr, Xte, ytr, yte = X_all.iloc[tr], X_all.iloc[te], y_all[tr], y_all[te]
    log_cols = skewed_columns(Xtr, num)
    pre = preprocessor(num, cat, log_cols)
    b, fold_scores, n_flag = baselines(tab.risk.values[tr], ytr, tab.risk.values[te], yte)
    res = {"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)),
           "pos_train": int(ytr.sum()), "pos_test": int(yte.sum()),
           "base_rate_test": float(yte.mean()), "n_log_columns": len(log_cols),
           "missions_test": sorted(set(int(m) for m in tab.mission_id.values[te])),
           "models": {}}

    if nn_seed is not None:                       # extra network seed only
        e = KerasNet(seed=nn_seed)
        m, sc, _ = fit_and_score("nn", e, pre, Xtr, ytr, Xte, yte, rng, n_flag)
        res["models"]["nn"] = m; fold_scores = {"nn": sc}
        res["seconds"] = round(time.time() - t0, 1)
    elif control:
        # shuffled-label control: train on permuted labels, score on the true ones
        # (on the repeated split, only in the first repeat: folds 0-4)
        rng = np.random.default_rng(SEED + 1000 * k + 777)
        n_ctrl = N_CTRL_OFFICIAL if split == "official" else (N_CTRL_FOLD if k < N_FOLDS else 0)
        ctrl = {name: [] for name in ("logreg", "hgb")}
        fold_scores = {}
        for j in range(n_ctrl):
            yp = rng.permutation(ytr)
            for name in ctrl:
                m, sc, _ = fit_and_score(name, make_models(SEED + j)[name], pre, Xtr, yp, Xte, yte, rng, n_flag)
                ctrl[name].append(m); fold_scores[f"control_{name}_{j}"] = sc
        allc = [m for name in ctrl for m in ctrl[name]]
        if allc:
            res["models"]["shuffled_control"] = {q: float(np.mean([m[q] for m in allc])) for q in ("ap", "roc", "f2", "f2_at_lrp_rate")}
            res["models"]["shuffled_control"].update({f"{q}_sd": float(np.std([m[q] for m in allc])) for q in ("ap", "roc", "f2")})
            res["models"]["shuffled_control"]["n_permutations"] = len(allc)
            res["models"]["shuffled_control"]["runs"] = allc
        res["seconds"] = round(time.time() - t0, 1)
    else:
        res["models"].update(b)
        res["models"]["lrp"]["mse_hr"] = esa_mse_hr(tab.final_risk.values[te], tab.risk.values[te])
        dm, dsc = delta_model(pre, Xtr, Xte, tab.risk.values[tr], tab.risk.values[te],
                              tab.final_risk.values[tr], tab.final_risk.values[te], yte, rng, n_flag)
        res["models"].update(dm); fold_scores["hgb_delta"] = dsc; fold_scores["hgb_delta_tuned"] = dsc
        print(f"  fold {k} hgb_delta ap {dm['hgb_delta']['ap']:.3f} f2@-6 {dm['hgb_delta']['f2']:.3f} "
              f"tuned {dm['hgb_delta_tuned']['f2']:.3f} mse_hr {dm['hgb_delta']['mse_hr']:.3f} "
              f"(lrp mse_hr {res['models']['lrp']['mse_hr']:.3f})", flush=True)
        for name, est in make_models().items():
            m, sc, _ = fit_and_score(name, est, pre, Xtr, ytr, Xte, yte, rng, n_flag)
            res["models"][name] = m; fold_scores[name] = sc
            print(f"  fold {k} {name:8s} ap {m['ap']:.3f} roc {m['roc']:.3f} f2 {m['f2']:.3f} "
                  f"f2@lrp-rate {m['f2_at_lrp_rate']:.3f} (thr {m['threshold']:.3f}, p {m['precision']:.2f}, "
                  f"r {m['recall']:.2f})", flush=True)
        if split == "official":
            for name in list(b) + list(make_models()) + ["hgb_delta", "hgb_delta_tuned"]:
                res["models"][name]["ci95"] = bootstrap(yte, fold_scores[name], res["models"][name]["threshold"], rng)
        res["seconds"] = round(time.time() - t0, 1)

    t = tag(fs, split, k, nn_seed, control)
    res["code_sha256"] = {f: sha(os.path.join(HERE, f)) for f in ("config.py", "build_table.py", "models.py")}
    res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(res, open(os.path.join(OUT, f"fold_{t}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"fold_{t}.npz"), y=yte,
                        idx=tab.index.values[te], **{n: np.asarray(v) for n, v in fold_scores.items()})
    what = "nn seed %d" % nn_seed if nn_seed is not None else ("controls" if control else "all models")
    print(f"fold {k} ({what}) done in {res['seconds']} s")


def merge(fs, split):
    tab = load(); num, cat = feature_lists(fs, tab)
    n_folds = len(make_folds(tab, split, fs))
    per_fold, scores = [], {}
    for k in range(n_folds):
        p = os.path.join(OUT, f"fold_{tag(fs, split, k)}.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: run  python3 models.py {fs} {split} --fold {k}")
        res = json.load(open(p))
        z = np.load(os.path.join(OUT, f"fold_{tag(fs, split, k)}.npz"), allow_pickle=True)
        scores[f"fold{k}_y"] = z["y"]; scores[f"fold{k}_idx"] = z["idx"]
        for n in z.files:
            if n not in ("y", "idx"):
                scores[f"fold{k}_{n}"] = z[n]
        # shuffled-label controls, run as their own job
        pc = os.path.join(OUT, f"fold_{tag(fs, split, k, control=True)}.json")
        need_control = split == "official" or k < N_FOLDS
        if need_control and not os.path.exists(pc):
            raise SystemExit(f"missing {pc}: run  python3 models.py {fs} {split} --fold {k} --control")
        if os.path.exists(pc):
            c = json.load(open(pc))["models"].get("shuffled_control")
            if c:
                res["models"]["shuffled_control"] = c
                zc = np.load(os.path.join(OUT, f"fold_{tag(fs, split, k, control=True)}.npz"), allow_pickle=True)
                for n in zc.files:
                    if n.startswith("control_"):
                        scores[f"fold{k}_{n}"] = zc[n]
        # extra network seeds (official split)
        seeds = [res["models"]["nn"]]
        for sp in sorted(f for f in os.listdir(OUT) if f.startswith(f"fold_{tag(fs, split, k)}_nnseed") and f.endswith(".json")):
            seeds.append(json.load(open(os.path.join(OUT, sp)))["models"]["nn"])
        if len(seeds) > 1:
            nn = dict(seeds[0]); nn["seeds"] = seeds
            for q in ("ap", "roc", "f2", "f2_at_lrp_rate", "precision", "recall", "brier"):
                nn[q] = float(np.mean([r[q] for r in seeds])); nn[f"{q}_seed_sd"] = float(np.std([r[q] for r in seeds]))
            nn["n_seeds"] = len(seeds)
            res["models"]["nn"] = nn
        per_fold.append(res)
    names = sorted({n for f in per_fold for n in f["models"]})
    summary = {}
    for name in names:
        # the shuffled-label control is only run on the first N_FOLDS folds of a
        # repeated CV, so average each model over the folds that actually have it
        have = [f for f in per_fold if name in f["models"]]
        summary[name] = {"n_folds": len(have)}
        for q in ("ap", "roc", "f2", "f2_at_lrp_rate", "precision", "recall", "brier", "mse_hr"):
            v = [f["models"][name][q] for f in have if q in f["models"][name]]
            if v:
                summary[name][q] = float(np.mean(v)); summary[name][f"{q}_sd"] = float(np.std(v))
        if "ci95" in have[0]["models"][name]:
            summary[name]["ci95"] = have[0]["models"][name]["ci95"]
        if "n_seeds" in have[0]["models"][name]:
            summary[name]["n_seeds"] = have[0]["models"][name]["n_seeds"]
            summary[name].update({f"{q}_seed_sd": have[0]["models"][name][f"{q}_seed_sd"] for q in ("ap", "roc", "f2")})
        # paired, fold by fold, against the naive forecast on the same test events
        if name != "lrp" and len(have) > 1:
            d = {}
            for q, lab in (("ap", "ap"), ("roc", "roc"), ("f2_at_lrp_rate", "f2rate"), ("f2", "f2")):
                pairs = [(f["models"][name][q], f["models"]["lrp"][q]) for f in have
                         if q in f["models"][name] and q in f["models"]["lrp"]]
                if pairs:
                    diff = np.array([a - b for a, b in pairs])
                    d[f"{lab}_diff"] = float(diff.mean()); d[f"{lab}_diff_sd"] = float(diff.std())
                    d[f"{lab}_wins"] = int((diff > 0).sum()); d["n_folds"] = len(diff)
            summary[name]["vs_lrp"] = d
    out = dict(manifest=manifest(fs, split, SEED), features=dict(numeric=num, categorical=cat),
               n_folds=n_folds, summary=summary, folds=per_fold)
    json.dump(out, open(os.path.join(OUT, f"metrics_{fs}__{split}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"scores_{fs}__{split}.npz"), **scores)
    for name in names:
        s_ = summary[name]
        print(f"{name:17s} ap {s_['ap']:.3f}±{s_['ap_sd']:.3f}  roc {s_['roc']:.3f}  f2 {s_['f2']:.3f}±{s_['f2_sd']:.3f}"
              + (f"  f2@lrp-rate {s_['f2_at_lrp_rate']:.3f}" if "f2_at_lrp_rate" in s_ else ""))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("featureset", choices=["latest", "history", "no_risk"])
    ap.add_argument("split", choices=["official", "cv", "mission"])
    ap.add_argument("--fold", type=int, help="run this fold only (0-based)")
    ap.add_argument("--nn-seed", type=int, help="with --fold: run only the network with this seed")
    ap.add_argument("--control", action="store_true", help="with --fold: run only the shuffled-label controls")
    ap.add_argument("--merge", action="store_true", help="merge the fold files into metrics_*.json")
    a = ap.parse_args()
    if a.merge:
        merge(a.featureset, a.split)
    elif a.fold is not None:
        run_fold(a.featureset, a.split, a.fold, a.nn_seed, a.control)
    else:
        for k in range(len(make_folds(load(), a.split, a.featureset))):
            run_fold(a.featureset, a.split, k)
            if a.split == "official" or k < N_FOLDS:
                run_fold(a.featureset, a.split, k, control=True)
        merge(a.featureset, a.split)
