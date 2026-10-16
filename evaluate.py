"""Step 3 — collect every run, write results/RESULTS.md and the figures.

Figures
  risk_before_after.png   latest usable risk vs final risk, all 10,460 eligible events
  example_events.png      risk time series of events the naive forecast gets right and wrong
  pr_curves_official.png  precision-recall on ESA's test set, history features
  splits.png              PR-AUC and F2 of every model on the three splits (history features)
  featuresets.png         latest vs history vs no_risk, stratified CV
  calibration.png         reliability of the probabilistic models on ESA's test set
  importance.png          permutation importance of gradient boosting on ESA's test set
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from config import DATA, PROC, OUT, FIG, HIGH_RISK, CUTOFF_DAYS

MODELS = ["lrp", "lrp_tuned", "hgb_delta", "hgb_delta_tuned", "logreg", "tree", "svm", "nn", "hgb",
          "constant", "shuffled_control"]
LABEL = {"lrp": "Latest risk, decided at -6 (ESA's LRP)", "lrp_tuned": "Latest risk, tuned threshold",
         "hgb_delta": "Boosting on the change in risk, at -6 (paper's framing)",
         "hgb_delta_tuned": "Boosting on the change in risk, tuned threshold",
         "logreg": "Logistic regression", "tree": "Decision tree", "svm": "SVM (RBF)",
         "nn": "Neural network", "hgb": "Gradient boosting (classifier)",
         "constant": "Constant score", "shuffled_control": "Shuffled-label control"}
SHORT = {"lrp": "LRP (-6)", "lrp_tuned": "LRP tuned", "hgb_delta": "HGB Δrisk (-6)", "hgb_delta_tuned": "HGB Δrisk tuned",
         "logreg": "logistic", "tree": "tree", "svm": "SVM", "nn": "network", "hgb": "boosting",
         "constant": "constant", "shuffled_control": "shuffled"}
COL = {"lrp": "#2B2B2B", "lrp_tuned": "#555555", "hgb_delta": "#8E6C3A", "hgb_delta_tuned": "#B8955E",
       "logreg": "#5B8DB8", "tree": "#8FBF6A", "svm": "#C9A84A", "nn": "#B86B5B", "hgb": "#6B3FA0",
       "constant": "#AAAAAA", "shuffled_control": "#C8C8C8"}
SPLITS = ["official", "cv", "mission"]
SPLIT_LABEL = {"official": "ESA's competition split", "cv": "stratified 5-fold", "mission": "satellites held out"}
FS = ["latest", "history", "no_risk"]
FS_LABEL = {"latest": "latest CDM", "history": "latest + history", "no_risk": "history, no risk columns"}


def load_runs():
    runs = {}
    for p in glob.glob(os.path.join(OUT, "metrics_*.json")):
        d = json.load(open(p)); m = d["manifest"]
        runs[(m["featureset"], m["split"])] = d
    return runs


def fmt(v, sd=None, ci=None, nd=3):
    if v is None:
        return "–"
    s = f"{v:.{nd}f}"
    if ci is not None:
        s += f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
    elif sd is not None:
        s += f" ± {sd:.{nd}f}"
    return s


def table(d, split):
    rows = []
    for name in MODELS:
        if name not in d["summary"]:
            continue
        s = d["summary"][name]
        ci = s.get("ci95") if split == "official" else None
        sd = None if split == "official" else s.get("ap_sd")
        row = [LABEL[name],
               fmt(s.get("ap"), None if split == "official" else s.get("ap_sd"), ci.get("ap") if ci else None),
               fmt(s.get("roc"), None if split == "official" else s.get("roc_sd")),
               fmt(s.get("f2"), None if split == "official" else s.get("f2_sd"), ci.get("f2") if ci else None),
               fmt(s.get("f2_at_lrp_rate")),
               fmt(s.get("precision"), nd=2), fmt(s.get("recall"), nd=2),
               fmt(s.get("brier"), nd=3) if "brier" in s else "–"]
        if name == "nn" and "n_seeds" in s:
            # the point estimates are means over the seeds; the brackets (official split)
            # are the bootstrap of the first seed only, so say so
            row[1] = fmt(s.get("ap"), None if split == "official" else s.get("ap_sd")) + f" (seed sd {s['ap_seed_sd']:.3f}" + \
                (f"; first-seed CI [{ci['ap'][0]:.3f}, {ci['ap'][1]:.3f}])" if ci else ")")
            row[3] = fmt(s.get("f2"), None if split == "official" else s.get("f2_sd")) + f" (seed sd {s['f2_seed_sd']:.3f}" + \
                (f"; first-seed CI [{ci['f2'][0]:.3f}, {ci['f2'][1]:.3f}])" if ci else ")")
        v = s.get("vs_lrp", {})
        for q in ("ap", "f2rate"):
            row.append(f"{v[f'{q}_diff']:+.3f} ± {v[f'{q}_diff_sd']:.3f}, better in {v[f'{q}_wins']}/{v['n_folds']} folds"
                       if f"{q}_diff" in v else "–")
        rows.append(row)
    head = ["model", "PR-AUC", "ROC-AUC", "F2 (own threshold)", "F2 at LRP's alert count", "precision", "recall", "Brier",
            "PR-AUC vs LRP, paired", "F2 at LRP's count vs LRP, paired"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_results(runs, eda):
    L = ["# Results", "", f"Eligible events: {eda['eligible_events']:,} ({eda['eligible_high_risk']} high risk, "
         f"base rate {eda['eligible_base_rate']:.4f}). Every number is on events the model never saw. "
         "On the official split the brackets are 95 % bootstrap intervals over the 2,167 test events "
         "(200 resamples); the network's point estimates are means over 3 seeds and its brackets are the first "
         "seed's bootstrap. On the cross-validated splits ± is the standard deviation over the folds "
         "(25 folds = 5 repeats of a stratified 5-fold split for the history feature set, 5 folds otherwise; "
         "5 folds with satellites held out; the shuffled-label control runs on 5 folds of every split). "
         "The last two columns compare each model with the naive forecast fold by fold, on the same test events. "
         "F2 (own threshold) uses a threshold chosen by 3-fold cross-validation inside the training partition; "
         "'F2 at LRP's alert count' flags exactly as many events as the naive forecast flags at -6, so it "
         "compares rankings without any threshold.", ""]
    for split in SPLITS:
        L += [f"## {SPLIT_LABEL[split]}", ""]
        for fs in FS:
            d = runs.get((fs, split))
            if d is None:
                continue
            f0 = d["folds"][0]
            L += [f"### Features: {FS_LABEL[fs]} ({len(d['features']['numeric'])} numeric + "
                  f"{len(d['features']['categorical'])} categorical)", ""]
            if split == "official":
                L += [f"Train {f0['n_train']:,} events ({f0['pos_train']} high risk), test {f0['n_test']:,} "
                      f"({f0['pos_test']} high risk, base rate {f0['base_rate_test']:.3f}).", ""]
            L += [table(d, split), ""]
            if "mse_hr" in d["summary"].get("hgb_delta", {}):
                L += [f"ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): "
                      f"naive forecast {d['summary']['lrp']['mse_hr']:.3f}, boosting on the change in risk "
                      f"{d['summary']['hgb_delta']['mse_hr']:.3f}"
                      + (" (the paper: LRP 0.513, winning team 0.407)." if split == "official" else "."), ""]
        if split == "mission" and (("history", split) in runs):
            d = runs[("history", split)]
            L += ["Per fold (history features), PR-AUC:", "",
                  "| fold | test events | high risk | base rate | missions in test | LRP | boosting | logistic | network |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for f in d["folds"]:
                m = f["models"]
                L.append(f"| {f['fold']} | {f['n_test']:,} | {f['pos_test']} | {f['base_rate_test']:.3f} | "
                         f"{', '.join(map(str, f['missions_test'])) if split == 'mission' else 'all'} | "
                         f"{m['lrp']['ap']:.3f} | {m['hgb']['ap']:.3f} | {m['logreg']['ap']:.3f} | {m['nn']['ap']:.3f} |")
            L.append("")
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L))
    print("wrote results/RESULTS.md")


# ------------------------------------------------------------------ figures
def fig_before_after(tab):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    x = tab.risk.clip(lower=-31); y = tab.final_risk.clip(lower=-31)
    hi = tab.high_risk == 1
    ax.scatter(x[~hi], y[~hi], s=4, alpha=0.25, color="#5B8DB8", label="final risk < -6")
    ax.scatter(x[hi], y[hi], s=9, alpha=0.8, color="#B8322C", label="final risk ≥ -6 (high risk)")
    ax.axvline(HIGH_RISK, color="k", lw=0.8, ls="--"); ax.axhline(HIGH_RISK, color="k", lw=0.8, ls="--")
    c = eda_counts(tab)
    ax.set_ylim(-31.5, 0.3); ax.set_xlim(-31.5, 0.3)
    ax.text(-5.7, -0.2, f"naive forecast right\n{c['TP']} events", fontsize=8, va="top")
    ax.text(-5.7, -27.6, f"false alarm\n{c['FP']} events", fontsize=8, va="top")
    ax.text(-27.5, -0.2, f"missed\n{c['FN']} events", fontsize=8, va="top")
    ax.text(-27.5, -27.6, f"{c['TN']:,} events", fontsize=8, color="#555", va="top")
    ax.set_xlabel("log10 risk in the latest CDM at least 2 days before closest approach")
    ax.set_ylabel("log10 risk in the final CDM (the label)")
    ax.set_title("What the naive forecast sees, and what happens")
    ax.legend(loc="center left", fontsize=8, frameon=False)
    ax = axes[1]
    bins = np.arange(-31, 0, 1)
    ax.hist(x, bins=bins, alpha=0.6, color="#5B8DB8", label="latest usable CDM")
    ax.hist(y, bins=bins, alpha=0.6, color="#8E6C3A", label="final CDM")
    ax.axvline(HIGH_RISK, color="k", lw=0.8, ls="--"); ax.set_yscale("log")
    ax.set_xlabel("log10 risk (values of -30 mean 'no risk')"); ax.set_ylabel("events (log scale)")
    ax.set_title("Risk moves down as closest approach nears"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "risk_before_after.png"), dpi=160); plt.close(fig)


def eda_counts(tab):
    p = tab.risk >= HIGH_RISK; y = tab.high_risk == 1
    return dict(TP=int((p & y).sum()), FP=int((p & ~y).sum()), FN=int((~p & y).sum()), TN=int((~p & ~y).sum()))


def fig_examples(tab):
    raw = pd.read_csv(os.path.join(DATA, "Collision Avoidance Challenge - Dataset", "raw_data", "raw_data_2015-2019.gz"),
                      usecols=["event_id", "time_to_tca", "risk"])
    rng = np.random.default_rng(3)
    groups = {"high risk, naive forecast right": tab[(tab.high_risk == 1) & (tab.lrp_high == 1)],
              "high risk, naive forecast missed it": tab[(tab.high_risk == 1) & (tab.lrp_high == 0)],
              "low risk, naive forecast raised a false alarm": tab[(tab.high_risk == 0) & (tab.lrp_high == 1)]}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (title, sub) in zip(axes, groups.items()):
        ids = rng.choice(sub.index.values, min(8, len(sub)), replace=False)
        for i in ids:
            e = raw[raw.event_id == i].sort_values("time_to_tca", ascending=False)
            ax.plot(-e.time_to_tca, e.risk.clip(lower=-12), "-o", ms=2.5, lw=1, alpha=0.8)
        ax.axvspan(-CUTOFF_DAYS, 0.2, color="#DDD", alpha=0.6)
        ax.axhline(HIGH_RISK, color="k", lw=0.8, ls="--")
        ax.set_title(f"{title}\n({len(sub)} such events; 8 shown)", fontsize=9)
        ax.set_xlabel("days before closest approach")
    axes[0].set_ylabel("log10 risk (clipped at -12)")
    axes[0].text(-1.9, -11.5, "last 2 days:\nnot available\nto the model", fontsize=7, color="#444")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "example_events.png"), dpi=160); plt.close(fig)


def fig_pr_official(runs):
    d = runs.get(("history", "official"))
    if d is None:
        return
    z = np.load(os.path.join(OUT, "scores_history__official.npz"), allow_pickle=True)
    y = z["fold0_y"]
    fig, ax = plt.subplots(figsize=(6.6, 5))
    for n in z.files:
        if n.startswith("fold0_control_"):
            p, r, _ = precision_recall_curve(y, z[n]); ax.plot(r, p, color="#DDD", lw=0.8)
    for name in ["lrp", "hgb_delta", "logreg", "tree", "svm", "nn", "hgb"]:
        s = z[f"fold0_{name}"]; p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=COL[name], lw=1.6 if name in ("lrp", "hgb") else 1.1,
                label=f"{SHORT[name]}  AP {average_precision_score(y, s):.3f}")
    m = d["folds"][0]["models"]["lrp"]
    ax.plot([m["recall"]], [m["precision"]], "k*", ms=12, label="LRP decision at -6 (F2 0.739)")
    ax.axhline(y.mean(), color="#999", lw=0.8, ls=":", label=f"base rate {y.mean():.3f}")
    ax.set_xlabel("recall (share of the 150 high-risk events flagged)"); ax.set_ylabel("precision")
    ax.set_title("ESA's test set, 2,167 events: precision–recall\n(grey: shuffled-label controls)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pr_curves_official.png"), dpi=160, bbox_inches="tight"); plt.close(fig)


def fig_splits(runs):
    names = ["lrp", "hgb_delta", "logreg", "tree", "svm", "nn", "hgb", "shuffled_control"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, q, ttl in zip(axes, ["ap", "f2"], ["PR-AUC", "F2 (own threshold)"]):
        w = 0.1
        for j, name in enumerate(names):
            vals, errs = [], []
            for split in SPLITS:
                d = runs.get(("history", split))
                s = d["summary"].get(name, {}) if d else {}
                vals.append(s.get(q, np.nan))
                if split == "official":
                    # the network's value is a 3-seed mean while its CI was bootstrapped on
                    # seed 0, so the interval can sit slightly off the mean: clip at zero
                    ci = s.get("ci95", {}).get(q)
                    errs.append([max(0.0, s[q] - ci[0]), max(0.0, ci[1] - s[q])] if ci else [0, 0])
                else:
                    errs.append([s.get(f"{q}_sd", 0)] * 2)
            x = np.arange(len(SPLITS)) + (j - len(names) / 2 + 0.5) * w
            ax.bar(x, vals, w, color=COL[name], label=SHORT[name], yerr=np.array(errs).T, capsize=2, error_kw=dict(lw=0.7))
        ax.set_xticks(range(len(SPLITS))); ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS], fontsize=8)
        ax.set_title(ttl + ", history features"); ax.set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(FIG, "splits.png"), dpi=160); plt.close(fig)


def fig_featuresets(runs):
    names = ["lrp", "logreg", "nn", "hgb"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, q, ttl in zip(axes, ["ap", "f2"], ["PR-AUC", "F2 (own threshold)"]):
        w = 0.18
        for j, name in enumerate(names):
            vals = [runs[("%s" % fs, "cv")]["summary"][name][q] if (fs, "cv") in runs else np.nan for fs in FS]
            sds = [runs[(fs, "cv")]["summary"][name].get(f"{q}_sd", 0) if (fs, "cv") in runs else 0 for fs in FS]
            x = np.arange(len(FS)) + (j - 1.5) * w
            ax.bar(x, vals, w, color=COL[name], label=SHORT[name], yerr=sds, capsize=2, error_kw=dict(lw=0.7))
        ax.set_xticks(range(len(FS))); ax.set_xticklabels([FS_LABEL[f] for f in FS], fontsize=8)
        ax.set_title(ttl + ", stratified 5-fold CV\n(history: 5 folds × 5 repeats; others: 5 folds)", fontsize=10); ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "featuresets.png"), dpi=160); plt.close(fig)


def fig_calibration(runs):
    if ("history", "official") not in runs:
        return
    z = np.load(os.path.join(OUT, "scores_history__official.npz"), allow_pickle=True)
    y = z["fold0_y"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="perfect")
    edges = np.linspace(0, 1, 11)
    for name in ["logreg", "nn", "hgb", "tree"]:
        p = np.clip(z[f"fold0_{name}"], 0, 1)
        b = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs, ys, ns = [], [], []
        for i in range(10):
            m = b == i
            if m.sum() >= 5:                      # skip bins with fewer than 5 events
                xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
        ax.plot(xs, ys, "-o", ms=3, color=COL[name], label=SHORT[name])
        if name == "hgb":
            for x_, y_, n_ in zip(xs, ys, ns):
                ax.annotate(str(n_), (x_, y_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6, color=COL[name])
    ax.set_xlabel("mean predicted probability (10 equal-width bins, ≥5 events)")
    ax.set_ylabel("observed share of high-risk events")
    ax.set_title("Reliability on ESA's test set, history features\n(trained with balanced class weights at 0.8 % base rate, tested at 6.9 %;\n"
                 "numbers: events per bin for boosting)", fontsize=8.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration.png"), dpi=160); plt.close(fig)


def fig_importance(tab, runs):
    """Permutation importance of the gradient-boosting classifier (history features) on ESA's
    test set: drop in PR-AUC when one column is shuffled, 10 repeats."""
    import models
    d = runs.get(("history", "official"))
    if d is None:
        return
    num, cat = d["features"]["numeric"], d["features"]["categorical"]
    X = tab[num + cat]; y = tab.high_risk.values
    tr = np.where(tab.esa_test.values == 0)[0]; te = np.where(tab.esa_test.values == 1)[0]
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("pre", models.preprocessor(num, cat, models.skewed_columns(X.iloc[tr], num))),
                     ("clf", models.make_models()["hgb"])]).fit(X.iloc[tr], y[tr])
    r = permutation_importance(pipe, X.iloc[te], y[te], scoring="average_precision", n_repeats=10, random_state=0)
    imp = pd.Series(r.importances_mean, index=num + cat).sort_values(ascending=False)
    sd = pd.Series(r.importances_std, index=num + cat)
    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(range(len(top))[::-1], top.values, xerr=sd[top.index].values, color="#6B3FA0", capsize=2)
    ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("drop in PR-AUC when the column is shuffled (10 repeats)")
    ax.set_title("Gradient boosting, history features, ESA's test set", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "importance.png"), dpi=160); plt.close(fig)
    imp.to_csv(os.path.join(OUT, "importance_hgb_history_official.csv"), header=["drop_in_ap"])
    return top


def main():
    runs = load_runs()
    eda = json.load(open(os.path.join(PROC, "eda.json")))
    tab = pd.read_csv(os.path.join(PROC, "events.csv.gz"), index_col=0)
    write_results(runs, eda)
    fig_before_after(tab); fig_examples(tab)
    fig_pr_official(runs); fig_splits(runs); fig_featuresets(runs); fig_calibration(runs)
    top = fig_importance(tab, runs)
    if top is not None:
        print(top.round(4).to_string())
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
