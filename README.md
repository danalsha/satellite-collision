# Will this close approach still be dangerous at the moment of closest approach? ESA Collision Avoidance Challenge, 2015–2019

Two days before a satellite's closest approach to a piece of debris, the
operator has to decide whether to plan an avoidance manoeuvre. All they
have is the series of Conjunction Data Messages (CDMs) received so far,
each carrying a collision-probability estimate (`risk`, as log10). Can a
model, given only those CDMs, say better than the latest estimate itself
whether the *final* risk (in the last CDM, inside the last day) will be
above ESA's alert threshold of 10⁻⁶? Four course classifiers plus gradient
boosting, three ways of splitting the data, three feature sets, ESA's own
naive forecast as the baseline, a permutation control, bootstrap intervals,
a manifest on every result, and a direct re-run of the published paper's
experiment.

**Scope, stated plainly.** The naive forecast ("the latest risk is the final
risk", ESA's LRP baseline) is very hard to beat, and this project does not
beat it at the decision point. Gradient boosting *ranks* events better than
the latest risk (PR-AUC +0.08 ± 0.06 over 25 stratified folds, better in 21
of 25) and, after hyperparameter tuning, ties it on F2 (−0.001, better in 14
of 25 folds); the four course models stay 0.03–0.13 below it. The paper's
own result, a 1.3 % F2 gain, reproduces here (+1.0 %) only with the paper's
exact, deliberately weak LightGBM; the same recipe in scikit-learn loses
4 %, and a stronger regressor loses 22 %. The gain exists only when the
model barely moves the naive forecast.

**Deliverables for the course rubric.** `CS465_Project.ipynb` is the single
runnable notebook that walks through every rubric step in order (problem
statement, data preparation, EDA, feature engineering, model building with
hyperparameter optimisation, evaluation, insights, conclusion); the mapping
is in the last section of this README. The scripts described below are the
larger study behind the notebook's Section 7.

## Data

ESA's *Collision Avoidance Challenge* dataset (Kelvins competition,
Oct–Dec 2019), published on Zenodo (record 4463683, CC BY 4.0; Uriot et al.,
"Spacecraft collision avoidance challenge: design and results of a machine
learning competition", Astrodynamics 6(2), 121–140, 2022, online April 2021,
doi:10.1007/s42064-021-0101-5, arXiv:2008.03069; "the paper" below).
Downloaded 15 October 2026 (`get_data.py`; MD5 of the zip
`d19dc887…` as published; SHA-256 of every file in `data/CHECKSUMS.txt`).

The raw database `raw_data_2015-2019.gz` holds **199,082 CDMs for 15,321
close-approach events** of ESA satellites, 103 columns per CDM: `time_to_tca`
(days to closest approach), `risk`, `max_risk_estimate`, `max_risk_scaling`,
`miss_distance`, `mahalanobis_distance`, `relative_speed`, relative
position and velocity, then 40 fields each for the target (`t_`, the ESA
satellite) and the chaser (`c_`): orbital elements, covariance sigmas and
correlations, orbit-determination quality (`c_obs_used`, `c_time_lastob_*`,
`c_recommended_od_span`), drag and solar-radiation coefficients, span, and
the chaser's `c_object_type` (the only categorical column). `mission_id`
identifies 19 anonymised satellites. `risk` of −30 means "no risk". There
are no calendar dates in the public data, only `time_to_tca`.

The three competition files (`train_data.csv`, `test_data.csv`,
`test_data_private.csv`) are read only to find which events ESA used as its
test set, by matching CDMs back to the raw database on eight columns.

## What the data contains (findings that shaped the analysis)

1. **The label is the last CDM.** ESA's definition: an event is high risk
   when the risk in its final CDM (smallest `time_to_tca`) is ≥ −6. Over all
   15,321 events that is 515 events; 131 are ≥ −5 and 30 are ≥ −4. These
   three counts match the paper exactly.
2. **Only CDMs at least two days out may be used**, because the manoeuvre
   decision is taken then. 488 CDMs even arrive after closest approach
   (negative `time_to_tca`). Every feature in this project is computed from
   CDMs with `time_to_tca ≥ 2`; the last two days, including the CDM that
   carries the label, are never seen (checked by a test that recomputes 40
   events from the raw file).
3. **Eligibility.** An event counts only if its final CDM is within one day
   of closest approach (otherwise the "final" risk is not final) and it has
   at least one usable CDM. That leaves **10,460 events, 216 high risk
   (2.07 %)**, again exactly the paper's numbers. Per event the model sees
   a median of 14 CDMs (mean 11.2, range 1–17); the latest usable one is on
   average 2.31 days before closest approach.
4. **ESA's split is not stratified.** Its test set has 2,167 of these events
   with 150 high risk (6.9 %); the 8,293 training events have only 66
   (0.8 %). The labels derived here from the raw database agree with ESA's
   private test labels for all 2,167 events. The paper itself shows this
   split is an outlier among 10,000 random stratified splits.
5. **The naive forecast is strong and biased one way.** Deciding on the
   latest usable risk at −6 gets 162 of the 216 high-risk events, misses 54,
   and raises 324 false alarms (F2 0.600 over all eligible events; 0.739 on
   ESA's test set, the published LRP baseline). When the risk changes
   inside the last two days it usually *drops* (37 % of events drop, 8 %
   rise, 55 % stay put; `figures/risk_before_after.png`): 202 of the 324
   false alarms end at −30. Of the 54 misses, 26 had a latest risk between
   −7 and −6, just under the line, and only 5 were at −30, which is why a
   slightly lower threshold helps (results, point 1).
6. **Missions are very unequal**: five satellites account for 7,847 of the
   10,460 events; five have fewer than 20. Holding missions out therefore
   gives folds with 15 to 62 positives.
7. **Missing values** are rare except `c_rcs_estimate` (30 %) and the chaser
   covariance correlations (11 %); imputed with the training median plus a
   missing indicator.

## Method

| file | role |
|---|---|
| `CS465_Project.ipynb` | the rubric notebook: full pipeline on a stratified 80/20 split with hyperparameter search, its own 5-fold cross-validation and feature-selection experiment, then the saved results of the scripts below |
| `get_data.py` | download/verify the Zenodo zip (MD5, size), unpack |
| `build_table.py` | one row per eligible event: label from the last CDM, features from CDMs ≥ 2 days out only, ESA test-set flag, EDA summary |
| `models.py` | 5 models, 3 feature sets, 3 splits, baselines, controls, bootstrap, manifest; one fold per process |
| `paper_check.py` | the paper's own experiment: 505 stratified 80/20 splits, LightGBM on Δrisk, versus the naive forecast |
| `evaluate.py` | `results/RESULTS.md`, figures, permutation importance |
| `checksums.py` | SHA-256 of every data file |
| `tests/test_pipeline.py` | 7 tests: counts match the paper, no feature inside the last two days, label columns never in the features, LRP F2 reproduces, folds do not leak |

**Feature sets.** *latest*: the latest usable CDM's 100 numeric columns plus
`c_object_type`. *history*: the same plus 66 summaries of the usable CDM
series (first/mean/min/max/sd/change/slope of `risk`, `miss_distance`,
`mahalanobis_distance`, `max_risk_estimate`, the three chaser sigmas and
`t_sigma_t`, `c_obs_used`; number of CDMs, span, largest risk jump).
*no_risk*: history
without every risk column (`risk`, `max_risk_estimate`, `max_risk_scaling`
and their summaries), to ask what the other fields know.

**Splits.** *official*: ESA's competition split, train 8,293 / test 2,167.
*cv*: stratified 5-fold over the 10,460 events, repeated 5 times with
different shuffles for the history set (25 folds; 5 for the others).
*mission*: 5-fold with whole satellites held out (`GroupKFold` on
`mission_id`). Every metric is computed within its fold and reported as
mean ± sd over folds; pooled out-of-fold scores are saved for curves only.

**Models and hyperparameters.** Logistic regression, decision tree, RBF
SVM, Keras dense network (ReLU hidden layers, sigmoid output, Adam, early
stopping on a 15 % stratified slice to find the epoch, then retrained on the
full partition for that many epochs; three seeds on the official split),
histogram gradient boosting. All with balanced class weights, median
imputation with indicators, log1p on skewed non-negative columns,
standardisation, one-hot `c_object_type`. **Hyperparameters were chosen in
the notebook (section 5.2)** by 3-fold stratified cross-validation on the
training 80 % of a stratified split, scored by average precision, over
small grids: logistic C = 0.01 (of 0.01, 0.1, 1, 10); tree depth 6 with at
least 100 rows per leaf (of depth 3/6/10 × leaf 5/25/100); network
64–32–1 with learning rate 1e-3 (of (32), (64, 32), (128, 64) × 1e-3, 1e-2);
SVM C = 0.3, γ = 0.001 (of 0.3/1/3 × scale/0.001/0.01); boosting learning
rate 0.05 with 7 leaves per tree, 150 trees, L2 = 1 (of 0.05/0.1 × 7/15/31).
For the tree, depth 6 and depth 10 tie exactly at 100 rows per leaf and the
shallower tree is kept.
`models.py` uses the same values on every split. Each model's alert
threshold is chosen by 3-fold cross-validation *inside the training
partition* (maximum F2), never on the test events.

**Baselines.** *LRP*: latest usable risk, decided at −6 (ESA's). *LRP
tuned*: same score, threshold tuned by the same inner CV. *Boosting on
Δrisk*: the paper's framing, a regressor on h = final − latest risk with a
quantile-transformed target, prediction = latest risk + ĥ, decided at −6.
*Constant*: every event the same score. *Shuffled-label control*: labels
permuted 3 times per fold (official split and the first five folds of each
CV split); logistic regression and boosting are both fitted to every
permutation and the 6 runs are averaged.

**Metrics.** PR-AUC (ranking at every threshold), ROC-AUC, F2 at the
model's own threshold (ESA's competition metric, recall weighted 4× over
precision), and **F2 at LRP's alert count**: the model flags exactly as many
events as the naive forecast flags at −6, so two rankings are compared at
the same number of alerts without any threshold. Brier score for the
probabilistic models; ESA's MSE_HR (over true high-risk events, clipped at
−6.001) for the regression framing.

**Uncertainty.** 200-resample bootstrap of the 2,167 test events on the
official split; fold sd and paired fold-by-fold differences against LRP
(with the count of folds the model wins) on the others; seed sd for the
network.

**Manifest.** Each `results/metrics_*.json` records SHA-256 of `config.py`,
`build_table.py`, `models.py`, of the raw file and the event table, the
thresholds, settings, package versions and timestamp; each `fold_*.json`
records the code hashes it was produced with. `requirements.txt` pins
versions. `run_all.sh` reproduces everything after the download: 92 fold
jobs of 8–126 s each single-threaded (84 min in total, independent so they
parallelise), then the paper check (about 40 min). The notebook alone takes
about three minutes from scratch on one core, under a minute with its cache;
two cold runs give identical numbers (TensorFlow's deterministic ops are
switched on in both the notebook and `models.py`).

## Results

### ESA's competition split, history features (test 2,167 events, 150 high risk)

95 % bootstrap intervals in brackets. Full tables for every split and
feature set are in `results/RESULTS.md`.

| model | PR-AUC | ROC-AUC | F2, own threshold | F2 at LRP's alert count (178) | precision / recall at own threshold |
|---|---|---|---|---|---|
| Latest risk, decided at −6 (ESA's LRP) | 0.665 [0.591, 0.739] | 0.944 | **0.739** [0.680, 0.795] | 0.739 | 0.65 / 0.77 |
| Latest risk, threshold tuned by inner CV (−6.04) | 0.665 | 0.944 | **0.753** [0.680, 0.804] | 0.739 | 0.64 / 0.79 |
| Logistic regression | 0.574 [0.498, 0.653] | 0.938 | 0.651 [0.593, 0.711] | 0.649 | 0.57 / 0.67 |
| Decision tree | 0.554 [0.474, 0.627] | 0.912 | 0.572 [0.497, 0.640] | 0.636 | 0.70 / 0.55 |
| SVM (RBF) | 0.586 [0.507, 0.674] | 0.934 | 0.522 [0.458, 0.580] | 0.623 | 0.59 / 0.51 |
| Neural network (mean of 3 seeds; bracket = first seed's bootstrap) | 0.512 [0.426, 0.579], seed sd 0.019 | 0.914 | 0.524, seed sd 0.099 | 0.572 | 0.51 / 0.54 |
| Gradient boosting (classifier) | **0.673** [0.597, 0.749] | **0.950** | 0.727 [0.666, 0.779] | 0.681 | 0.59 / 0.77 |
| Boosting on Δrisk, decided at −6 (paper's framing) | 0.438 [0.361, 0.522] | 0.883 | 0.313 [0.237, 0.385] | 0.437 | 0.75 / 0.27 |
| Constant score | 0.069 | 0.500 | 0.271 | 0.084 | 0.07 / 1.00 |
| Shuffled-label control (3 permutations × 2 models) | 0.060 | 0.431 | 0.233 | 0.051 | – |

Latest-CDM features only: boosting 0.696 [0.611, 0.769] / F2 at LRP's count
0.713; SVM 0.580 / 0.636; logistic regression 0.570 / 0.656; tree 0.530 /
0.643; network 0.475 / 0.555; Δrisk boosting 0.444. Without the risk
columns: boosting 0.502 [0.419, 0.580] / 0.521, Δrisk boosting 0.476, the
course models 0.32–0.39, all far above the control (0.063) and far below
LRP.

ESA's MSE_HR on this test set: naive forecast 0.513 (the paper: 0.513),
Δrisk boosting 1.516; the competition winner reached 0.407 with hand-set
threshold rules. The tuned LRP threshold found by inner CV, −6.04, is the
winning team's "step 0" rule and gives the same F2 they report for it,
0.753.

### Three splits, history features (mean ± sd over folds; official is a point estimate)

PR-AUC, then F2 at the model's own threshold. "vs LRP" is the paired
difference on the same test events and the number of folds the model wins.

| model | ESA split | stratified 5-fold × 5 | vs LRP (25 folds) | satellites held out | vs LRP (5 folds) |
|---|---|---|---|---|---|
| LRP (latest risk) | 0.665 / 0.739 | 0.394 ± 0.076 / 0.600 ± 0.041 | – | 0.353 ± 0.159 / 0.535 ± 0.200 | – |
| Gradient boosting | 0.673 / 0.727 | **0.474 ± 0.065** / 0.599 ± 0.039 | AP +0.080 ± 0.061, 21/25; F2 −0.001, 14/25 | **0.412 ± 0.160** / 0.533 ± 0.200 | AP +0.059 ± 0.043, 4/5; F2 −0.002, 3/5 |
| SVM (RBF) | 0.586 / 0.522 | 0.410 ± 0.076 / 0.518 ± 0.043 | AP +0.016 ± 0.049, 15/25; F2 −0.081, 1/25 | 0.330 ± 0.154 / 0.435 ± 0.177 | AP −0.023, 2/5; F2 −0.099, 0/5 |
| Boosting on Δrisk (−6) | 0.438 / 0.313 | 0.386 ± 0.097 / 0.463 ± 0.087 | AP −0.008, 11/25; F2 −0.136, 1/25 | 0.351 ± 0.103 / 0.397 ± 0.134 | AP −0.001, 3/5; F2 −0.138, 1/5 |
| Decision tree | 0.554 / 0.572 | 0.347 ± 0.058 / 0.566 ± 0.041 | AP −0.047, 6/25; F2 −0.034, 6/25 | 0.295 ± 0.121 / 0.502 ± 0.195 | AP −0.058, 0/5; F2 −0.033, 0/5 |
| Logistic regression | 0.574 / 0.651 | 0.345 ± 0.074 / 0.510 ± 0.052 | AP −0.049, 6/25; F2 −0.090, 1/25 | 0.279 ± 0.140 / 0.416 ± 0.187 | AP −0.074, 0/5; F2 −0.119, 0/5 |
| Neural network | 0.512 / 0.524 | 0.332 ± 0.070 / 0.471 ± 0.058 | AP −0.062, 5/25; F2 −0.129, 0/25 | 0.249 ± 0.126 / 0.376 ± 0.175 | AP −0.103, 0/5; F2 −0.158, 0/5 |
| Shuffled control (6 permutation runs per fold; 5 folds on the CV splits) | 0.060 / 0.233 | 0.022 ± 0.002 / 0.093 ± 0.002 | – | 0.024 ± 0.008 / 0.088 ± 0.031 | – |
| base rate | 0.069 | 0.021 | | 0.007–0.030 | |

Mission folds (history features, LRP → boosting PR-AUC): missions 1, 4, 18,
20, 22, 24 (62 positives) 0.421 → 0.541; 9, 10, 15, 16 (44) 0.409 → 0.399;
3, 5, 13, 14, 23 (43) 0.337 → 0.421; 2, 19 (52) 0.536 → 0.577; 6, 7
(15 positives in 2,145 events) 0.061 → 0.122.

### Feature sets, stratified CV, PR-AUC (history: 25 folds; others: 5)

| model | latest CDM | latest + history | history without risk columns |
|---|---|---|---|
| LRP | 0.398 ± 0.105 | 0.394 ± 0.076 | 0.398 ± 0.105 |
| Gradient boosting | 0.461 ± 0.074 | 0.474 ± 0.065 | 0.340 ± 0.062 |
| SVM (RBF) | 0.410 ± 0.065 | 0.410 ± 0.076 | 0.219 ± 0.035 |
| Boosting on Δrisk | 0.396 ± 0.111 | 0.386 ± 0.097 | 0.377 ± 0.113 |
| Decision tree | 0.339 ± 0.076 | 0.347 ± 0.058 | 0.171 ± 0.013 |
| Neural network | 0.347 ± 0.068 | 0.332 ± 0.070 | 0.227 ± 0.032 |
| Logistic regression | 0.336 ± 0.060 | 0.345 ± 0.074 | 0.195 ± 0.015 |
| Shuffled control (5 folds) | 0.023 ± 0.003 | 0.022 ± 0.002 | 0.023 ± 0.003 |

### The paper's experiment, re-run (`paper_check.py`)

505 stratified 80/20 splits of the 10,460 events, 3 models per split,
latest CDM's 100 numeric columns, regressor on Δrisk with a
quantile-transformed target, decided at −6. "Shift" is how far, in log10
risk, the model moves the forecast away from the latest risk.

| | F2 (mean ± sd over splits) | gain over LRP | splits better | paired t, p | mean shift |
|---|---|---|---|---|---|
| Naive forecast (LRP) | 0.598 ± 0.046 (paper 0.600 ± 0.044) | – | – | – | 0 |
| LightGBM exactly as in the paper (25 trees, lr 0.05, feature_fraction 0.25) | **0.604 ± 0.048** (paper 0.607 ± 0.049) | **+1.00 %** (paper +1.27 %) | 326 / 505 | +7.0, 9 × 10⁻¹² (paper +10.2, 2 × 10⁻²²) | 0.14 |
| scikit-learn HGB, same settings | 0.573 ± 0.057 | −4.2 % | 135 / 505 | −14.0, 10⁻³⁷ | 0.57 |
| scikit-learn HGB, this project's settings (150 trees, lr 0.1) | 0.468 ± 0.068 | −21.7 % | 5 / 505 | −46.1, 7 × 10⁻¹⁸³ | 2.92 |

### What the numbers support

1. **Nothing here beats the naive forecast at the decision point.** On
   ESA's test set LRP's F2 is 0.739 [0.680, 0.795]; the best learned model,
   boosting, reaches 0.727 [0.666, 0.779] at its own threshold (116 of the
   150 high-risk events caught against LRP's 115, with 82 false alarms
   against 63) and 0.681 when it is allowed exactly LRP's 178 alerts. Over
   25 stratified folds boosting's F2 is 0.599 ± 0.039 against LRP's 0.600 ±
   0.041, better in 14 of 25 folds and worse in 11: a tie. With satellites
   held out 0.533 vs 0.535. The simplest improvement is to lower LRP's
   threshold: −6.04 chosen by inner CV gives 0.753 on ESA's test set, which
   is what the competition winner's first rule does.
2. **Boosting ranks events better than the latest risk, and the margin is
   only visible with many folds.** PR-AUC is a near-tie on ESA's split
   (0.673 vs 0.665, intervals [0.597, 0.749] and [0.591, 0.739]), +0.080 ±
   0.061 and better in 21 of 25 stratified folds, +0.059 and better in 4 of
   5 mission folds. The ranking advantage does not turn into a better alert
   list at the operating point: the gains are at recall 0.2–0.5
   (`figures/pr_curves_official.png`), while at recall 0.75–0.9, where the
   −6 decision sits, LRP is as good or better.
3. **Hyperparameter tuning helped the course models and left the ordering
   unchanged.** Against the fixed settings of the first run, tuning raised
   the cross-validated PR-AUC of logistic regression from 0.29 to 0.34, the
   tree from 0.28 to 0.35 and the SVM from 0.36 to 0.41 (strong
   regularisation won every time: C = 0.01, 100 rows per leaf, C = 0.3 with
   a wide kernel, γ = 0.001, below the default 1/216), while the network's chosen settings were the original
   ones and boosting moved by less than its fold sd. The SVM is now the best
   of the four course models on ranking (it ties LRP, +0.016, 15 of 25
   folds) and the tree is the best of the four on F2 (0.566, because it
   effectively relearns the −6 rule), but all four lose to LRP on F2 in at
   least 19 of 25 folds.
4. **Thresholds do not transfer across ESA's split.** Every model's
   threshold is learned on 66 positives at a 0.8 % base rate and applied at
   6.9 %. The SVM's own-threshold F2 (0.522) is 0.10 below its F2 at LRP's
   alert count (0.623); for the tree the gap is 0.06. Probabilities trained
   with balanced class weights at 0.8 % are not calibrated at 6.9 %:
   logistic regression and the tree over-predict (a score of 0.8 corresponds
   to 20–50 % observed high risk), the network and boosting sit closer to
   the diagonal (`figures/calibration.png`). The paper notes the same
   base-rate difference between ESA's training and test sets.
5. **The other CDM fields carry real but much smaller information.**
   Without any risk column boosting still reaches PR-AUC 0.340 ± 0.062
   (control 0.023) and 0.502 on ESA's test set, so covariance, miss distance
   and orbit-determination quality do predict the final risk, but less than
   the latest risk alone does (0.398), and far less than the two together
   (0.474). Permutation importance of the full model on ESA's test set:
   `risk` 0.32, `risk_mean` 0.053, `max_risk_scaling` 0.018, `miss_distance`
   0.015, then everything else below 0.01 (`figures/importance.png`); the
   paper's relevance table also puts `risk` and `max_risk_scaling` first.
6. **History features add nothing measurable.** Latest-CDM boosting 0.461 ±
   0.074 vs 0.474 ± 0.065 with history in CV, 0.696 vs 0.673 on ESA's split
   (opposite signs, both inside the fold sd). The series mostly repeats the
   latest value: the median event has 14 CDMs and the risk moves little
   until the last two days.
7. **The paper's 1.3 % gain is real and fragile.** It reproduces (+1.00 %,
   326 of 505 splits, p = 9 × 10⁻¹²) with the paper's exact LightGBM, whose
   25 shallow trees each see a quarter of the columns and move the forecast
   by 0.14 on average. The same hyper-parameters in scikit-learn (where
   `max_features` samples per split, not per tree) move it 0.57 and lose
   4 %; the 150-tree regressor of `models.py` moves it 2.9 and loses 22 %,
   which is why the Δrisk rows in the main tables are poor. The lesson is
   that on this data the best ML model is one that almost reproduces the
   naive forecast; the room above it is about one percent of F2.
8. **The network is the weakest and least stable model.** PR-AUC 0.33 in
   CV against LRP's 0.39, and its three seeds on ESA's split differ by 0.019
   in PR-AUC but by 0.10 in own-threshold F2 (0.417, 0.500, 0.656), so its
   threshold is the least reliable; the chosen settings (64–32, 1e-3) were
   the best of six by inner CV but every setting was within noise.
9. **Controls and calibration.** The shuffled-label control sits at the base
   rate in PR-AUC (0.059–0.063 at 6.9 %, 0.022–0.024 at 2.1 %) and at
   0.43–0.49 ROC in every split. Boosting has the best Brier score on ESA's
   test set (0.036) against 0.051–0.063 for the other probabilistic models
   and 0.068 for the constant.

### Corrections made during the work

The first version of the naive baseline read the *earliest* CDM instead of
the latest usable one (a `.last()` after a descending sort) and gave F2
about 0.32; fixed, it reproduces the published 0.739 and 0.600 exactly. The
first Δrisk model regressed on raw h and flagged no event at all on ESA's
test set (most h are 0 or a collapse to −30); the paper's
quantile-transformed target was adopted. Merging the 25-fold run assumed
every fold had a control and crashed; fixed so each model is averaged over
the folds that ran it. The first complete run used hyperparameters fixed in
advance (logistic C = 1, tree 25 rows per leaf, SVM C = 1 with γ = scale,
boosting learning rate 0.1 with 15 leaves); when the rubric notebook was
written, the hyperparameter search of its section 5.2 chose different
values for four of the five models (only the network kept its settings),
`models.py` was updated to the chosen values and all 92 fold jobs, merges
and figures were regenerated; the first run survives only as the "before"
values in finding 3. One last edit to `models.py` (switching on
TensorFlow's deterministic ops, so that two runs give identical network
numbers) came after those jobs: three folds re-run with it reproduce their
saved metrics to the last digit, the nine `metrics_*.json` manifests and
the figures were regenerated with it, and the other 89 `fold_*.json` files
carry the previous `models.py` hash.

## Honest limits

216 positive events, 150 of them in ESA's test set and 66 in its training
set; bootstrap intervals on the test set are ± 0.08 wide in PR-AUC and the
learned models' thresholds rest on 66 examples. There are no dates, so a
forward-in-time split is impossible; the stratified folds mix years.
Mission folds have 15 to 62 positives, and their sd is large by
construction. The hyperparameters were chosen once, on the training 80 %
of one stratified split, and then used on every fold of the 25-fold and
mission studies, whose test folds overlap that 80 %; this is a mild
optimism for the learned models (not for LRP, which has nothing to tune)
and would be removed by a nested search, at five times the compute. All
features come from the latest usable CDM and its history and ignore
anything an operator would also know (the catalogue, the manoeuvre already
planned). F2 and MSE_HR are ESA's choices, not an operational cost.

## How this maps to the course rubric

| rubric item | where it is |
|---|---|
| Problem statement and objectives (week 6) | notebook section 1 |
| Data preparation: cleaning, missing values, encoding; cleaned dataset file (week 7) | notebook section 2; `build_table.py`; `data/processed/events.csv.gz` |
| EDA: distributions, trends, relationships, observations, features to engineer (week 8) | notebook section 3 |
| Feature engineering: scaling, selection, rationale; feature-engineered dataset (week 9) | notebook sections 4 and 7.4 (selection experiment); `data/processed/features_train.csv.gz`, `features_test.csv.gz` |
| Model building: split, four models (logistic, tree, network, SVM) plus boosting, hyperparameter optimisation with rationale (week 10) | notebook section 5; `results/tuned_params.json`; `models.py` |
| Evaluation: accuracy, F1 and the imbalance-aware metrics, comparison, best model (week 11) | notebook section 6 (6.5: five-fold cross-validation inside the notebook); `results/notebook_test_metrics.csv` |
| Insights and visualisations (weeks 12–13) | notebook section 7; `figures/`; this README |
| Conclusion (week 14) | notebook section 8 |
| Runnable code with the dataset | `CS465_Project.ipynb` + `data/processed/` (the raw 221 MB database via `get_data.py`) |

**What to submit.** Zip the whole `satellite-collision/` folder as it is on
disk (the `.gitignore` only affects git; `data/processed/` with the cleaned
and feature-engineered tables, and `results/nb_cache/`, must be in the
zip). Without the raw database the notebook still runs end to end from the
shipped table; the two raw-file cells say so and are skipped.

## Provenance

ESA Collision Avoidance Challenge dataset, Zenodo record 4463683, CC BY
4.0; data originally from ESA's Space Debris Office, built on orbital data
from the US Space Surveillance Network. Paper: T. Uriot, D. Izzo, L. F.
Simões, R. Abay, N. Einecke, S. Rebhan, J. Martinez-Heras, F. Letizia,
J. Siminski, K. Merz, Astrodynamics 6(2), 121–140, 2022,
doi:10.1007/s42064-021-0101-5 (arXiv:2008.03069). Competition page:
kelvins.esa.int/collision-avoidance-challenge.
