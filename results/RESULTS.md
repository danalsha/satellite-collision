# Results

Eligible events: 10,460 (216 high risk, base rate 0.0207). Every number is on events the model never saw. On the official split the brackets are 95 % bootstrap intervals over the 2,167 test events (200 resamples); the network's point estimates are means over 3 seeds and its brackets are the first seed's bootstrap. On the cross-validated splits ± is the standard deviation over the folds (25 folds = 5 repeats of a stratified 5-fold split for the history feature set, 5 folds otherwise; 5 folds with satellites held out; the shuffled-label control runs on 5 folds of every split). The last two columns compare each model with the naive forecast fold by fold, on the same test events. F2 (own threshold) uses a threshold chosen by 3-fold cross-validation inside the training partition; 'F2 at LRP's alert count' flags exactly as many events as the naive forecast flags at -6, so it compares rankings without any threshold.

## ESA's competition split

### Features: latest CDM (100 numeric + 1 categorical)

Train 8,293 events (66 high risk), test 2,167 (150 high risk, base rate 0.069).

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.665 [0.591, 0.739] | 0.944 | 0.739 [0.680, 0.795] | 0.739 | 0.65 | 0.77 | – | – | – |
| Latest risk, tuned threshold | 0.665 [0.588, 0.737] | 0.944 | 0.753 [0.680, 0.804] | 0.739 | 0.64 | 0.79 | – | – | – |
| Boosting on the change in risk, at -6 (paper's framing) | 0.444 [0.352, 0.522] | 0.881 | 0.321 [0.248, 0.398] | 0.443 | 0.76 | 0.28 | – | – | – |
| Boosting on the change in risk, tuned threshold | 0.444 [0.362, 0.541] | 0.881 | 0.325 [0.253, 0.402] | 0.443 | 0.70 | 0.29 | – | – | – |
| Logistic regression | 0.570 [0.502, 0.654] | 0.940 | 0.598 [0.532, 0.661] | 0.656 | 0.59 | 0.60 | 0.063 | – | – |
| Decision tree | 0.530 [0.454, 0.614] | 0.911 | 0.434 [0.349, 0.491] | 0.643 | 0.65 | 0.40 | 0.064 | – | – |
| SVM (RBF) | 0.580 [0.499, 0.664] | 0.937 | 0.531 [0.468, 0.601] | 0.636 | 0.66 | 0.51 | – | – | – |
| Neural network | 0.475 (seed sd 0.007; first-seed CI [0.402, 0.552]) | 0.904 | 0.481 (seed sd 0.087; first-seed CI [0.366, 0.514]) | 0.555 | 0.50 | 0.48 | 0.054 | – | – |
| Gradient boosting (classifier) | 0.696 [0.611, 0.769] | 0.948 | 0.596 [0.507, 0.661] | 0.713 | 0.71 | 0.57 | 0.034 | – | – |
| Constant score | 0.069 [0.059, 0.079] | 0.500 | 0.271 [0.237, 0.301] | 0.084 | 0.07 | 1.00 | 0.068 | – | – |
| Shuffled-label control | 0.059 | 0.427 | 0.211 | 0.048 | – | – | – | – | – |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.513, boosting on the change in risk 1.484 (the paper: LRP 0.513, winning team 0.407).

### Features: latest + history (166 numeric + 1 categorical)

Train 8,293 events (66 high risk), test 2,167 (150 high risk, base rate 0.069).

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.665 [0.591, 0.739] | 0.944 | 0.739 [0.680, 0.795] | 0.739 | 0.65 | 0.77 | – | – | – |
| Latest risk, tuned threshold | 0.665 [0.588, 0.737] | 0.944 | 0.753 [0.680, 0.804] | 0.739 | 0.64 | 0.79 | – | – | – |
| Boosting on the change in risk, at -6 (paper's framing) | 0.438 [0.361, 0.522] | 0.883 | 0.313 [0.237, 0.385] | 0.437 | 0.75 | 0.27 | – | – | – |
| Boosting on the change in risk, tuned threshold | 0.438 [0.356, 0.532] | 0.883 | 0.330 [0.265, 0.410] | 0.437 | 0.67 | 0.29 | – | – | – |
| Logistic regression | 0.574 [0.498, 0.653] | 0.938 | 0.651 [0.593, 0.711] | 0.649 | 0.57 | 0.67 | 0.059 | – | – |
| Decision tree | 0.554 [0.474, 0.627] | 0.912 | 0.572 [0.497, 0.640] | 0.636 | 0.70 | 0.55 | 0.063 | – | – |
| SVM (RBF) | 0.586 [0.507, 0.674] | 0.934 | 0.522 [0.458, 0.580] | 0.623 | 0.59 | 0.51 | – | – | – |
| Neural network | 0.512 (seed sd 0.019; first-seed CI [0.426, 0.579]) | 0.914 | 0.524 (seed sd 0.099; first-seed CI [0.346, 0.495]) | 0.572 | 0.51 | 0.54 | 0.051 | – | – |
| Gradient boosting (classifier) | 0.673 [0.597, 0.749] | 0.950 | 0.727 [0.666, 0.779] | 0.681 | 0.59 | 0.77 | 0.036 | – | – |
| Constant score | 0.069 [0.059, 0.079] | 0.500 | 0.271 [0.237, 0.301] | 0.084 | 0.07 | 1.00 | 0.068 | – | – |
| Shuffled-label control | 0.060 | 0.431 | 0.233 | 0.051 | – | – | – | – | – |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.513, boosting on the change in risk 1.516 (the paper: LRP 0.513, winning team 0.407).

### Features: history, no risk columns (148 numeric + 1 categorical)

Train 8,293 events (66 high risk), test 2,167 (150 high risk, base rate 0.069).

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.665 [0.591, 0.739] | 0.944 | 0.739 [0.680, 0.795] | 0.739 | 0.65 | 0.77 | – | – | – |
| Latest risk, tuned threshold | 0.665 [0.588, 0.737] | 0.944 | 0.753 [0.680, 0.804] | 0.739 | 0.64 | 0.79 | – | – | – |
| Boosting on the change in risk, at -6 (paper's framing) | 0.476 [0.389, 0.552] | 0.862 | 0.407 [0.323, 0.492] | 0.443 | 0.72 | 0.37 | – | – | – |
| Boosting on the change in risk, tuned threshold | 0.476 [0.404, 0.571] | 0.862 | 0.443 [0.376, 0.523] | 0.443 | 0.63 | 0.41 | – | – | – |
| Logistic regression | 0.365 [0.306, 0.449] | 0.890 | 0.389 [0.321, 0.469] | 0.482 | 0.43 | 0.38 | 0.088 | – | – |
| Decision tree | 0.318 [0.256, 0.386] | 0.805 | 0.311 [0.234, 0.382] | 0.456 | 0.55 | 0.28 | 0.111 | – | – |
| SVM (RBF) | 0.386 [0.317, 0.462] | 0.883 | 0.510 [0.438, 0.582] | 0.450 | 0.38 | 0.56 | – | – | – |
| Neural network | 0.326 (seed sd 0.017; first-seed CI [0.287, 0.426]) | 0.850 | 0.345 (seed sd 0.082; first-seed CI [0.215, 0.355]) | 0.413 | 0.38 | 0.35 | 0.063 | – | – |
| Gradient boosting (classifier) | 0.502 [0.419, 0.580] | 0.909 | 0.507 [0.428, 0.575] | 0.521 | 0.48 | 0.51 | 0.047 | – | – |
| Constant score | 0.069 [0.059, 0.079] | 0.500 | 0.271 [0.237, 0.301] | 0.084 | 0.07 | 1.00 | 0.068 | – | – |
| Shuffled-label control | 0.063 | 0.447 | 0.261 | 0.062 | – | – | – | – | – |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.513, boosting on the change in risk 1.389 (the paper: LRP 0.513, winning team 0.407).

## stratified 5-fold

### Features: latest CDM (100 numeric + 1 categorical)

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.398 ± 0.105 | 0.949 ± 0.008 | 0.600 ± 0.043 | 0.600 | 0.33 | 0.75 | – | – | – |
| Latest risk, tuned threshold | 0.398 ± 0.105 | 0.949 ± 0.008 | 0.615 ± 0.042 | 0.600 | 0.33 | 0.78 | – | +0.000 ± 0.000, better in 0/5 folds | +0.000 ± 0.000, better in 0/5 folds |
| Boosting on the change in risk, at -6 (paper's framing) | 0.396 ± 0.111 | 0.913 ± 0.009 | 0.462 ± 0.078 | 0.433 | 0.55 | 0.44 | – | -0.002 ± 0.053, better in 2/5 folds | -0.167 ± 0.047, better in 0/5 folds |
| Boosting on the change in risk, tuned threshold | 0.396 ± 0.111 | 0.913 ± 0.009 | 0.472 ± 0.089 | 0.433 | 0.47 | 0.48 | – | -0.002 ± 0.053, better in 2/5 folds | -0.167 ± 0.047, better in 0/5 folds |
| Logistic regression | 0.336 ± 0.060 | 0.944 ± 0.009 | 0.521 ± 0.055 | 0.511 | 0.25 | 0.72 | 0.079 | -0.062 ± 0.069, better in 1/5 folds | -0.089 ± 0.025, better in 0/5 folds |
| Decision tree | 0.339 ± 0.076 | 0.899 ± 0.031 | 0.578 ± 0.045 | 0.570 | 0.30 | 0.76 | 0.071 | -0.059 ± 0.068, better in 1/5 folds | -0.030 ± 0.039, better in 1/5 folds |
| SVM (RBF) | 0.410 ± 0.065 | 0.949 ± 0.009 | 0.524 ± 0.039 | 0.526 | 0.26 | 0.71 | – | +0.012 ± 0.060, better in 2/5 folds | -0.074 ± 0.017, better in 0/5 folds |
| Neural network | 0.347 ± 0.068 | 0.922 ± 0.008 | 0.486 ± 0.060 | 0.489 | 0.22 | 0.71 | 0.032 | -0.051 ± 0.050, better in 1/5 folds | -0.111 ± 0.039, better in 0/5 folds |
| Gradient boosting (classifier) | 0.461 ± 0.074 | 0.954 ± 0.012 | 0.594 ± 0.040 | 0.615 | 0.29 | 0.81 | 0.034 | +0.063 ± 0.059, better in 4/5 folds | +0.015 ± 0.014, better in 3/5 folds |
| Constant score | 0.021 ± 0.000 | 0.500 ± 0.000 | 0.095 ± 0.001 | 0.055 | 0.02 | 1.00 | 0.020 | -0.378 ± 0.105, better in 0/5 folds | -0.544 ± 0.052, better in 0/5 folds |
| Shuffled-label control | 0.023 ± 0.003 | 0.483 ± 0.042 | 0.093 ± 0.003 | 0.039 | – | – | – | -0.375 ± 0.106, better in 0/5 folds | -0.561 ± 0.052, better in 0/5 folds |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.457, boosting on the change in risk 0.997.

### Features: latest + history (166 numeric + 1 categorical)

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.394 ± 0.076 | 0.949 ± 0.013 | 0.600 ± 0.041 | 0.600 | 0.33 | 0.75 | – | – | – |
| Latest risk, tuned threshold | 0.394 ± 0.076 | 0.949 ± 0.013 | 0.615 ± 0.036 | 0.600 | 0.34 | 0.78 | – | +0.000 ± 0.000, better in 0/25 folds | +0.000 ± 0.000, better in 0/25 folds |
| Boosting on the change in risk, at -6 (paper's framing) | 0.386 ± 0.097 | 0.916 ± 0.021 | 0.463 ± 0.087 | 0.453 | 0.55 | 0.45 | – | -0.008 ± 0.077, better in 11/25 folds | -0.147 ± 0.071, better in 0/25 folds |
| Boosting on the change in risk, tuned threshold | 0.386 ± 0.097 | 0.916 ± 0.021 | 0.483 ± 0.089 | 0.453 | 0.45 | 0.50 | – | -0.008 ± 0.077, better in 11/25 folds | -0.147 ± 0.071, better in 0/25 folds |
| Logistic regression | 0.345 ± 0.074 | 0.948 ± 0.011 | 0.510 ± 0.052 | 0.505 | 0.25 | 0.69 | 0.077 | -0.049 ± 0.058, better in 6/25 folds | -0.095 ± 0.059, better in 1/25 folds |
| Decision tree | 0.347 ± 0.058 | 0.910 ± 0.021 | 0.566 ± 0.041 | 0.565 | 0.30 | 0.73 | 0.071 | -0.047 ± 0.067, better in 6/25 folds | -0.035 ± 0.050, better in 4/25 folds |
| SVM (RBF) | 0.410 ± 0.076 | 0.950 ± 0.012 | 0.518 ± 0.043 | 0.529 | 0.24 | 0.73 | – | +0.016 ± 0.049, better in 15/25 folds | -0.071 ± 0.055, better in 2/25 folds |
| Neural network | 0.332 ± 0.070 | 0.930 ± 0.017 | 0.471 ± 0.058 | 0.472 | 0.23 | 0.65 | 0.030 | -0.062 ± 0.063, better in 5/25 folds | -0.128 ± 0.082, better in 0/25 folds |
| Gradient boosting (classifier) | 0.474 ± 0.065 | 0.962 ± 0.013 | 0.599 ± 0.039 | 0.610 | 0.31 | 0.79 | 0.032 | +0.080 ± 0.061, better in 21/25 folds | +0.010 ± 0.029, better in 12/25 folds |
| Constant score | 0.021 ± 0.000 | 0.500 ± 0.000 | 0.095 ± 0.001 | 0.059 | 0.02 | 1.00 | 0.020 | -0.373 ± 0.076, better in 0/25 folds | -0.541 ± 0.050, better in 0/25 folds |
| Shuffled-label control | 0.022 ± 0.002 | 0.478 ± 0.037 | 0.093 ± 0.002 | 0.034 | – | – | – | -0.376 ± 0.106, better in 0/5 folds | -0.566 ± 0.043, better in 0/5 folds |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.457, boosting on the change in risk 1.583.

### Features: history, no risk columns (148 numeric + 1 categorical)

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.398 ± 0.105 | 0.949 ± 0.008 | 0.600 ± 0.043 | 0.600 | 0.33 | 0.75 | – | – | – |
| Latest risk, tuned threshold | 0.398 ± 0.105 | 0.949 ± 0.008 | 0.615 ± 0.042 | 0.600 | 0.33 | 0.78 | – | +0.000 ± 0.000, better in 0/5 folds | +0.000 ± 0.000, better in 0/5 folds |
| Boosting on the change in risk, at -6 (paper's framing) | 0.377 ± 0.113 | 0.908 ± 0.016 | 0.464 ± 0.056 | 0.481 | 0.50 | 0.46 | – | -0.021 ± 0.072, better in 2/5 folds | -0.119 ± 0.067, better in 0/5 folds |
| Boosting on the change in risk, tuned threshold | 0.377 ± 0.113 | 0.908 ± 0.016 | 0.509 ± 0.059 | 0.481 | 0.40 | 0.55 | – | -0.021 ± 0.072, better in 2/5 folds | -0.119 ± 0.067, better in 0/5 folds |
| Logistic regression | 0.195 ± 0.015 | 0.909 ± 0.007 | 0.408 ± 0.012 | 0.378 | 0.16 | 0.67 | 0.103 | -0.204 ± 0.102, better in 0/5 folds | -0.222 ± 0.047, better in 0/5 folds |
| Decision tree | 0.171 ± 0.013 | 0.868 ± 0.027 | 0.328 ± 0.036 | 0.359 | 0.24 | 0.41 | 0.119 | -0.228 ± 0.101, better in 0/5 folds | -0.241 ± 0.033, better in 0/5 folds |
| SVM (RBF) | 0.219 ± 0.035 | 0.910 ± 0.009 | 0.377 ± 0.025 | 0.356 | 0.16 | 0.60 | – | -0.180 ± 0.132, better in 1/5 folds | -0.244 ± 0.050, better in 0/5 folds |
| Neural network | 0.227 ± 0.032 | 0.885 ± 0.014 | 0.370 ± 0.030 | 0.366 | 0.16 | 0.58 | 0.033 | -0.172 ± 0.095, better in 0/5 folds | -0.234 ± 0.045, better in 0/5 folds |
| Gradient boosting (classifier) | 0.340 ± 0.062 | 0.927 ± 0.010 | 0.422 ± 0.043 | 0.441 | 0.17 | 0.70 | 0.051 | -0.058 ± 0.074, better in 1/5 folds | -0.159 ± 0.018, better in 0/5 folds |
| Constant score | 0.021 ± 0.000 | 0.500 ± 0.000 | 0.095 ± 0.001 | 0.055 | 0.02 | 1.00 | 0.020 | -0.378 ± 0.105, better in 0/5 folds | -0.544 ± 0.052, better in 0/5 folds |
| Shuffled-label control | 0.023 ± 0.003 | 0.489 ± 0.039 | 0.093 ± 0.002 | 0.035 | – | – | – | -0.375 ± 0.105, better in 0/5 folds | -0.565 ± 0.040, better in 0/5 folds |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.457, boosting on the change in risk 1.057.

## satellites held out

### Features: latest CDM (100 numeric + 1 categorical)

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.353 ± 0.159 | 0.926 ± 0.063 | 0.535 ± 0.200 | 0.535 | 0.31 | 0.66 | – | – | – |
| Latest risk, tuned threshold | 0.353 ± 0.159 | 0.926 ± 0.063 | 0.547 ± 0.207 | 0.535 | 0.30 | 0.69 | – | +0.000 ± 0.000, better in 0/5 folds | +0.000 ± 0.000, better in 0/5 folds |
| Boosting on the change in risk, at -6 (paper's framing) | 0.331 ± 0.153 | 0.875 ± 0.091 | 0.367 ± 0.154 | 0.400 | 0.45 | 0.35 | – | -0.022 ± 0.054, better in 1/5 folds | -0.135 ± 0.067, better in 0/5 folds |
| Boosting on the change in risk, tuned threshold | 0.331 ± 0.153 | 0.875 ± 0.091 | 0.411 ± 0.179 | 0.400 | 0.37 | 0.42 | – | -0.022 ± 0.054, better in 1/5 folds | -0.135 ± 0.067, better in 0/5 folds |
| Logistic regression | 0.293 ± 0.137 | 0.906 ± 0.075 | 0.419 ± 0.159 | 0.426 | 0.20 | 0.62 | 0.094 | -0.060 ± 0.045, better in 1/5 folds | -0.108 ± 0.036, better in 0/5 folds |
| Decision tree | 0.275 ± 0.116 | 0.886 ± 0.069 | 0.517 ± 0.199 | 0.504 | 0.26 | 0.70 | 0.089 | -0.078 ± 0.062, better in 0/5 folds | -0.031 ± 0.040, better in 1/5 folds |
| SVM (RBF) | 0.343 ± 0.150 | 0.919 ± 0.068 | 0.435 ± 0.166 | 0.433 | 0.22 | 0.60 | – | -0.010 ± 0.041, better in 2/5 folds | -0.101 ± 0.034, better in 0/5 folds |
| Neural network | 0.234 ± 0.139 | 0.848 ± 0.104 | 0.340 ± 0.134 | 0.385 | 0.19 | 0.52 | 0.034 | -0.118 ± 0.071, better in 0/5 folds | -0.150 ± 0.095, better in 0/5 folds |
| Gradient boosting (classifier) | 0.377 ± 0.137 | 0.939 ± 0.047 | 0.546 ± 0.192 | 0.539 | 0.28 | 0.73 | 0.033 | +0.025 ± 0.059, better in 3/5 folds | +0.004 ± 0.015, better in 2/5 folds |
| Constant score | 0.021 ± 0.008 | 0.500 ± 0.000 | 0.095 ± 0.033 | 0.056 | 0.02 | 1.00 | 0.020 | -0.332 ± 0.152, better in 0/5 folds | -0.479 ± 0.194, better in 0/5 folds |
| Shuffled-label control | 0.023 ± 0.005 | 0.453 ± 0.028 | 0.089 ± 0.031 | 0.040 | – | – | – | -0.329 ± 0.155, better in 0/5 folds | -0.495 ± 0.202, better in 0/5 folds |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.489, boosting on the change in risk 0.989.

### Features: latest + history (166 numeric + 1 categorical)

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.353 ± 0.159 | 0.926 ± 0.063 | 0.535 ± 0.200 | 0.535 | 0.31 | 0.66 | – | – | – |
| Latest risk, tuned threshold | 0.353 ± 0.159 | 0.926 ± 0.063 | 0.547 ± 0.207 | 0.535 | 0.30 | 0.69 | – | +0.000 ± 0.000, better in 0/5 folds | +0.000 ± 0.000, better in 0/5 folds |
| Boosting on the change in risk, at -6 (paper's framing) | 0.351 ± 0.103 | 0.888 ± 0.058 | 0.397 ± 0.134 | 0.378 | 0.50 | 0.38 | – | -0.001 ± 0.078, better in 3/5 folds | -0.157 ± 0.094, better in 0/5 folds |
| Boosting on the change in risk, tuned threshold | 0.351 ± 0.103 | 0.888 ± 0.058 | 0.415 ± 0.157 | 0.378 | 0.37 | 0.43 | – | -0.001 ± 0.078, better in 3/5 folds | -0.157 ± 0.094, better in 0/5 folds |
| Logistic regression | 0.279 ± 0.140 | 0.904 ± 0.083 | 0.416 ± 0.187 | 0.410 | 0.19 | 0.62 | 0.101 | -0.074 ± 0.047, better in 0/5 folds | -0.125 ± 0.053, better in 0/5 folds |
| Decision tree | 0.295 ± 0.121 | 0.865 ± 0.104 | 0.502 ± 0.195 | 0.506 | 0.27 | 0.66 | 0.076 | -0.058 ± 0.064, better in 0/5 folds | -0.029 ± 0.030, better in 0/5 folds |
| SVM (RBF) | 0.330 ± 0.154 | 0.914 ± 0.076 | 0.435 ± 0.177 | 0.425 | 0.22 | 0.60 | – | -0.023 ± 0.045, better in 2/5 folds | -0.110 ± 0.062, better in 0/5 folds |
| Neural network | 0.249 ± 0.126 | 0.865 ± 0.097 | 0.376 ± 0.175 | 0.386 | 0.22 | 0.50 | 0.029 | -0.103 ± 0.078, better in 0/5 folds | -0.149 ± 0.126, better in 1/5 folds |
| Gradient boosting (classifier) | 0.412 ± 0.160 | 0.947 ± 0.039 | 0.533 ± 0.200 | 0.538 | 0.28 | 0.70 | 0.031 | +0.059 ± 0.043, better in 4/5 folds | +0.003 ± 0.026, better in 2/5 folds |
| Constant score | 0.021 ± 0.008 | 0.500 ± 0.000 | 0.095 ± 0.033 | 0.056 | 0.02 | 1.00 | 0.020 | -0.332 ± 0.152, better in 0/5 folds | -0.479 ± 0.194, better in 0/5 folds |
| Shuffled-label control | 0.024 ± 0.008 | 0.452 ± 0.039 | 0.088 ± 0.031 | 0.037 | – | – | – | -0.328 ± 0.153, better in 0/5 folds | -0.498 ± 0.187, better in 0/5 folds |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.489, boosting on the change in risk 0.834.

### Features: history, no risk columns (148 numeric + 1 categorical)

| model | PR-AUC | ROC-AUC | F2 (own threshold) | F2 at LRP's alert count | precision | recall | Brier | PR-AUC vs LRP, paired | F2 at LRP's count vs LRP, paired |
|---|---|---|---|---|---|---|---|---|---|
| Latest risk, decided at -6 (ESA's LRP) | 0.353 ± 0.159 | 0.926 ± 0.063 | 0.535 ± 0.200 | 0.535 | 0.31 | 0.66 | – | – | – |
| Latest risk, tuned threshold | 0.353 ± 0.159 | 0.926 ± 0.063 | 0.547 ± 0.207 | 0.535 | 0.30 | 0.69 | – | +0.000 ± 0.000, better in 0/5 folds | +0.000 ± 0.000, better in 0/5 folds |
| Boosting on the change in risk, at -6 (paper's framing) | 0.319 ± 0.169 | 0.884 ± 0.062 | 0.415 ± 0.207 | 0.406 | 0.40 | 0.42 | – | -0.033 ± 0.063, better in 1/5 folds | -0.129 ± 0.074, better in 0/5 folds |
| Boosting on the change in risk, tuned threshold | 0.319 ± 0.169 | 0.884 ± 0.062 | 0.431 ± 0.200 | 0.406 | 0.32 | 0.48 | – | -0.033 ± 0.063, better in 1/5 folds | -0.129 ± 0.074, better in 0/5 folds |
| Logistic regression | 0.151 ± 0.075 | 0.846 ± 0.108 | 0.324 ± 0.137 | 0.281 | 0.13 | 0.57 | 0.123 | -0.201 ± 0.089, better in 0/5 folds | -0.254 ± 0.057, better in 0/5 folds |
| Decision tree | 0.135 ± 0.059 | 0.811 ± 0.061 | 0.301 ± 0.088 | 0.308 | 0.22 | 0.37 | 0.139 | -0.217 ± 0.107, better in 0/5 folds | -0.227 ± 0.096, better in 0/5 folds |
| SVM (RBF) | 0.183 ± 0.080 | 0.879 ± 0.054 | 0.323 ± 0.125 | 0.303 | 0.14 | 0.50 | – | -0.169 ± 0.087, better in 0/5 folds | -0.232 ± 0.076, better in 0/5 folds |
| Neural network | 0.137 ± 0.080 | 0.809 ± 0.103 | 0.261 ± 0.131 | 0.243 | 0.12 | 0.43 | 0.046 | -0.215 ± 0.087, better in 0/5 folds | -0.291 ± 0.083, better in 0/5 folds |
| Gradient boosting (classifier) | 0.243 ± 0.107 | 0.904 ± 0.041 | 0.380 ± 0.131 | 0.366 | 0.15 | 0.64 | 0.052 | -0.110 ± 0.087, better in 1/5 folds | -0.168 ± 0.111, better in 1/5 folds |
| Constant score | 0.021 ± 0.008 | 0.500 ± 0.000 | 0.095 ± 0.033 | 0.056 | 0.02 | 1.00 | 0.020 | -0.332 ± 0.152, better in 0/5 folds | -0.479 ± 0.194, better in 0/5 folds |
| Shuffled-label control | 0.023 ± 0.006 | 0.473 ± 0.041 | 0.091 ± 0.025 | 0.031 | – | – | – | -0.329 ± 0.153, better in 0/5 folds | -0.503 ± 0.193, better in 0/5 folds |

ESA's regression metric MSE_HR (over true high-risk events, predictions clipped at -6.001): naive forecast 0.489, boosting on the change in risk 0.928.

Per fold (history features), PR-AUC:

| fold | test events | high risk | base rate | missions in test | LRP | boosting | logistic | network |
|---|---|---|---|---|---|---|---|---|
| 0 | 2,071 | 62 | 0.030 | 1, 4, 18, 20, 22, 24 | 0.421 | 0.541 | 0.280 | 0.266 |
| 1 | 2,078 | 44 | 0.021 | 9, 10, 15, 16 | 0.409 | 0.399 | 0.325 | 0.187 |
| 2 | 2,071 | 43 | 0.021 | 3, 5, 13, 14, 23 | 0.337 | 0.421 | 0.332 | 0.289 |
| 3 | 2,095 | 52 | 0.025 | 2, 19 | 0.536 | 0.577 | 0.439 | 0.443 |
| 4 | 2,145 | 15 | 0.007 | 6, 7 | 0.061 | 0.122 | 0.018 | 0.060 |
