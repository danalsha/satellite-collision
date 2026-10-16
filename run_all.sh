#!/usr/bin/env bash
# Satellite collision project: everything after get_data.py has filled ./data.
# Folds run one per process; --merge collects them.
set -e
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
python3 build_table.py
python3 checksums.py
for fs in latest history no_risk; do
  python3 models.py $fs official --fold 0
  python3 models.py $fs official --fold 0 --control
  python3 models.py $fs official --fold 0 --nn-seed 466
  python3 models.py $fs official --fold 0 --nn-seed 467
  python3 models.py $fs official --merge
  for split in cv mission; do
    n=5; [ $split = cv ] && [ $fs = history ] && n=25
    for k in $(seq 0 $((n-1))); do python3 models.py $fs $split --fold $k; done
    for k in 0 1 2 3 4; do python3 models.py $fs $split --fold $k --control; done
    python3 models.py $fs $split --merge
  done
done
python3 paper_check.py      # the paper's own experiment, 505 splits (about 40 min single-threaded)
python3 evaluate.py
python3 -m pytest -q tests
