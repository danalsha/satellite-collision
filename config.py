"""Paths and constants for the satellite collision project.

Override the data folder with the environment variable CAC_DATA
(default ./data, which is where get_data.py unpacks the Zenodo zip).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CAC_DATA", os.path.join(HERE, "data"))
PROC = os.path.join(DATA, "processed")
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
for d in (PROC, OUT, FIG):
    os.makedirs(d, exist_ok=True)

SEED = 465

# ESA's definitions (kelvins.esa.int/collision-avoidance-challenge, Uriot et al. 2022):
#   - risk is log10 of the collision probability
#   - "high risk" means final risk >= -6 (the notification threshold used in the competition)
#   - an avoidance manoeuvre is planned at least 2 days before closest approach, so only
#     CDMs with time_to_tca >= 2 may be used as inputs
#   - the final CDM must be within 1 day of closest approach for its risk to count as "final"
HIGH_RISK = -6.0
CUTOFF_DAYS = 2.0
FINAL_WITHIN_DAYS = 1.0

# Two columns identify the event and the satellite, not the physics of the approach.
ID_COLS = ["event_id", "mission_id"]

# The label is computed from the LAST CDM of each event. Nothing from that CDM (or any
# CDM with time_to_tca < 2) may appear in the features: build_table.py builds every
# feature from the CDMs received at least CUTOFF_DAYS before closest approach only.
LABEL_SOURCE = ["risk"]

CATEGORICAL = ["c_object_type"]
