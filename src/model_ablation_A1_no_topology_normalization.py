"""A1: No topology normalization — same model as baseline (CalibratedTopoAR).

The ablation is entirely in preprocessing (removing cu_net /= N_DU),
not in the model. This file exists to satisfy the per-ablation model
file convention and to make imports self-describing.
"""
from model_calibrated import CalibratedTopoAR  # noqa: F401 — re-exported as ablation model

AblationModel = CalibratedTopoAR
