"""A6: Open-loop only — same model as baseline (CalibratedTopoAR).

The ablation is entirely in inference (always using phase_infer() instead of
phase_infer_closed_loop()), not in the model. This file exists to satisfy the
per-ablation model file convention and to make imports self-describing.
"""
from model_calibrated import CalibratedTopoAR  # noqa: F401 — re-exported as ablation model

AblationModel = CalibratedTopoAR
