"""Read active stress state from the last row of the anomaly CSV."""
import logging
from dataclasses import dataclass

import pandas as pd

CU_CONTAINERS = {"srscu0", "srscu1", "srscu2"}
DU_CONTAINERS = {"srsdu0", "srsdu1", "srsdu2", "srsdu3", "srsdu4", "srsdu5"}
ALL_CONTAINERS = sorted(CU_CONTAINERS | DU_CONTAINERS)

STRESS_NAMES = {0: "NONE", 1: "CPU", 2: "MEM", 3: "NET"}

# DUs served by each CU (topology: 3 CUs × 2 DUs each)
CU_TO_DUS = {
    "srscu0": ["srsdu0", "srsdu1"],
    "srscu1": ["srsdu2", "srsdu3"],
    "srscu2": ["srsdu4", "srsdu5"],
}
DU_TO_CU = {du: cu for cu, dus in CU_TO_DUS.items() for du in dus}

# DU → PCI mapping (one PCI per DU, one UE per cell)
DU_TO_PCIS = {
    "srsdu0": [1], "srsdu1": [2], "srsdu2": [3],
    "srsdu3": [4], "srsdu4": [5], "srsdu5": [6],
}
CU_TO_PCIS = {
    "srscu0": [1, 2], "srscu1": [3, 4], "srscu2": [5, 6],
}


@dataclass
class ActiveStress:
    container:   str
    stress_type: int
    intensity:   float
    stress_name: str
    node_type:   str   # "CU" or "DU"
    affected_pcis: list[int]


def _pcis_for(container: str) -> list[int]:
    if container in CU_TO_PCIS:
        return CU_TO_PCIS[container]
    return DU_TO_PCIS.get(container, [])


class StressTracker:
    def get_active(self, anomaly_csv_path: str) -> list[ActiveStress]:
        try:
            df = pd.read_csv(anomaly_csv_path, low_memory=False)
            if df.empty:
                return []
            last = df.iloc[-1]
        except FileNotFoundError:
            return []
        except Exception as e:
            logging.debug(f"[StressTracker] Could not read {anomaly_csv_path}: {e}")
            return []

        active: list[ActiveStress] = []
        for c in ALL_CONTAINERS:
            s_type_col = f"{c}_stressType"
            s_int_col  = f"{c}_stepStress"
            if s_type_col not in last.index:
                continue
            try:
                s_type    = int(last[s_type_col])
                intensity = float(last.get(s_int_col, 0))
            except (ValueError, TypeError):
                continue
            if s_type == 0:
                continue
            active.append(ActiveStress(
                container=c,
                stress_type=s_type,
                intensity=intensity,
                stress_name=STRESS_NAMES.get(s_type, "UNKNOWN"),
                node_type="CU" if c in CU_CONTAINERS else "DU",
                affected_pcis=_pcis_for(c),
            ))
        return active
