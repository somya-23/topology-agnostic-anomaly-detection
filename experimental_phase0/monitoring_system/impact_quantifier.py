"""Direction-aware Z-score impact quantification.

Key empirical findings baked in:
  - DU NET stress  → ul_brate +217%, bsr +335% (PHY queuing, F1-U drops after DU counts bits)
  - CU NET/CPU     → ul_brate  -24%  (scheduler starved via CU→DU signalling disruption)
  - MEM stress     → radio KPIs near zero (p=0.46, capped at INFO)
  - CU stress      → infra eth0_tx_bytes drop is primary signal (-60 to -90%)
"""
from dataclasses import dataclass, field

from baseline_stats import BaselineStats
from stress_tracker import ActiveStress

SEVERITY_THRESHOLDS = [
    (3.0, "CRITICAL"),
    (2.0, "WARNING"),
    (1.0, "INFO"),
    (0.0, "NORMAL"),
]

# (node_type, stress_name, kpi_substring) → expected Z-score sign (+1 / -1 / 0)
_DIRECTION: dict[tuple[str, str, str], int] = {
    # Radio — DU stress raises ul_brate (DU PHY queuing, retransmissions counted)
    ("DU", "NET", "ul_brate"):        +1,
    ("DU", "NET", "bsr"):             +1,
    ("DU", "NET", "harq_error_rate"): +1,
    ("DU", "CPU", "ul_brate"):        +1,
    ("DU", "CPU", "bsr"):             +1,
    # Radio — CU stress lowers ul_brate (PUSCH grant suppression)
    ("CU", "NET", "ul_brate"):        -1,
    ("CU", "NET", "bsr"):             +1,   # buffer fills slightly
    ("CU", "CPU", "ul_brate"):        -1,
    # Infra — CU NET stress drops CU egress tx (tc netem on eth0)
    ("CU", "NET", "eth0_tx_bytes"):   -1,
    # Infra — CPU stress raises cpu_pct for that container
    ("CU", "CPU", "cpu_pct"):         +1,
    ("DU", "CPU", "cpu_pct"):         +1,
    # Infra — MEM stress raises memory_bytes
    ("CU", "MEM", "memory_bytes"):    +1,
    ("DU", "MEM", "memory_bytes"):    +1,
}


@dataclass
class KPIImpact:
    kpi:           str
    current:       float
    baseline_mean: float
    z_score:       float
    pct_deviation: float
    severity:      str
    expected:      bool


@dataclass
class ImpactResult:
    container:   str
    stress_name: str
    intensity:   float
    node_type:   str
    affected_pcis: list[int]
    kpi_impacts: dict[str, KPIImpact] = field(default_factory=dict)


def _severity(abs_z: float, stress_name: str, kpi: str) -> str:
    # MEM stress has no significant radio impact — cap at INFO
    if stress_name == "MEM" and any(
        x in kpi for x in ("ul_brate", "bsr", "harq", "snr", "cqi", "dl_brate")
    ):
        return "INFO" if abs_z >= 1.0 else "NORMAL"
    for threshold, label in SEVERITY_THRESHOLDS:
        if abs_z >= threshold:
            return label
    return "NORMAL"


def _expected_direction(node_type: str, stress_name: str, kpi: str) -> int:
    for (nt, sn, substr), sign in _DIRECTION.items():
        if nt == node_type and sn == stress_name and substr in kpi:
            return sign
    return 0  # direction unknown


class ImpactQuantifier:
    def quantify(
        self,
        live_metrics: dict[str, float],
        active_stresses: list[ActiveStress],
        baseline: BaselineStats,
    ) -> list[ImpactResult]:
        results: list[ImpactResult] = []
        for stress in active_stresses:
            impacts: dict[str, KPIImpact] = {}
            for kpi, value in live_metrics.items():
                if not baseline.has(kpi):
                    continue
                z    = baseline.z_score(kpi, value)
                pct  = baseline.pct_deviation(kpi, value)
                sev  = _severity(abs(z), stress.stress_name, kpi)
                exp_dir  = _expected_direction(stress.node_type, stress.stress_name, kpi)
                expected = (exp_dir * z > 0) if exp_dir != 0 else True
                impacts[kpi] = KPIImpact(kpi, value, baseline.stats[kpi]["mean"],
                                          z, pct, sev, expected)
            results.append(ImpactResult(
                container=stress.container,
                stress_name=stress.stress_name,
                intensity=stress.intensity,
                node_type=stress.node_type,
                affected_pcis=stress.affected_pcis,
                kpi_impacts=impacts,
            ))
        return results
