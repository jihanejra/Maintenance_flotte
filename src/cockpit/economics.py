"""Coûts unitaires en euros dérivés de `config/costs.yaml` (hypothèses documentées)."""

from __future__ import annotations

from dataclasses import dataclass

from cockpit.config import load_config


@dataclass(frozen=True)
class UnitCosts:
    breakdown: float      # panne : pièce, urgence, remorquage, pénalités client, immobilisation
    replacement: float    # remplacement planifié : pièce + main d'œuvre + immobilisation courte
    inspection: float     # contrôle sans remplacement (fausse alerte) + immobilisation courte
    downtime_day: float

    @property
    def threshold(self) -> float:
        """Probabilité de panne au-delà de laquelle une intervention est rentable.

        Intervenir coûte l'inspection, plus le remplacement si la panne était imminente ;
        ne rien faire coûte la panne avec probabilité q. Intervenir est rentable si
        q · breakdown > inspection + q · (replacement - inspection), soit
        q > inspection / (breakdown - replacement + inspection).
        """
        return self.inspection / (self.breakdown - self.replacement + self.inspection)


def unit_costs(overrides: dict | None = None) -> UnitCosts:
    e = {**load_config("costs")["economics"], **(overrides or {})}
    day = e["downtime_cost_per_day"]
    return UnitCosts(
        breakdown=e["part_cost"] + e["breakdown_labour"] + e["towing_cost"]
        + e["customer_penalty"] + e["downtime_days_breakdown"] * day,
        replacement=e["part_cost"] + e["preventive_labour"] + e["downtime_days_replacement"] * day,
        inspection=e["inspection_cost"] + e["downtime_days_inspection"] * day,
        downtime_day=day,
    )
