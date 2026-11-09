# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:45:26 2026

Paks drought / electricity-import-substitution scenarios for the Hungary IO
model.

Implements the shock as an alternative technical-coefficient (A) matrix,
 when part of the electricity shortfall is met through
imports rather than domestic generation, the *domestic* input coefficient
for electricity (the "Electricity, gas, steam and air conditioning supply"
row of A) must fall, and the freed-up coefficient mass must be added to the
import ratio (m_ratio) - not simply dropped. Because the source IO table is
"Domestic uses" only, imports already sit outside A as the primary-input
ratio m_ratio, so every industry's column identity

    sum_i A[i, j] + tls_prds_ratio[j] + m_ratio[j] + gva_ratio[j] == 1

is preserved before and after the shift: only the *source* of the input
changes, not the total amount required per unit of output.

@author: njp
"""

from dataclasses import dataclass, field
import numpy as np
import openpyxl
import pandas as pd

from cemodel.typing import PathType
from hungarian_input_output_model import EurostatInputOutputModel


INPUT_DATA_FILEPATH: PathType = "data.npz"
ELECTRICITY_SECTOR = "Electricity, gas, steam and air conditioning supply"
CONSTRUCTION_SECTORS = (
    "Construction of buildings",
    "Civil engineering",
)
PARAMETER_FILE = "Data/EDP Energy impacts and imports.xlsx"

SECTOR_LABEL_MAP = {
    "Chemicals": "Manufacture of chemicals and chemical products",
    "Electrical equipment": "Manufacture of electrical equipment",
    "Land transport": "Land transport and transport via pipelines",
    "Motor vehicles": "Manufacture of motor vehicles, trailers and semi-trailers",
    "Other non-metallic mineral products": "Manufacture of other non-metallic mineral products",
    "Paper products": "Manufacture of paper and paper products",
    "Basic metals": "Manufacture of basic metals",
}

def load_scenario_parameters(scenario: str):
    """Read one scenario's parameters from Excel."""

    df = pd.read_excel(PARAMETER_FILE, sheet_name="Parameters")
    df = df[df["Scenario"] == scenario]
    return dict(zip(df["Parameter"], df["Value"]))

def sector_index(model: EurostatInputOutputModel, label: str) -> int:
    """Look up an industry's position in the model's retained industry list."""
    industries = list(model.CLASSIFICATIONS["Industries"])
    if label not in industries:
        raise KeyError(
            f"{label!r} not found in model's retained industries - it may "
            "have been dropped as an aggregate or zero-output sector during "
            "data processing."
        )
    return industries.index(label)


def calculate_coefficient_share(
    model: EurostatInputOutputModel,
    production_loss_eur: float,
    import_replacement_share: float,
) -> float:
    """
    Convert a monetary import shock into a coefficient shift.
    """
    production_loss = production_loss_eur / 1_000_000
    additional_imports = (production_loss * import_replacement_share)
    elec_idx = sector_index(model, ELECTRICITY_SECTOR)
    print(production_loss_eur)
    print(import_replacement_share)
    intermediate_electricity_sales = np.sum(
        model.A[0, elec_idx, :] * model.q[0, :])
    return additional_imports / intermediate_electricity_sales

# ---------------------------------------------------------------------------
# Channel 1: electricity import substitution
# ---------------------------------------------------------------------------

@dataclass
class ElectricityShock:
    """Parameters for the electricity import-substitution shock.

    import_replacement_share: fraction of every industry's *domestic*
        electricity input coefficient that gets reassigned to imports.
    electricity_sector: label of the electricity row/column in the
        model's industry classification.
    """
    import_replacement_share: float
    electricity_sector: str = ELECTRICITY_SECTOR

def apply_electricity_shock(
    model: EurostatInputOutputModel,
    shock: ElectricityShock,
) -> tuple[np.ndarray, np.ndarray]:
    """Return alternative (A, m_ratio) arrays with the electricity row of A
    shifted into m_ratio, for every time period in the loaded model.

    Both arrays keep their full (time, industry[, industry]) shape so they
    can be assigned straight back with model["A"] = ... / model["m_ratio"] = ...
    """
    elec_idx = sector_index(model, shock.electricity_sector)

    A = model.A.copy()
    m_ratio = model.m_ratio.copy()

    # A has shape (T, n, n); the electricity row is A[:, elec_idx, :]
    shifted = A[:, elec_idx, :] * shock.import_replacement_share
    A[:, elec_idx, :] -= shifted
    m_ratio += shifted

    return A, m_ratio


# ---------------------------------------------------------------------------
# Channel 2: industrial self-curtailment - AGREED METHOD (A matrix)
# ---------------------------------------------------------------------------

@dataclass
class CurtailmentShock:
    """Reduces A[elec, sector] for each curtailing sector. See module
    docstring for the demand/supply-consistency limitation this implies -
    state it wherever these results are shown."""
    sector_reduction_share: dict

    electricity_sector: str = ELECTRICITY_SECTOR


def load_curtailment_shares(model: EurostatInputOutputModel) -> dict:
    """Reads the Sector weightings (Summary!B18:C24) and total central
    reduction (Controls and outputs!B10) from the workbook, and converts
    each sector's curtailed MWh into a fractional cut to that sector's own
    A[elec, sector] coefficient."""
    wb = openpyxl.load_workbook(PARAMETER_FILE, data_only=True)

    total_reduction_mwh = wb["Controls and outputs"]["B10"].value
    price_eur_mwh = wb["Summary"]["C7"].value

    sector_weights = {}
    ws = wb["Summary"]
    for row in range(18, 25):
        sector_name = ws[f"B{row}"].value
        weight = ws[f"C{row}"].value
        if sector_name and weight:
            sector_weights[sector_name] = weight

    elec_idx = sector_index(model, ELECTRICITY_SECTOR)

    reduction_shares = {}
    for short_name, weight in sector_weights.items():
        full_name = SECTOR_LABEL_MAP.get(short_name)
        if full_name is None:
            continue
        idx = sector_index(model, full_name)
        curtailed_mwh = total_reduction_mwh * weight
        curtailed_value_eur = curtailed_mwh * price_eur_mwh
        current_purchase_eur = model.A[0, elec_idx, idx] * model.q[0, idx] * 1_000_000
        if current_purchase_eur <= 0:
            reduction_shares[full_name] = 0.0
            continue
        reduction_shares[full_name] = min(curtailed_value_eur / current_purchase_eur, 1.0)

    return reduction_shares


def apply_curtailment(model: EurostatInputOutputModel, shock: CurtailmentShock) -> np.ndarray:
    elec_idx = sector_index(model, shock.electricity_sector)
    A = model.A.copy()

    for sector, reduction_share in shock.sector_reduction_share.items():
        idx = sector_index(model, sector)
        A[:, elec_idx, idx] *= (1 - reduction_share)

    return A


# ---------------------------------------------------------------------------
# Channel 3: investment / mitigation shock
# ---------------------------------------------------------------------------

@dataclass
class InvestmentShock:
    """A final-demand (gross capital formation) injection for mitigation
    infrastructure, split across one or more sectors.

    allocation: {industry label: HUF million (or model's native unit) of
        additional capital formation} - use the model's own unit (the
        source table was Million euros, so convert HUF figures accordingly
        before use).
    """
    allocation: dict[str, float] = field(default_factory=dict)

def apply_investment_shock(model: EurostatInputOutputModel,shock: InvestmentShock,
                           ) -> np.ndarray:
    """Return an alternative gross-capital-formation array (i) with the
    investment allocation added on top of the baseline."""
    i_array = model.i.copy()  # shape (T, n)
    for label, amount in shock.allocation.items():
        idx = sector_index(model, label)
        i_array[:, idx] += amount
    return i_array

# ---------------------------------------------------------------------------
# Load, solve, compare
# ---------------------------------------------------------------------------

def load_and_solve(
    electricity_shock: ElectricityShock | None = None,
    curtailment_shock: CurtailmentShock | None = None,
    investment_shock: InvestmentShock | None = None,
) -> EurostatInputOutputModel:
    """Load a fresh copy of the baseline data, apply the requested shocks,
    and solve."""
    model = EurostatInputOutputModel.from_npz(INPUT_DATA_FILEPATH)

    if electricity_shock is not None:
        A, m_ratio = apply_electricity_shock(model,electricity_shock,)
        model["A"] = A
        model["m_ratio"] = m_ratio

    if curtailment_shock is not None:
        model["A"] = apply_curtailment(model,curtailment_shock)
    if investment_shock is not None:
        model["i"] = apply_investment_shock(model,investment_shock,)

    # Force recalculation rather than reusing the cached historical output
    model.q = 0
    model.solve()
    return model

def compare(
    baseline: EurostatInputOutputModel,
    scenario: EurostatInputOutputModel,
    label: str,
) -> pd.DataFrame:
    """Industry-level and total change in output and GVA vs baseline."""

    industries = list(baseline.CLASSIFICATIONS["Industries"])
    assert baseline.q[0].shape == (len(industries),), (
        f"Expected q[0] to have one value per industry ({len(industries)}), "
        f"got shape {baseline.q[0].shape} - something upstream is off."
    )
    dq = pd.Series(scenario.q[0] - baseline.q[0], index=industries)
    dgva = pd.Series(scenario.gva[0] - baseline.gva[0], index=industries)

    result = pd.DataFrame({"Output change": dq, "GVA change": dgva})
    result.attrs["scenario"] = label
    result.attrs["total_gva_change"] = dgva.sum()
    return result


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def scenario_A_observed_event() -> tuple[ElectricityShock, CurtailmentShock]:
    params = load_scenario_parameters("A")
    model = EurostatInputOutputModel.from_npz(INPUT_DATA_FILEPATH)
    share = calculate_coefficient_share(
        model, params["ProductionLossEUR"], params["ImportReplacementShare"]
    )
    curtailment = CurtailmentShock(sector_reduction_share=load_curtailment_shares(model))
    return ElectricityShock(import_replacement_share=share), curtailment

def scenario_B_severe_case() -> tuple[ElectricityShock, CurtailmentShock]:
    params = load_scenario_parameters("B")
    model = EurostatInputOutputModel.from_npz(INPUT_DATA_FILEPATH)
    share = calculate_coefficient_share(
        model, params["ProductionLossEUR"], params["ImportReplacementShare"]
    )
    # Curtailment data is event-specific (not re-scaled for severity) -
    # same curtailment shares reused here as a simplifying assumption.
    curtailment = CurtailmentShock(sector_reduction_share=load_curtailment_shares(model))
    return ElectricityShock(import_replacement_share=share), curtailment


def scenario_C_mitigated() -> tuple[ElectricityShock, CurtailmentShock, InvestmentShock]:
    """Reads AvoidanceEffectiveness, InvestmentEUR and (optionally)
    InvestmentSector for scenario "C" from the Parameters sheet.

    Scenario A's shock, scaled down by an assumed risk-reduction
    effectiveness from cooling-security infrastructure, plus the
    infrastructure spend itself as a separate capital-formation injection.

    avoidance_effectiveness: fraction of the Scenario A shock avoided
        (e.g. 0.6 = infrastructure cuts the import-replacement share by 60%).
    investment_allocation: {industry: amount in model units (million EUR)}.
        Placeholder split across Construction of buildings / Civil
        engineering - replace with your own allocation once you've decided
        how to convert the HUF investment figures and timing.
    """
    params = load_scenario_parameters("C")
    model = EurostatInputOutputModel.from_npz(INPUT_DATA_FILEPATH)

    avoidance_effectiveness = params["AvoidanceEffectiveness"]
    residual_production_loss = params["ProductionLossEUR"] * (1 - avoidance_effectiveness)
    residual_share = calculate_coefficient_share(
        model, residual_production_loss, params["ImportReplacementShare"]
    )
    electricity_shock = ElectricityShock(import_replacement_share=residual_share)
    print(electricity_shock)
    curtailment = CurtailmentShock(sector_reduction_share=load_curtailment_shares(model))

    investment_value = params["InvestmentEUR"] / 1_000_000
    investment_sector = params.get("InvestmentSector", "Civil engineering")
    investment_shock = InvestmentShock(allocation={investment_sector: investment_value})

    return electricity_shock, curtailment, investment_shock


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_sectoral_changes(result: pd.DataFrame, label: str, top_n: int = 15) -> str:
    """Horizontal bar chart of GVA change by sector, top N by magnitude
    (both gains and losses). Saves a PNG and returns its path."""
    import matplotlib.pyplot as plt

    top = result["GVA change"].reindex(
        result["GVA change"].abs().sort_values(ascending=False).index
    ).head(top_n).sort_values()

    colors = ["#c0392b" if v < 0 else "#2980b9" for v in top.values]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("GVA change (million EUR)")
    ax.set_title(f"Most affected sectors — {label}")
    fig.tight_layout()

    safe_label = label.split(":")[0].strip().replace(" ", "_")
    path = f"sectoral_changes_{safe_label}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    baseline = load_and_solve()
    results = []


    # Scenario A - observed event
    elec_a, curt_a = scenario_A_observed_event()
    scenario_a = load_and_solve(electricity_shock=elec_a, curtailment_shock=curt_a)
    results.append(compare(baseline, scenario_a, "A: Observed event"))

    # Scenario B - severe case
    elec_b, curt_b = scenario_B_severe_case()
    scenario_b = load_and_solve(electricity_shock=elec_b, curtailment_shock=curt_b)
    results.append(compare(baseline, scenario_b, "B: Severe case"))

    # Scenario C - mitigated (damage-avoided leg only, C1)
    elec_c, curt_c, invest_c = scenario_C_mitigated()
    scenario_c1 = load_and_solve(electricity_shock=elec_c, curtailment_shock=curt_c)
    c1_result = compare(baseline, scenario_c1, "C1: Mitigated - damage avoided")
    results.append(c1_result)

    # Scenario C - investment stimulus leg only (C2), run in isolation so the
    # two channels stay separable
    scenario_c2 = load_and_solve(investment_shock=invest_c)
    c2_result = compare(baseline, scenario_c2, "C2: Mitigated - investment stimulus")
    results.append(c2_result)

    c3_result = c1_result + c2_result
    c3_result.attrs["scenario"] = "C3: Net impact"
    c3_result.attrs["total_gva_change"] = (
        c1_result.attrs["total_gva_change"] + c2_result.attrs["total_gva_change"]
    )
    results.append(c3_result)

    with pd.ExcelWriter("paks_scenario_results.xlsx", engine="openpyxl") as writer:
        for r in results:
            r.to_excel(writer, sheet_name=r.attrs["scenario"].split(":")[0][:31])

    for r in results:
        print(f"\n{r.attrs['scenario']}")
        print(f"Total GVA change: {r.attrs['total_gva_change']:,.2f} million EUR")
        print(r.sort_values("GVA change").tail(5))


    # -----------------------------
    # Visualisations
    # -----------------------------
    for r in results:
        chart_path = plot_sectoral_changes(r, r.attrs["scenario"])
        print(f"Saved: {chart_path}")