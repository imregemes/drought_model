# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:59:10 2026
Data processing script to reas input-output table for Hungary downloaded 
from Eurostat.

Processing of raw Symmetric input-output table at basic prices (industry by industry) 
downloaded from EUROSTAT dataset name: [naio_10_cp1750__custom_21969973]
link: https://ec.europa.eu/eurostat/databrowser/view/naio_10_cp1750__custom_22728660/default/table

selected country: Hungary
selected industries, adjustments and value added: all
selected industries and final uses: all
selected stock or flow: Domestic uses
selected time: 2023
selected unit of measure: Million euros

@author: njp
"""
import warnings

import pandas as pd
import numpy as np

from cemodel.typing import PathType
from hungarian_input_output_model import EurostatInputOutputModel

##############################################################################
# Settings
##############################################################################
COUNTRY = "Hungary"
YEAR = 2023

TOLERANCE = 0.1

RAW_DATA_FILEPATH: PathType = r'Data\Raw data\naio_10_cp1750__custom_22728660_spreadsheet.xlsx'
RAW_DATA_SHEETNAME: str = 'Sheet 1'
OUTPUT_DATA_FILEPATH: PathType = 'data.npz'
CHECKS_FILEPATH: PathType = r"Data\IO_checks.xlsx"


# This dictionary maps the column titles in the raw data (the keys) to the
# codes to be used in the model (the values)
FINAL_DEMAND_COMPONENTS: dict[str, str] = {
    'Final consumption expenditure by households': 'c',
    'Final consumption expenditure by government': 'g',
    'Gross capital formation': 'i',
    'Exports of goods and services': 'x',
    'Final consumption expenditure by non-profit organisations serving households (NPISH)': 'n_pish',
}

# This dictionary maps the row titles in the raw data (the keys) to the codes 
#to be used in the model (the values)
PRIMARY_INPUT_COMPONENTS: dict[str, str] = {
    'Use of imported products': 'm',
    'Taxes less subsidies on products': 'tls_prds',
    'Other taxes less other subsidies on production': 'tls_prdn',
    'Compensation of employees': 'coe',
    'Operating surplus and mixed income, gross': 'gos',
    'Value added, gross': 'gva',
    'Output': 'q',  
}


# Combined rows which overlap with more detailed industries
AGGREGATE_SECTORS = {
    "Total",
    "Mining and quarrying",
    "Manufacture of food products; beverages and tobacco products",
    "Manufacture of textiles, wearing apparel, leather and related products",
    "Manufacture of furniture; other manufacturing",
    "Sewerage, waste management, remediation activities",
    "Construction",
    "Accommodation and food service activities",
    "Motion picture, video, television programme production; programming and broadcasting activities",
    "Computer programming, consultancy, and information service activities",
    "Legal and accounting activities; activities of head offices; management consultancy activities",
    "Other professional, scientific and technical activities; veterinary activities",
    "Security and investigation, service and landscape, office administrative and support activities",
    "Residential care activities and social work activities without accommodation",
    "Creative, arts and entertainment activities; libraries, archives, museums and other cultural activities; gambling and betting activities",
    "Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use",
}


# ------------------------------------------------------------------
# Read and clean Eurostat table
# ------------------------------------------------------------------
def read_eurostat_io(filepath: PathType) -> pd.DataFrame:
    # Find the relevant header/data rows
    raw = pd.read_excel(filepath, sheet_name="Sheet 1", header=None, dtype=str)
    column_header = raw.index[raw.iloc[:, 0].eq("IND_USE (Labels)")][0]
    row_header = raw.index[raw.iloc[:, 0].eq("GEO (Labels)")][0]
    # Keep first two identifying columns + only actual data columns
    # Eurostat has blank spacer columns, so we drop columns where the column label is empty
    data_columns = raw.iloc[column_header, 2:].dropna().index.tolist()
    data = raw.loc[row_header + 1:, [0, 1] + data_columns].copy()
    data.columns = (["GEO", "IND_AVA"] + raw.loc[column_header, data_columns].tolist())
    data = data.loc[data["GEO"].eq(COUNTRY)].copy()
    data = data.drop(columns="GEO").set_index("IND_AVA")

    return (data.replace(":", np.nan)
            .replace(",", "", regex=True)
            .apply(pd.to_numeric, errors="coerce"))

# ------------------------------------------------------------------
# Extract one-year model data
# ------------------------------------------------------------------
def extract_data_single_year(
        table: pd.DataFrame,) -> tuple[dict[str, np.ndarray], tuple[str, ...], list[str]]:

    sector_end = table.index.get_loc(
        "Activities of extraterritorial organisations and bodies")

    candidate_sectors = table.index[:sector_end + 1].tolist()

    sectors = [sector for sector in candidate_sectors
               if sector not in AGGREGATE_SECTORS]

    output_all = table.loc["Output", sectors]
    zero_output_sectors = output_all[output_all.fillna(0).eq(0)
                                     ].index.tolist()

    sectors = [sector for sector in sectors
               if sector not in zero_output_sectors]

    # Ensure identical industry rows and columns
    missing_columns = sorted(set(sectors) - set(table.columns))

    if missing_columns:
        raise ValueError(
            f"Industry columns missing for: {missing_columns}")

    extracted = {
        "Z": table.loc[sectors, sectors].to_numpy(dtype=float)}

    for title, model_code in FINAL_DEMAND_COMPONENTS.items():
        extracted[model_code] = (
            table.loc[sectors, title].to_numpy(dtype=float))

    for title, model_code in PRIMARY_INPUT_COMPONENTS.items():
        extracted[model_code] = (
            table.loc[title, sectors].to_numpy(dtype=float))

    return extracted, tuple(sectors), zero_output_sectors
    

# ------------------------------------------------------------------
# Checks
# ------------------------------------------------------------------
    

def create_checks(
    table: pd.DataFrame,
    data: dict[str, np.ndarray],
    sectors: tuple[str, ...],
    zero_output_sectors: list[str],
) -> dict[str, pd.DataFrame]:

    Z = data["Z"]
    q = data["q"]
    gva = data["gva"]
    coe = data["coe"]
    gos = data["gos"]
    tls_prdn = data["tls_prdn"]
    tls_prds = data["tls_prds"]
    m = data["m"]

    intermediate_reported = table.loc[
        "Total intermediate consumption /final use", list(sectors)
    ].to_numpy(dtype=float)

    final_use_reported = table.loc[
        list(sectors), "Total final use"
    ].to_numpy(dtype=float)

    # Column-side identities
    column_checks = pd.DataFrame(index=sectors)

    column_checks["Output"] = q
    column_checks["Z column sum"] = Z.sum(axis=0)
    column_checks["Reported intermediate consumption"] = intermediate_reported
  
    column_checks["Z vs reported IC discrepancy"] = (
        Z.sum(axis=0) + tls_prds + m - intermediate_reported)

    column_checks["Output identity discrepancy"] = (
        q - intermediate_reported - gva)

    column_checks["GVA identity discrepancy"] = (
        gva - coe - gos - tls_prdn)

    column_checks["Column check passed"] = (
        column_checks[            [
                "Z vs reported IC discrepancy",
                "Output identity discrepancy",
                "GVA identity discrepancy",]].abs().max(axis=1) <= TOLERANCE )

    # Row-side identity: output = intermediate sales + total final use
    row_checks = pd.DataFrame(index=sectors)

    row_checks["Output"] = q
    row_checks["Z row sum"] = Z.sum(axis=1)
    row_checks["Reported total final use"] = final_use_reported
    row_checks["Output identity discrepancy"] = (
        q - Z.sum(axis=1) - final_use_reported
    )
    row_checks["Row check passed"] = (
        row_checks["Output identity discrepancy"].abs() <= TOLERANCE
    )

    # Final-demand component identities
    fd_checks = pd.DataFrame(index=sectors)

    fd_checks["Final consumption discrepancy"] = (
        table.loc[list(sectors), "Final consumption expenditure"]
        - table.loc[
            list(sectors),
            ["Final consumption expenditure by government",
                "Final consumption expenditure by households",
                (
                    "Final consumption expenditure by non-profit "
                    "organisations serving households (NPISH)"
                ),
            ],
        ].sum(axis=1)
    )

    fd_checks["Capital formation discrepancy"] = (
        table.loc[list(sectors), "Gross capital formation"]
        - table.loc[
            list(sectors),
            [
                "Gross fixed capital formation",
                "Changes in inventories",
                "Acquisitions less disposals of valuables",
            ],
        ].sum(axis=1)
    )

    fd_checks["Exports geography discrepancy"] = (
        table.loc[list(sectors), "Exports of goods and services"]
        - table.loc[
            list(sectors),
            [
                "Exports of goods and services to EU countries",
                "Exports of goods and services to non-EU countries",
            ],
        ].sum(axis=1)
    )

    fd_checks["Final-demand checks passed"] = (fd_checks.abs().max(axis=1) <= TOLERANCE)

    summary = pd.DataFrame({
        "Metric": [
            "Industries retained",
            "Aggregate industries dropped",
            "Zero-output industries dropped",
            "Missing values in model inputs",
            "Negative values in Z",
            "Maximum column discrepancy",
            "Maximum row discrepancy",
            "Maximum final-demand discrepancy",
            "All checks passed",
        ],
        "Value": [
            len(sectors),
            len(AGGREGATE_SECTORS),
            len(zero_output_sectors),
            sum(np.isnan(value).sum() for value in data.values()),
            int((Z < 0).sum()),
            column_checks.iloc[:, 3:6].abs().max().max(),
            row_checks["Output identity discrepancy"].abs().max(),
            fd_checks.iloc[:, :3].abs().max().max(),
            (
                column_checks["Column check passed"].all()
                and row_checks["Row check passed"].all()
                and fd_checks["Final-demand checks passed"].all()
            ),
        ],
    })

    sector_summary = pd.DataFrame(index=sectors)
    sector_summary["Output"] = q
    sector_summary["Output share"] = q / q.sum()
    sector_summary["GVA share"] = gva / q
    sector_summary["Compensation share"] = coe / q
    sector_summary["Imported-use share"] = data["m"] / q

    return {
        "Summary": summary,
        "Column checks": column_checks,
        "Row checks": row_checks,
        "Final demand checks": fd_checks,
        "Sector indicators": sector_summary,
        "Dropped zero output": pd.DataFrame(
            {"Sector": zero_output_sectors}
        ),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    table = read_eurostat_io(RAW_DATA_FILEPATH)

    data, industry_labels, zero_output_sectors = (
        extract_data_single_year(table)
    )

    n_industries = len(industry_labels)

    # Structural checks
    if data["Z"].shape != (n_industries, n_industries):
        raise ValueError(f"Z is not square: {data['Z'].shape}")

    for name, array in data.items():
        if name != "Z" and array.shape != (n_industries,):
            raise ValueError(f"Unexpected shape for {name}: {array.shape}")

    checks = create_checks(
        table, data, industry_labels, zero_output_sectors
    )

    with pd.ExcelWriter(CHECKS_FILEPATH, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Clean IO table")

        for sheet_name, check_table in checks.items():
            check_table.to_excel(writer, sheet_name=sheet_name[:31])

    # cemodel expects a time dimension, even though there is only one year
    gross_output = data["q"]

    data["A"] = data.pop("Z") / gross_output[np.newaxis, :]

    for model_code in PRIMARY_INPUT_COMPONENTS.values():
        if model_code == "q":
            continue

        data[f"{model_code}_ratio"] = (
            data.pop(model_code) / gross_output
        )

    # Convert:
    # vectors: (industry,) -> (time, industry)
    # matrix:  (industry, industry) -> (time, industry, industry)
    data = {
        name: array[np.newaxis, ...]
        for name, array in data.items()
    }

    model = EurostatInputOutputModel(range(YEAR, YEAR + 1))

    for name, array in data.items():
        if name in model:
            model[name] = array
        else:
            warnings.warn(f"'{name}' not defined in model; skipping")

    model.to_npz(OUTPUT_DATA_FILEPATH)

    print(f"Industries retained: {n_industries}")
    print(f"Checks exported to: {CHECKS_FILEPATH}")
    print(f"Model data exported to: {OUTPUT_DATA_FILEPATH}")    
      
    
    