# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:34:12 2026

cemodel implementation of a static input-output model for Hungary.

@author: njp
"""

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from cemodel import IterativeSolutionModel
from cemodel.containers import VariableDefinition
from cemodel.extensions import NumPyInputOutputMixin, PandasOutputMixin


class EurostatInputOutputModel(
    PandasOutputMixin,
    NumPyInputOutputMixin,
    IterativeSolutionModel,
):
    """Static industry-by-industry IO model for Hungary."""

    CLASSIFICATIONS: ClassVar[dict[str, Sequence[str]]] = {
        "Industries": (
            "Crop and animal production, hunting and related service activities",
            "Forestry and logging",
            "Fishing and aquaculture",
            "Mining of coal and lignite",
            "Extraction of crude petroleum and natural gas",
            "Mining of metal ores",
            "Other mining and quarrying",
            "Mining support servic activities",
            "Manufacture of food products",
            "Manufacture of beverages",
            "Manufacture of tobacco products",
            "Manufacture of textiles",
            "Manufacture of wearing apparel",
            "Manufacutre of leather and related products",
            "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials",
            "Manufacture of paper and paper products",
            "Printing and reproduction of recorded media",
            "Manufacture of coke and refined petroleum products",
            "Manufacture of chemicals and chemical products",
            "Manufacture of basic pharmaceutical products and pharmaceutical preparations",
            "Manufacture of rubber and plastic products",
            "Manufacture of other non-metallic mineral products",
            "Manufacture of basic metals",
            "Manufacture of fabricated metal products, except machinery and equipment",
            "Manufacture of computer, electronic and optical products",
            "Manufacture of electrical equipment",
            "Manufacture of machinery and equipment n.e.c.",
            "Manufacture of motor vehicles, trailers and semi-trailers",
            "Manufacture of other transport equipment",
            "Manufacture of furniture",
            "Other manufacturing",
            "Repair and installation of machinery and equipment",
            "Electricity, gas, steam and air conditioning supply",
            "Water collection, treatment and supply",
            "Sewerage",
            "Waste collection, treatment and disposal activities; materials recovery",
            "Remediation activities and other waste management services",
            "Construction of buildings",
            "Civil engineering",
            "Specialised construction activities",
            "Wholesale and retail trade and repair of motor vehicles and motorcycles",
            "Wholesale trade, except of motor vehicles and motorcycles",
            "Retail trade, except of motor vehicles and motorcycles",
            "Land transport and transport via pipelines",
            "Water transport",
            "Air transport",
            "Warehousing and support activities for transportation",
            "Postal and courier activities",
            "Accommodation",
            "Food and beverage service activities",
            "Publishing activities",
            "Motion picture, video, television programme production, sound recording and music publishing activities",
            "Programming and broadcasting activities",
            "Telecommunications",
            "Computer programming, consultancy and related activities",
            "Information service activities",
            "Financial service activities, except insurance and pension funding",
            "Insurance, reinsurance and pension funding, except compulsory social security",
            "Activities auxiliary to financial services and insurance activities",
            "Imputed rents of owner-occupied dwellings",
            "Real estate activities excluding imputed rents",
            "Legal and accounting activities",
            "Activities of head offices; management consultancy activities",
            "Architectural and engineering activities; technical testing and analysis",
            "Scientific research and development",
            "Advertising and market research",
            "Other professional, scientific and technical activities",
            "Veterinary activities",
            "Rental and leasing activities",
            "Employment activities",
            "Travel agency, tour operator reservation service and related activities",
            "Security and investigation activities",
            "Services to buildings and landscape activities",
            "Office administrative, office support and other business support activities",
            "Public administration and defence; compulsory social security",
            "Education",
            "Human health activities",
            "Residential care activities",
            "Social work activities without accommodation",
            "Creative, arts and entertainment activities",
            "Libraries, archives, museums and other cultural activities",
            "Gambling and betting activities",
            "Sports activities and amusement and recreation activities",
            "Activities of membership organisations",
            "Repair of computers and personal and household goods",
            "Other personal service activities",
            "Activities of households as employers of domestic personnel",
        )
    }



    VARIABLES: ClassVar[dict[str, VariableDefinition]] = {
        # Core IO variables
        "Z": (("Industries", "Industries"), float, 0.0),
        "A": (("Industries", "Industries"), float, 0.0),
        "L": (("Industries", "Industries"), float, 0.0),

        # Output and final demand
        "q": ("Industries", float, 0.0),
        "f": ("Industries", float, 0.0),
        "c": ("Industries", float, 0.0),
        "g": ("Industries", float, 0.0),
        "i": ("Industries", float, 0.0),
        "x": ("Industries", float, 0.0),
        "n_pish": ("Industries", float, 0.0),

        # Primary-input coefficients
        "m_ratio": ("Industries", float, 0.0),
        "tls_prds_ratio": ("Industries", float, 0.0),
        "tls_prdn_ratio": ("Industries", float, 0.0),
        "coe_ratio": ("Industries", float, 0.0),
        "gos_ratio": ("Industries", float, 0.0),
        "gva_ratio": ("Industries", float, 0.0),

        # Calculated impacts
        "gva": ("Industries", float, 0.0),
    }

    CHECK: ClassVar[Sequence[str]] = ["q"]

    def solve_t_before(self, t: int, iteration: int, **kwargs: Any,) -> None:
        """Calculate the Leontief inverse."""
        identity = np.eye(self.A[t].shape[0])
        self.L[t] = np.linalg.inv(identity - self.A[t])

    def evaluate_t(self, t: int, iteration: int, **kwargs: Any, ) -> None:
        """Calculate total final demand, output and GVA."""
        self.f[t] = (
            self.c[t]
            + self.g[t]
            + self.i[t]
            + self.x[t]
            + self.n_pish[t]
        )

        self.q[t] = self.L[t] @ self.f[t]
        self.gva[t] = self.gva_ratio[t] * self.q[t]

    def solve_t_after(self, t: int, iteration: int, **kwargs: Any,) -> None:
        """Recover the intermediate transaction matrix."""
        self.Z[t] = self.A[t] * self.q[t].reshape((1, -1))