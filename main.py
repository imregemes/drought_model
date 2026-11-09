# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:38:47 2026

Run and validate the Hungary input-output model.
@author: njp
"""
import numpy as np

from cemodel.typing import PathType
from hungarian_input_output_model import EurostatInputOutputModel


INPUT_DATA_FILEPATH: PathType = "data.npz"


if __name__ == "__main__":

    # Load processed historical data
    model = EurostatInputOutputModel.from_npz(INPUT_DATA_FILEPATH)

    # Store expected historical output
    gross_output_expected = (model.get_variable_as_dataframe("q"))

    # Force the model to recalculate output
    model.q = 0

    # Solve q = Lf
    model.solve()

    gross_output_result = (model.get_variable_as_dataframe("q"))

    # Check that the model reproduces historical output
    if not np.allclose(gross_output_result.values, gross_output_expected.values,
        rtol=1e-5,atol=0.1,):
        difference = (gross_output_result.values - gross_output_expected.values)

        raise ValueError(
            "Solved gross output differs from historical output. "
            f"Maximum difference: {np.abs(difference).max():.4f}"
        )

    print("Model successfully reproduces historical output.")