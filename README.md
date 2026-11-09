# EDP Hungary Input-Output Model

This repository contains the Python code and input data used to implement the EDP Hungary input-output analysis. The model uses Eurostat input-output data to construct a Hungarian input-output model and apply the EDP scenarios.

## Project structure

The code is organised into four main Python scripts:

### `data_processing.py`

Processes the raw Eurostat input-output data and prepares it for use in the model.

The resulting processed input-output table is stored in the `Data` folder.

### `hungarian_input_output_model.py`

Defines the Hungarian input-output model.

This contains the core model functionality used by the other scripts to construct and solve the input-output model.

### `main.py`

Runs the baseline model and checks that it can reproduce historical results.

This script can therefore be used to verify that the processed data and input-output model are working correctly before implementing the EDP scenarios.

### `scenario implementation.py`

Defines and runs the EDP scenarios.

The script uses the input-output model defined in `hungarian_input_output_model.py` together with the scenario assumptions contained in the scenario input file in the `Data` folder.

## Data

The `Data` folder contains the principal inputs required to run the model, including:

- the processed Hungarian input-output table produced by `data_processing.py`; and
- the scenario input file containing the assumptions used in `scenario implementation.py`.

## Workflow

The overall workflow is:

1. **Process the raw Eurostat data** using `data_processing.py`.
2. **Define the input-output model** using `hungarian_input_output_model.py`.
3. **Run and validate the baseline model** using `main.py`, checking that it can reproduce historical results.
4. **Define and run the EDP scenarios** using `scenario implementation.py`, drawing on the scenario assumptions stored in the `Data` folder.

In simplified form:

Raw Eurostat data
→ `data_processing.py`
→ Processed input-output table
→ `hungarian_input_output_model.py`
→ `main.py` (baseline validation)
→ `scenario implementation.py` (EDP scenarios)

## Running the model

For a new or updated set of Eurostat input-output data, run:

`data_processing.py`

To check that the underlying model is functioning correctly, run:

`main.py`

To produce the EDP scenario results, run:

`scenario implementation.py`

The model and scenario scripts expect the required processed data and scenario inputs to be available in the `Data` folder.