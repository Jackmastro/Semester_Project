<h1 align="center">Model, Identification and Optimal Control of a Multispectral Optogenetic Simulator</h1>
<h3 align="center">Read Me</h3>

## Repository Structure

The repository is organized into two main folders:

### `notebooks/`

- `bounds_simulation.ipynb`: Grid search algorithm to find the operational bounds.
- `main_simulation.ipynb`: Simulates the DIYA system for multiple inputs (ambient temperature, LED light, controller, ...)
- `symbolic_ODE_ss_matrices.ipynb`: Symbolic equations from the modeling.
- `test_communicator.ipynb`: NOT FINISHED: used to run real experiments.

### `src/`

Contains folders used to run the simulations and real experiments, as well as images or outputs from the simulations:

#### `classes/`
- `diya_communicator.py`: NOT FINISHED: used to connect with the DIYA device for real experiments.
- `LED_params.py`: Connects the Program from biologists with power consumption and waste heat.
- `model.py`: Model of the components.
- `mqtt_listener.py`: File written by Samuel Balula to plot live operations.
- `simulation.py`: Simulation class that runs the model given a controller and parameters from the LED.

#### `controllers/`
- `controller_base.py`: Base abstract class for the controllers.
- PID, LQR, MPC, ...

#### `data/`
- `battery/`: Matlab codes to analyze and estimate the battery parameters.
- `fan/`: To analyze and estimate the fan parameters.
- `fan/`: To analyze and estimate the Peltier module parameters.

#### `helper_functions/`
- `phase_portraits/`: Matlab folder downloaded online for plotting.
- `rga/`: Matlab folder downloaded online for Relative Gain Array calculation.

#### `img/`
Contains .pdf plots for the report.
- `__init__.py`: Contains the function that saves the pdf.

#### `measurements/`
Contains .csv file from real experiments (THE PATH NOW IN `diya_communicator.py` IS NOT UPDATED AND SHOULD BE IMPLEMENTED IN `__init__.py`)

#### `output_notebooks/`
Contains .pdf plots for the report.
- `__init__.py`: Contains the functions that save the bounds and ss-matrices.
- `operational_bounds/`: Contains .csv files with values of the parameters of the least-squares of the bounds.
- `state_space/`: Contains .csv files with the state-space matrices.

## Prerequisites

The project requires Python 3.11 or higher. To set up the project, run the following commands:

### macOS/Linux
```bash
git clone git@github.comJackmastro/Semester_Thesis_1.git
cd Semester_Thesis_1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Windows
```bash
[git clone git@github.com:Jackmastro/Semester_Thesis_1.git](https://github.com/Jackmastro/Semester_Project.git)
cd Semester_Thesis_1
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Work in Matlab
Always run `setup.m` to add all path to the working directory. Navigate to the `src/data` folder to find the Matlab scripts used for plotting and fit the data.

## Running Simulations in Python

To run the experiments, navigate to the `notebooks/` folder and execute the desired Jupyter notebook.

## Authors
Giacomo Matroddi
