import pandas as pd
from IPython.display import display

from classes import Model
from controllers import ControllerBase


class Simulation:
    def __init__(self, model:Model, controller:ControllerBase, dt:float, time_span:float) -> None:
        self.model = model
        self.controller = controller

        assert dt > 0, "dt must be greater than 0"
        assert time_span > 0, "time_span must be greater than 0"
        assert time_span >= dt, "time_span must be greater than or equal to dt"

        self.dt = dt
        self.time_span = time_span
        self.time_steps = int(self.time_span / self.dt)

        self.x0 = self.model.get_initial_state

        self.data = {
            "time":     [],
            "SoC":      [self.x0[0]],
            "T_c":      [self.x0[1]],
            "T_h":      [self.x0[2]],
            "I_HP":     [],
            "x_FAN":    [],
            "T_cell":   [self.x0[1]], # Assuming equal to Tc
            "I_BT":     [],
            "U_BT":     [],
            "U_oc":     [],
            "U_HP":     [],
            "COP":      [],
        }

    def run(self) -> pd.DataFrame:
        x = self.x0

        for t in range(self.time_steps):
            current_time = t * self.dt

            y = self.model.get_output()

            u = self.controller.get_control_input(x, y)

            # Advance system states
            x = self.model.discretized_update(u, self.dt)

            # Get internal values
            values = self.model.get_values(x, u)

            self.data["time"].append(current_time)
            self.data["SoC"].append(x[0])
            self.data["T_c"].append(x[1])
            self.data["T_h"].append(x[2])
            self.data["I_HP"].append(u[0])
            self.data["x_FAN"].append(u[1])
            self.data["T_cell"].append(values["T_cell"])
            self.data["I_BT"].append(values["I_BT"])
            self.data["U_BT"].append(values["U_BT"])
            self.data["U_oc"].append(values["U_oc"])
            self.data["U_HP"].append(values["U_HP"])
            self.data["COP"].append(values["COP"])

        # Remove last element of the states and temperatures
        self.data["SoC"].pop()
        self.data["T_c"].pop()
        self.data["T_h"].pop()
        self.data["T_cell"].pop()

        return pd.DataFrame(self.data)