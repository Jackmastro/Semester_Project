import pandas as pd
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

        self.data = {
            "time":     [],
            "SoC":      [],
            "T_HP_c":   [],
            "T_HP_h":   [],
            "I_HP":     [],
            "x_FAN":    [],
            "T_cell":   [],
            "I_BT":     [],
            "U_BT":     [],
            "U_oc":     [],
            "U_HP":     [],
            "COP":      [],
        }

    def run(self) -> pd.DataFrame:
        x = self.model.get_initial_state

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
            self.data["T_HP_c"].append(x[1])
            self.data["T_HP_h"].append(x[2])
            self.data["I_HP"].append(u[0])
            self.data["x_FAN"].append(u[1])
            self.data["T_cell"].append(values[5])
            self.data["I_BT"].append(values[4])
            self.data["U_BT"].append(values[0])
            self.data["U_oc"].append(values[1])
            self.data["U_HP"].append(values[2])
            self.data["COP"].append(values[3])

        return pd.DataFrame(self.data)