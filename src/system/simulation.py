import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import convert_temperature as conv_temp

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
        self.data_df:pd.DataFrame = pd.DataFrame()

    def run(self) -> pd.DataFrame:
        x_prev = self.x0

        for t in range(self.time_steps):
            current_time = t * self.dt

            y = self.model.get_output()

            u = self.controller.get_control_input(x_prev, y)

            # Advance system states
            x_next, u_bounded = self.model.discretized_update(u, self.dt)

            # Get internal values
            values = self.model.get_values(x_prev, u_bounded)

            self.data["time"].append(current_time)
            self.data["SoC"].append(x_next[0])
            self.data["T_c"].append(x_next[1])
            self.data["T_h"].append(x_next[2])
            self.data["I_HP"].append(u_bounded[0])
            self.data["x_FAN"].append(u_bounded[1])
            self.data["T_cell"].append(values["T_cell"])
            self.data["I_BT"].append(values["I_BT"])
            self.data["U_BT"].append(values["U_BT"])
            self.data["U_oc"].append(values["U_oc"])
            self.data["U_HP"].append(values["U_HP"])
            self.data["COP"].append(values["COP"])

            x_prev = x_next

        # Remove last element of the states and temperatures to have equal length
        self.data["SoC"].pop()
        self.data["T_c"].pop()
        self.data["T_h"].pop()
        self.data["T_cell"].pop()

        self.data_df = pd.DataFrame(self.data)

        return self.data_df
    
    def plot_time_results(self, results:pd.DataFrame=None) -> None:
        if results is None:
            if self.data_df is None:
                raise ValueError("No data to plot")
            results = self.data_df

        # Conversion
        results[["T_cell", "T_c", "T_h"]] = conv_temp(results[["T_cell", "T_c", "T_h"]].to_numpy(), 'K', 'C')

        xlimits = (0, self.time_span)

        fig, axs = plt.subplots(2, 2, figsize=(15, 3))

        # Temperatures (first row spanning both columns)
        axs[0, 0].remove()  # Remove the first subplot
        axs[0, 1].remove()  # Remove the second subplot
        ax_temp = fig.add_subplot(2, 1, 1)  # Create a new subplot spanning the top row
        ax_temp.axhline(y=0, lw=1, color='k', label='_nolegend_')
        ax_temp.axvline(x=0, lw=1, color='k', label='_nolegend_')
        results.plot(
            x="time",
            y=["T_cell", "T_c", "T_h"],
            xlabel="Time [s]",
            ylabel="Temperature [°C]",
            title="Temperatures",
            ax=ax_temp,
            color=["lightblue", "blue", "red"]
        )
        ax_temp.axhline(y=conv_temp(self.controller.setpoint, 'K', 'C'), color='black', linestyle='--', label='Setpoint')
        ax_temp.axhline(y=conv_temp(self.model.T_amb, 'K', 'C'), color='gray', linestyle='--', label='Ambient')
        ax_temp.legend()
        ax_temp.grid()
        ax_temp.set_xlim(xlimits)

        # SoC and x_FAN (second row, first column)
        axs[1, 0].axhline(y=0, lw=1, color='k', label='_nolegend_')
        axs[1, 0].axvline(x=0, lw=1, color='k', label='_nolegend_')
        results.plot(
            x="time",
            y=["SoC", "x_FAN"],
            xlabel="Time [s]",
            ylabel="Percentage [%]",
            title="SoC and x_FAN",
            ax=axs[1, 0],
            color=["green", "orange"]
        )
        axs[1, 0].grid()
        axs[1, 0].set_xlim(xlimits)

        # Currents and Voltages (second row, second column, dual y-axis)
        ax_curr = axs[1, 1]
        ax_volt = ax_curr.twinx()

        ax_curr.axhline(y=0, lw=1, color='k', label='_nolegend_')
        ax_curr.axvline(x=0, lw=1, color='k', label='_nolegend_')
        results.plot(
            x="time",
            y=["I_HP", "I_BT"],
            xlabel="Time [s]",
            ylabel="Current [A]",
            title="Currents and Voltages",
            ax=ax_curr,
            legend=False,
            color=["orange", "green"]
        )
        results.plot(
            x="time",
            y=["U_oc", "U_BT", "U_HP"],
            xlabel="Time [s]",
            ylabel="Voltage [V]",
            ax=ax_volt,
            legend=False,
            color=["lightgreen", "green", "orange"],
            style="--"
        )
        ax_curr.legend(["I_HP", "I_BT"], loc="lower left")
        ax_volt.legend(["U_oc", "U_BT", "U_HP"], loc="lower right")
        ax_curr.grid()
        ax_curr.set_xlim(xlimits)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_current_temperature(self, results:pd.DataFrame=None) -> None:
        if results is None:
            if self.data_df is None:
                raise ValueError("No data to plot")
            results = self.data_df

        # Conversion
        results[["T_cell", "T_c", "T_h"]] = conv_temp(results[["T_cell", "T_c", "T_h"]].to_numpy(), 'K', 'C')

        x_sim = results["T_h"].to_numpy() - results["T_c"].to_numpy()
        y_sim = results["I_HP"].to_numpy()
        COP_sim = results["COP"].to_numpy()

        # Arrows
        arrow_step = 80

        # Voltage constraints
        x_vec = np.linspace(-100, 100, self.time_steps)
        y_vec_min, y_vec_max = self.model.get_voltage_constraints(x_vec)

        # Plot
        plt.figure(figsize=(8, 3))
        plt.axhline(y=0, lw=1, color='black', label='_nolegend_')
        plt.axvline(x=0, lw=1, color='black', label='_nolegend_')
        plt.plot(x_sim, y_sim, label="Simulation", color='red')
        for i in range(0, len(x_sim) - arrow_step, arrow_step):
            plt.arrow(
                x_sim[i], y_sim[i], 
                x_sim[i + 1] - x_sim[i], y_sim[i + 1] - y_sim[i], 
                head_width=0.15, head_length=1.5, fc='red', ec='red', alpha=0.2
            )

        plt.plot(x_vec, y_vec_min, color='black', linestyle=':', label=r'$U_{max}$')
        plt.plot(x_vec, y_vec_max, color='black', linestyle=':')
        plt.axhline(y=self.model.I_HP_max, color='black', linestyle='--', label=r'$I_{max}$')
        plt.axhline(y=-self.model.I_HP_max, color='black', linestyle='--')
        plt.axvline(x=self.model.DeltaT_max, color='black', linestyle='-.', label=r'$\Delta T_{max}$')
        plt.axvline(x=-self.model.DeltaT_max, color='black', linestyle='-.')

        # Configure plot
        plt.xlim(-self.model.DeltaT_max * 1.1, self.model.DeltaT_max * 1.1)
        plt.ylim(-self.model.I_HP_max * 1.1, self.model.I_HP_max * 1.1)
        plt.xlabel(r'$\Delta T \; [°]$')
        plt.ylabel(r'$I_\mathrm{HP} \; [A]$')
        plt.title('Current-Temperature Phase Space')
        plt.legend()
        plt.grid()
        plt.show()