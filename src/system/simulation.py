import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
    
    def plot_time_results(self, title: str = "", results: pd.DataFrame = None) -> None:
        if results is None:
            if self.data_df is None:
                raise ValueError("No data to plot")
            results = self.data_df

        # Conversion
        results[["T_cell", "T_c", "T_h"]] = conv_temp(results[["T_cell", "T_c", "T_h"]].to_numpy(), 'K', 'C')

        xlimits = (0, self.time_span)

        fig, axs = plt.subplots(2, 2, figsize=(15, 4))

        # Temperatures (first row spanning both columns)
        axs[0, 0].remove()  # Remove the first subplot
        axs[0, 1].remove()  # Remove the second subplot
        ax_temp = fig.add_subplot(2, 1, 1)  # Create a new subplot spanning the top row
        ax_temp.axhline(y=0, lw=1, color="black", label='_nolegend_')
        ax_temp.axvline(x=0, lw=1, color="black", label='_nolegend_')
        results.plot(
            x="time",
            y=["T_cell", "T_c", "T_h"],
            xlabel="Time [s]",
            ylabel="Temperature [°C]",
            title="Temperatures " + title,
            ax=ax_temp,
            color=["lightblue", "blue", "red"]
        )
        ax_temp.axhline(y=conv_temp(self.controller.setpoint, 'K', 'C'), color='black', linestyle='--', label='Setpoint')
        ax_temp.axhline(y=conv_temp(self.model.T_amb, 'K', 'C'), color='gray', linestyle='-.', label='Ambient')
        ax_temp.legend(loc="center right")
        ax_temp.grid()
        ax_temp.set_xlim(xlimits)

        # Add minor ticks
        ax_temp.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_temp.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_temp.tick_params(axis='x', which='minor', direction='in', top=True)
        ax_temp.tick_params(axis='y', which='minor', direction='in', left=True)
        ax_temp.tick_params(axis='x', which='major', top=True)
        ax_temp.tick_params(axis='y', which='major', left=True)

        # SoC and x_FAN (second row, first column)
        axs[1, 0].axhline(y=0, lw=1, color="black", label='_nolegend_')
        axs[1, 0].axvline(x=0, lw=1, color="black", label='_nolegend_')
        results.plot(
            x="time",
            y=["SoC", "x_FAN"],
            xlabel="Time [s]",
            ylabel="Percentage [%]",
            title="SoC and FAN duty cycle",
            ax=axs[1, 0],
            color=["green", "purple"]
        )
        axs[1, 0].grid()
        axs[1, 0].set_xlim(xlimits)
        axs[1, 0].set_ylim(-0.1, 1.1)

        # Add minor ticks
        axs[1, 0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1, 0].tick_params(axis='x', which='minor', direction='in', top=True)
        axs[1, 0].tick_params(axis='y', which='minor', direction='in', left=True)
        axs[1, 0].tick_params(axis='x', which='major', top=True)
        axs[1, 0].tick_params(axis='y', which='major', left=True)

        # Currents and Voltages (second row, second column, dual y-axis)
        ax_curr = axs[1, 1]
        ax_volt = ax_curr.twinx()

        ax_curr.axhline(y=0, lw=1, color="black", label='_nolegend_')
        ax_curr.axvline(x=0, lw=1, color="black", label='_nolegend_')
        results.plot(
            x="time",
            y=["I_BT", "I_HP"],
            xlabel="Time [s]",
            ylabel="Current [A]",
            title="Currents and Voltages",
            ax=ax_curr,
            legend=False,
            color=["green", "orange"]
        )
        results.plot(
            x="time",
            y=["U_BT", "U_HP"],
            xlabel="Time [s]",
            ylabel="Voltage [V]",
            ax=ax_volt,
            legend=False,
            color=["green", "orange"],
            style="--"
        )
        ax_curr.legend(["I_BT", "I_HP"], loc="center left")
        ax_volt.legend(["U_BT", "U_HP"], loc="center right")
        ax_curr.grid()
        ax_curr.set_xlim(xlimits)

        # Add minor ticks to both y-axes
        ax_curr.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_curr.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_volt.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_curr.tick_params(axis='x', which='minor', direction='in', top=True)
        ax_curr.tick_params(axis='y', which='minor', direction='in', left=True)
        ax_volt.tick_params(axis='y', which='minor', direction='in', right=True)
        ax_curr.tick_params(axis='x', which='major', top=True)
        ax_curr.tick_params(axis='y', which='major', left=True)
        ax_volt.tick_params(axis='y', which='major', right=True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_current_temperature(self, title: str = "", results: pd.DataFrame = None) -> None:
        if results is None:
            if self.data_df is None:
                raise ValueError("No data to plot")
            results = self.data_df

        # Conversion
        results[["T_cell", "T_c", "T_h"]] = conv_temp(results[["T_cell", "T_c", "T_h"]].to_numpy(), 'K', 'C')

        x_sim = results["T_h"].to_numpy() - results["T_c"].to_numpy()
        y_sim = results["I_HP"].to_numpy()

        # Arrows
        arrow_step = 80

        # Voltage constraints
        x_vec = np.linspace(-100, 100, self.time_steps)
        y_vec_min, y_vec_max = self.model.get_voltage_constraints(x_vec)

        # Color gradient for COP
        COP_sim = results["COP"].to_numpy()
        mask = y_sim < 0  # get negative current values
        COP = COP_sim.copy() - mask  # correct for negative values by subtracting 1

        cmap = LinearSegmentedColormap.from_list('COP_colormap', ['red', 'green'])
        norm = Normalize(vmin=0, vmax=5)
        points = np.array([x_sim, y_sim]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(COP)
        lc.set_linewidth(2)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axhline(y=0, lw=1, color='black', label='_nolegend_')
        ax.axvline(x=0, lw=1, color='black', label='_nolegend_')
        ax.add_collection(lc)

        for i in range(0, len(x_sim) - arrow_step, arrow_step):
            ax.arrow(
                x_sim[i], y_sim[i],
                x_sim[i + 1] - x_sim[i], y_sim[i + 1] - y_sim[i],
                head_width=0.4, head_length=2.5, fc=cmap(norm(COP[i])), ec=cmap(norm(COP[i])), alpha=0.4
            )

        ax.plot(x_vec, y_vec_min, color='black', linestyle=':', label=r'$U_{max}$')
        ax.plot(x_vec, y_vec_max, color='black', linestyle=':')
        ax.axhline(y=self.model.I_HP_max, color='black', linestyle='--', label=r'$I_{max}$')
        ax.axhline(y=-self.model.I_HP_max, color='black', linestyle='--')
        ax.axvline(x=self.model.DeltaT_max, color='black', linestyle='-.', label=r'$\Delta T_{max}$')
        ax.axvline(x=-self.model.DeltaT_max, color='black', linestyle='-.')

        # Configure plot
        ax.set_xlim(-self.model.DeltaT_max * 1.1, self.model.DeltaT_max * 1.1)
        ax.set_ylim(-self.model.I_HP_max * 1.1, self.model.I_HP_max * 1.1)
        ax.set_xlabel(r'$\Delta T \; [\degree]$')
        ax.set_ylabel(r'$I_\mathrm{HP} \; [A]$')
        ax.set_title('Current-Temperature Phase Space ' + title)
        ax.legend()
        ax.grid()

        # Add minor ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='y', which='minor', direction='in', left=True, right=True)
        ax.tick_params(axis='x', which='major', top=True)
        ax.tick_params(axis='y', which='major', left=True, right=True)

        # Add a box around all edges
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # Display colorbar
        cbar = plt.colorbar(lc, ax=ax, label='COP')
        
        plt.show()

    def plot_HP_operation(self, results:pd.DataFrame=None) -> None:
        if results is None:
            if self.data_df is None:
                raise ValueError("No data to plot")
            results = self.data_df

        # Conversion
        results[["T_cell", "T_c", "T_h"]] = conv_temp(results[["T_cell", "T_c", "T_h"]].to_numpy(), 'K', 'C')

        # Plot
        fig, axs = plt.subplots(1, 1, figsize=(15, 4))

        # U_HP vs DeltaT
        results.plot(
            x="T_h",
            y="U_HP",
            xlabel="T_h [°C]",
            ylabel="U_HP [V]",
            title="U_HP vs T_h",
            ax=axs,
            color="orange"
        )
        plt.plot(x_vec, y_vec_min, color='black', linestyle=':', label=r'$U_{max}$')
        plt.plot(x_vec, y_vec_max, color='black', linestyle=':')
        plt.axhline(y=self.model.I_HP_max, color='black', linestyle='--', label=r'$I_{max}$')
        plt.axhline(y=-self.model.I_HP_max, color='black', linestyle='--')
        plt.axvline(x=self.model.DeltaT_max, color='black', linestyle='-.', label=r'$\Delta T_{max}$')
        plt.axvline(x=-self.model.DeltaT_max, color='black', linestyle='-.')
        
        plt.xlim(-self.model.DeltaT_max * 1.1, self.model.DeltaT_max * 1.1)
        axs.grid()
        plt.show()