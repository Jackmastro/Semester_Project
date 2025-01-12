import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.constants import convert_temperature as conv_temp

from classes import Model
from controllers import ControllerBase
from img import save_plot2pdf


class Simulation:
    def __init__(self, model:Model, controller:ControllerBase, dt_sim:float, time_span:float) -> None:
        self.model = model
        self.controller = controller

        assert dt_sim > 0, "dt_sim must be greater than 0"
        assert time_span > 0, "time_span must be greater than 0"
        assert time_span >= dt_sim, "time_span must be greater than or equal to dt_sim"

        self.dt_sim = dt_sim
        self.time_span = time_span
        self.time_steps = int(self.time_span / self.dt_sim) + 1 # Python counting

        # Placeholder for plots
        self.initial_time_span = 0
        self.initial_time_steps = 0

        self.x0 = self.model.get_initial_state

        self.data = {
            "time":   [],
            "x_SoC":  [self.x0[0]],
            "T_top":  [self.x0[1]],
            "T_bot":  [self.x0[2]],
            "I_HP":   [],
            "x_FAN":  [],
            "T_cell": [self.x0[1]], # Assuming equal to Tc
            "I_BT":   [],
            "U_BT":   [],
            "U_oc":   [],
            "U_HP":   [],
            "COP":    [],
            "x_LED":  [],
        }
        self.data_df:pd.DataFrame = pd.DataFrame()

        self.colors = {
            "battery": "green",
            "HP":      "orange",
            "fan":     "purple",
            "cell":    "lightblue",
            "cold":    "blue",
            "hot":     "red",
            "led":     "pink",
            "rest":    "gray",
        }

    def _append_sim_data(self, current_time:float, x_next:np.ndarray, u_bounded:np.ndarray, values:dict) -> None:
        self.data["time"].append(current_time)
        self.data["x_SoC"].append(x_next[0])
        self.data["T_top"].append(x_next[1])
        self.data["T_bot"].append(x_next[2])
        self.data["I_HP"].append(u_bounded[0])
        self.data["x_FAN"].append(u_bounded[1])
        self.data["T_cell"].append(values["T_cell"])
        self.data["I_BT"].append(values["I_BT"])
        self.data["U_BT"].append(values["U_BT"])
        self.data["U_oc"].append(values["U_oc"])
        self.data["U_HP"].append(values["U_HP"])
        self.data["COP"].append(values["COP"])
        self.data["x_LED"].append(values["x_LED"])

    def run(self, with_initial_time:bool=False, initial_time_span:int=100) -> pd.DataFrame:
        x_prev = self.x0

        # Initial time steps
        if with_initial_time:
            assert initial_time_span > 0, "initial_time_span must be greater than 0"
            assert initial_time_span >= self.dt_sim, "initial_time_span must be greater than or equal to dt_sim"

            self.initial_time_span = -initial_time_span

            self.initial_time_steps = int(initial_time_span / self.dt_sim) + 1 # Python counting
            u = np.array([0.0,
                          0.0])

            for t in reversed(range(1, self.initial_time_steps)): # at 0 starts the simulation with LEDs
                current_time = -t * self.dt_sim

                # Advance system states
                x_next, u_bounded = self.model.discretized_update(u, self.dt_sim, LED_off=True)

                # Get internal values
                values = self.model.get_values(x_prev, u_bounded, LED_off=True)

                self._append_sim_data(current_time, x_next, u_bounded, values)

        # Main time steps
        for t in range(self.time_steps):
            current_time = t * self.dt_sim

            y = self.model.get_output()

            u = self.controller.get_control_input(current_time, x_prev, y)

            # Advance system states
            x_next, u_bounded = self.model.discretized_update(u, self.dt_sim)

            # Get internal values
            values = self.model.get_values(x_prev, u_bounded)

            self._append_sim_data(current_time, x_next, u_bounded, values)

            x_prev = x_next

        # Remove last element of the states and temperatures to have equal length
        self.data["x_SoC"].pop()
        self.data["T_top"].pop()
        self.data["T_bot"].pop()
        self.data["T_cell"].pop()

        self.data_df = pd.DataFrame(self.data)

        return self.data_df
    
    def plot_time_results(self, title:str="", save_plot:bool=False, filename:str=None, results:pd.DataFrame=None) -> None:
        if results is None:
            if self.data_df is None:
                raise ValueError("No data to plot")
            results = self.data_df

        xlimits = (self.initial_time_span, self.time_span)

        fig, axs = plt.subplots(2, 3, figsize=(15, 5))

        # Temperatures (first row spanning all columns)
        for ax in axs[0]:  # Remove all subplots in the first row
            ax.remove()
        ax_temp = fig.add_subplot(2, 1, 1)  # Create a new subplot spanning the top row
        ax_temp.axhline(y=0, lw=1, color="black", label='_nolegend_')
        ax_temp.axvline(x=0, lw=1, color="black", label='_nolegend_')
        ax_temp.axhline(y=conv_temp(self.controller.setpoint, 'K', 'C'), color='black', linestyle='--', label=r'$T_\mathrm{cell,ref}$')
        ax_temp.axhline(y=conv_temp(self.model.T_amb, 'K', 'C'), color='gray', linestyle='-.', label=r'$T_\mathrm{amb}$')
        ax_temp.plot(results["time"], conv_temp(results["T_cell"].to_numpy(), 'K', 'C'), color=self.colors["cell"], label=r'$T_\mathrm{cell}$')
        ax_temp.plot(results["time"], conv_temp(results["T_top"].to_numpy(), 'K', 'C'), color=self.colors["cold"], label=r'$T_\mathrm{top}$')
        ax_temp.plot(results["time"], conv_temp(results["T_bot"].to_numpy(), 'K', 'C'), color=self.colors["hot"], label=r'$T_\mathrm{bot}$')
        ax_temp.set_xlabel('Time [s]')
        ax_temp.set_ylabel('Temperature [°C]')
        ax_temp.set_title(title)
        ax_temp.legend(loc="center right")
        ax_temp.grid()
        ax_temp.set_xlim(xlimits)
        ax_temp.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_temp.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_temp.tick_params(axis='x', which='minor', direction='in', top=True)
        ax_temp.tick_params(axis='y', which='minor', direction='in', left=True, right=True)
        ax_temp.tick_params(axis='x', which='major', top=True)
        ax_temp.tick_params(axis='y', which='major', left=True, right=True)
        ax_temp.spines['top'].set_visible(True)
        ax_temp.spines['right'].set_visible(True)
        ax_temp.spines['bottom'].set_visible(True)
        ax_temp.spines['left'].set_visible(True)

        # x_SoC and x_FAN (second row, first column)
        axs[1, 0].axhline(y=0, lw=1, color="black", label='_nolegend_')
        axs[1, 0].axvline(x=0, lw=1, color="black", label='_nolegend_')
        axs[1, 0].plot(results["time"], results["x_SoC"], color=self.colors["battery"], label=r'$x_\mathrm{SoC}$')
        axs[1, 0].plot(results["time"], results["x_FAN"], color=self.colors["fan"], label=r'$x_\mathrm{fan}$')
        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 0].set_ylabel('Percentage [-]')
        axs[1, 0].set_title("SoC and fan duty cycle")
        axs[1, 0].grid()
        axs[1, 0].set_xlim(xlimits)
        axs[1, 0].set_ylim(-0.1, 1.1)
        axs[1, 0].legend(loc="center right")
        axs[1, 0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1, 0].tick_params(axis='x', which='minor', direction='in', top=True)
        axs[1, 0].tick_params(axis='y', which='minor', direction='in', left=True, right=True)
        axs[1, 0].tick_params(axis='x', which='major', top=True)
        axs[1, 0].tick_params(axis='y', which='major', left=True, right=True)
        axs[1, 0].spines['top'].set_visible(True)
        axs[1, 0].spines['right'].set_visible(True)
        axs[1, 0].spines['bottom'].set_visible(True)
        axs[1, 0].spines['left'].set_visible(True)

        # Currents (second row, second column)
        axs[1, 1].axhline(y=0, lw=1, color="black", label='_nolegend_')
        axs[1, 1].axvline(x=0, lw=1, color="black", label='_nolegend_')
        axs[1, 1].axhline(y=self.model.I_rest, color=self.colors["rest"], label=r'$I_{rest}$')
        axs[1, 1].plot(results["time"], results["x_LED"] * self.model.I_LED, color=self.colors["led"], label=r'$I_\mathrm{led}$')
        axs[1, 1].plot(results["time"], results["x_FAN"] * self.model.I_FAN, color=self.colors["fan"], label=r'$I_\mathrm{fan}$')
        axs[1, 1].plot(results["time"], results["I_BT"], color=self.colors["battery"], label=r'$I_\mathrm{bt}$')
        axs[1, 1].plot(results["time"], results["I_HP"], color=self.colors["HP"], label=r'$I_\mathrm{hp}$')
        # axs[1, 1].plot(results["time"], np.abs(results["I_HP"]) + 
        #                 results["x_FAN"] * self.model.I_FAN +
        #                 self.model.I_LED * self.model.x_LED_tot +
        #                 self.model.I_rest, color='black', linestyle='--', label=r'$I_{tot}$')
        axs[1, 1].set_xlabel('Time [s]')
        axs[1, 1].set_ylabel('Current [A]')
        axs[1, 1].set_title("Currents")
        axs[1, 1].grid()
        axs[1, 1].set_xlim(xlimits)
        # axs[1, 1].set_ylim(bottom=-6.1)
        axs[1, 1].legend(loc="center right")
        axs[1, 1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1, 1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1, 1].tick_params(axis='x', which='minor', direction='in', top=True)
        axs[1, 1].tick_params(axis='y', which='minor', direction='in', left=True, right=True)
        axs[1, 1].tick_params(axis='x', which='major', top=True)
        axs[1, 1].tick_params(axis='y', which='major', left=True, right=True)
        axs[1, 1].spines['top'].set_visible(True)
        axs[1, 1].spines['right'].set_visible(True)
        axs[1, 1].spines['bottom'].set_visible(True)
        axs[1, 1].spines['left'].set_visible(True)

        # Voltages (second row, third column)
        axs[1, 2].axhline(y=0, lw=1, color="black", label='_nolegend_')
        axs[1, 2].axvline(x=0, lw=1, color="black", label='_nolegend_')
        axs[1, 2].axhline(y=self.model.U_rest, color=self.colors["rest"], label=r'$U_{rest}$')
        axs[1, 2].plot(results["time"], results["U_BT"], color=self.colors["led"], label=r'$U_\mathrm{led}$')
        axs[1, 2].plot(results["time"], results["x_FAN"] * self.model.U_FAN, color=self.colors["fan"], label=r'$U_\mathrm{fan}$')
        axs[1, 2].plot(results["time"], results["U_BT"], color=self.colors["battery"], label=r'$U_\mathrm{bt}$')
        axs[1, 2].plot(results["time"], results["U_HP"], color=self.colors["HP"], label=r'$U_\mathrm{hp}$')
        axs[1, 2].set_xlabel('Time [s]')
        axs[1, 2].set_ylabel('Voltage [V]')
        axs[1, 2].set_title("Voltages")
        axs[1, 2].grid()
        axs[1, 2].set_xlim(xlimits)
        axs[1, 2].set_ylim(-4.3, 12.3)
        axs[1, 2].legend(loc="center right")
        axs[1, 2].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1, 2].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[1, 2].tick_params(axis='x', which='minor', direction='in', top=True)
        axs[1, 2].tick_params(axis='y', which='minor', direction='in', left=True, right=True)
        axs[1, 2].tick_params(axis='x', which='major', top=True)
        axs[1, 2].tick_params(axis='y', which='major', left=True, right=True)
        axs[1, 2].spines['top'].set_visible(True)
        axs[1, 2].spines['right'].set_visible(True)
        axs[1, 2].spines['bottom'].set_visible(True)
        axs[1, 2].spines['left'].set_visible(True)

        plt.tight_layout()
        if save_plot:
            filename = "sim_" + filename
            save_plot2pdf(filename, fig)
        plt.show()

    def plot_current_temperature(self, title:str="", save_plot:bool=False, filename:str=None, results:pd.DataFrame=None) -> None:
        if results is None:
            if self.data_df is None:
                raise ValueError("No data to plot")
            results = self.data_df

        DT_HP_sim = results["T_bot"].to_numpy() - results["T_top"].to_numpy()
        I_HP_sim = results["I_HP"].to_numpy()

        # Arrows
        arrow_step = 80

        # Voltage constraints
        x_vec = np.linspace(-100, 100, self.time_steps)
        y_vec_min, y_vec_max = self.model.get_constraints_U_BT2I_HP(x_vec)

        ### Color gradient for COP
        COP = results["COP"].to_numpy()

        cmap = LinearSegmentedColormap.from_list('COP_colormap', ['red', 'green'])
        norm = Normalize(vmin=0, vmax=5)
        points = np.array([DT_HP_sim, I_HP_sim]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(COP)
        lc.set_linewidth(2)

        ### Plot
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axhline(y=0, lw=1, color='black', label='_nolegend_')
        ax.axvline(x=0, lw=1, color='black', label='_nolegend_')
        ax.add_collection(lc)

        for i in range(0, len(DT_HP_sim) - arrow_step, arrow_step):
            ax.arrow(
                DT_HP_sim[i], I_HP_sim[i],
                DT_HP_sim[i + 1] - DT_HP_sim[i], I_HP_sim[i + 1] - I_HP_sim[i],
                head_width=0.4, head_length=2.5, fc=cmap(norm(COP[i])), ec=cmap(norm(COP[i])), alpha=0.4
            )

        # Constraints
        x_lim_max = self.model.DeltaT_max * 1.1
        y_lim_max = self.model.I_HP_max * 1.1

        fill_color = 'gray'
        fill_alpha = 0.5
        hatch_style = '\\\\'

        ax.plot(x_vec, y_vec_max, color='black', linestyle=':', label=r'$U_\mathrm{max}$')
        # ax.fill_between(x_vec, y_vec_max, y_lim_max, color=fill_color, alpha=fill_alpha)
        ax.fill_between(x_vec, y_vec_max, y_lim_max, color='none', hatch=hatch_style, edgecolor=fill_color, linewidth=0, label='_nolegend_')

        ax.plot(x_vec, y_vec_min, color='black', linestyle=':')
        # ax.fill_between(x_vec, y_vec_min, -y_lim_max, color=fill_color, alpha=fill_alpha)
        ax.fill_between(x_vec, y_vec_min, -y_lim_max, color='none', hatch=hatch_style, edgecolor=fill_color, linewidth=0, label='_nolegend_')

        ax.axhline(y=self.model.I_HP_max, color='black', linestyle='--', label=r'$I_\mathrm{max}$')
        # ax.fill_between(x_vec, self.model.I_HP_max, y_lim_max, color=fill_color, alpha=fill_alpha)
        ax.fill_between(x_vec, self.model.I_HP_max, y_lim_max, color='none', hatch=hatch_style, edgecolor=fill_color, linewidth=0, label='_nolegend_')

        ax.axhline(y=-self.model.I_HP_max, color='black', linestyle='--')
        # ax.fill_between(x_vec, -self.model.I_HP_max, -y_lim_max, color=fill_color, alpha=fill_alpha)
        ax.fill_between(x_vec, -self.model.I_HP_max, -y_lim_max, color='none', hatch=hatch_style, edgecolor=fill_color, linewidth=0, label='_nolegend_')

        ax.axvline(x=self.model.DeltaT_max, color='black', linestyle='-.', label=r'$\Delta T_\mathrm{hp,max}$')
        # ax.fill_betweenx([-y_lim_max, y_lim_max], self.model.DeltaT_max, x_lim_max, color=fill_color, alpha=fill_alpha)
        ax.fill_betweenx([-y_lim_max, y_lim_max], self.model.DeltaT_max, x_lim_max, color='none', hatch=hatch_style, edgecolor=fill_color, linewidth=0, label='_nolegend_')
        
        ax.axvline(x=-self.model.DeltaT_max, color='black', linestyle='-.')
        # ax.fill_betweenx([-y_lim_max, y_lim_max], -self.model.DeltaT_max, -x_lim_max, color=fill_color, alpha=fill_alpha)
        ax.fill_betweenx([-y_lim_max, y_lim_max], -self.model.DeltaT_max, -x_lim_max, color='none', hatch=hatch_style, edgecolor=fill_color, linewidth=0, label='_nolegend_')

        # Configure plot
        ax.set_xlim(-x_lim_max, x_lim_max)
        ax.set_ylim(-y_lim_max, y_lim_max)
        ax.set_xlabel(r'$\Delta T_\mathrm{hp} \; [°C]$')
        ax.set_ylabel(r'$I_\mathrm{hp} \; [A]$')
        ax.set_title(title)
        ax.legend(loc='upper left')
        # ax.legend()
        ax.grid()
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='y', which='minor', direction='in', left=True, right=True)
        ax.tick_params(axis='x', which='major', top=True)
        ax.tick_params(axis='y', which='major', left=True, right=True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        plt.colorbar(lc, ax=ax, label=r'$\mathrm{COP}_\mathrm{hp}$')
        
        if save_plot:
            filename = "I_DT_" + filename
            save_plot2pdf(filename, fig)
        plt.show()