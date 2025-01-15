from .controller_base import ControllerBase
import numpy as np
import pandas as pd
import cvxpy as cp
from classes import Model

class MPCController(ControllerBase):
    """
    Linear nominal MPC controller in discrete time
    """
    def __init__(self, model:Model, T_top_ref:float, T_cell_ref:float,
                 A_d:np.ndarray, B_d:np.ndarray, h_d:np.ndarray, Q:np.ndarray, S:np.ndarray, R:np.ndarray,
                 pred_time:int, sampling_time:int, discret_time:int,
                 print_output:bool=False, verbose:bool=False) -> None:
        
        self.setpoint = T_cell_ref # for plotting

        self.x_inf = np.array([1,
                               T_top_ref,
                               T_top_ref]) # TODO check if soc and bot make sense
        
        self.Q = Q
        self.S = S
        self.R = R
        self.pred_time = pred_time         # s: prediction horizon
        self.sampling_time = sampling_time # s: sampling time for the calculation of control inputs
        self.discret_time = discret_time   # s: discretization time for the model evolution

        self.N = int(pred_time / discret_time) # prediction step

        self.n = A_d.shape[0] # number of states
        self.m = B_d.shape[1] # number of inputs

        self.A_d, self.B_d, self.h_d = A_d, B_d, h_d

        self.model = model

        self.print_output = print_output
        self.verbose = verbose
        self.num_infeas = 0
        self.num_None_output = 0

        self._init_constraints()

        self._init_optimization_problem()

        self._update_open_loop_input(current_time=0)
        self.last_update_time = None

    def _init_constraints(self) -> None:
        # I_HP constraints
        I_HP_min_U_BT, I_HP_max_U_BT = self.model.get_constraints_U_BT2I_HP(self.x_inf[2] - self.x_inf[1]) # TODO distinguere se scaldando o raffreddando
        # I_HP_min_I_BT, I_HP_max_I_BT = self.model.get_constraints_I_BT2I_HP(u_bounded)
        self.I_min = max(I_HP_min_U_BT, self.model.I_HP_min)
        self.I_max = min(I_HP_max_U_BT, self.model.I_HP_max)

    def _init_optimization_problem(self) -> None:
        """
        Parametrization of the optimization problem in the initial state. DPP compliant for fast computations
        """
        # Define variables and dynamic parameters for the optimization problem
        self.x_var    = cp.Variable((self.n, self.N+1))
        self.i_var    = cp.Variable((self.m, self.N))
        self.x0_param = cp.Parameter(self.n)

        # Placeholders for parametrization
        # self.A_d_param   = cp.Parameter((self.n, self.n)) # maybe not needed: only for online linearization
        # self.B_d_param   = cp.Parameter((self.n, self.m))
        # self.h_d_param   = cp.Parameter(self.n)
        # self.x_inf_param = cp.Parameter(self.n)

        # Initialize cost and constraints (parametrized)
        self.cost = 0
        self.constraints = [self.x_var[:, 0] == self.x0_param]
        
        # Iterate over the horizon K - 1 for cost and constraints
        for k in range(self.N):
            self.cost += (
                cp.quad_form(self.x_var[:, k] - self.x_inf, self.Q) + 
                cp.quad_form(self.i_var[:, k], self.R)
                )
            self.constraints += [
                self.x_var[:, k+1] == self.A_d @ self.x_var[:, k] + self.B_d @ self.i_var[:, k] + self.h_d,
                self.i_var[0, k]   >= self.I_min, # Peltier
                self.i_var[0, k]   <= self.I_max,
                self.i_var[1, k]   >= 0, # fan
                self.i_var[1, k]   <= 1,
                # self.x_var[0, k]   >= 0, # SoC
                # self.x_var[0, k]   <= 1,
                ]
        
        # Terminal cost or constraint for stability
        self.cost += cp.quad_form(self.x_var[:, self.N] - self.x_inf, self.S)
        # self.constraints += [self.x_var[:, self.N] == self.x_inf]

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)

        if self.print_output:
            print("Is the problem DPP compliant?", self.problem.is_dpp())
            print("Is the problem DCP compliant?", self.problem.is_dcp())

    # def _update_linearized_matrices(self, A_d:np.ndarray, B_d:np.ndarray, h_d:np.ndarray) -> None:
    #     self.A_d_param.value = A_d
    #     self.B_d_param.value = B_d
    #     self.h_d_param.value = h_d

    def _update_open_loop_input(self, current_time:float, optimal_input_values:np.ndarray=None) -> None:
        """
        Updates open-loop control inputs
        """
        # Extend time to one before current_time
        time_index = np.hstack((current_time - self.discret_time,
                                np.arange(current_time,
                                          current_time + self.N * self.discret_time,
                                          self.discret_time)))
        
        # Initialize with zeros
        if optimal_input_values is None:
            self.open_loop_u = pd.DataFrame(np.zeros((time_index.shape[0], self.m)), columns=['I_HP', 'x_FAN'], index=time_index)
        
        # Update with optimal values
        else:
            opt_input = np.hstack((optimal_input_values[:, 0].reshape(-1, 1),
                                   optimal_input_values)).T
            self.open_loop_u = pd.DataFrame(opt_input, columns=['I_HP', 'x_FAN'], index=time_index)

    def _get_open_loop_u(self, current_time:float) -> np.ndarray:
        """
        Returns the open-loop control inputs saved at previous iterations
        """
        return self.open_loop_u.loc[self.open_loop_u.index <= current_time].iloc[-1]

    def get_control_input(self, current_time:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Solves the optimization problem at the current state with MOSEK solver (backup CLARABEL)
        """
        if self.print_output:
            print(f"--  Simulation time: {current_time} sec  -  Infeasible solutions: {self.num_infeas} of which None output: {self.num_None_output}  --")

        # # Update linearized matrices
        # if current_time == 0:
        #     self.x_inf_param = self.x_inf
        #     self._update_linearized_matrices(self.A_d, self.B_d, self.h_d)
        
        # Optimization at every sampling time
        if self.last_update_time is None or (current_time - self.last_update_time) >= self.sampling_time:

            self.x0_param.value = x
        
            # Solve the optimization problem
            try:
                self.problem.solve(solver=cp.MOSEK, verbose=self.verbose)

                # Check if the solution is available
                if self.i_var.value is None:
                    raise Exception("Solver did not find a feasible solution.")
            
            # Fallback solver
            except:
                try:
                    self.problem.solve(solver=cp.CLARABEL, verbose=self.verbose)

                    # Check if the solution is available
                    if self.i_var.value is None:
                        self.num_None_output += 1
                        raise Exception("Fallback solver did not find a feasible solution.")

                except:
                    self.num_infeas += 1
                    
                    control_input = self._get_open_loop_u(current_time)

                    if self.print_output:
                        print(f"Forced control output: {control_input}")
                    return control_input
            
            # Problem solved
            control_input = self.i_var[:, 0].value

            # Save open-loop input
            self._update_open_loop_input(current_time, self.i_var.value)

            if self.print_output:
                print(f"Optimized control output: {control_input}")

            # Update last_update_time
            self.last_update_time = current_time

            return control_input
        
        # Open-loop control
        else:
            control_input = self._get_open_loop_u(current_time)
            
            if self.print_output:
                print(f"Open-loop control output: {control_input}")
            return control_input