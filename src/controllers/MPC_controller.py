from .controller_base import ControllerBase
import numpy as np
import cvxpy as cp

class MPController(ControllerBase):
    def __init__(self, model, Q:np.ndarray, S:np.ndarray, R:np.ndarray, pred_time:int, sampling_time:int, discret_time:int, print_output:bool=False, verbose:bool=False) -> None:
        # self.setpoint = setpoint
        
        self.Q = Q
        self.S = S
        self.R = R
        self.pred_time = pred_time
        self.sampling_time = sampling_time
        self.discret_time = discret_time

        self.K = int(pred_time / discret_time)

        self.n = 3 # number of states
        self.m = 2 # number of inputs

        self.A, self.B, self.h, _, _, _ = model.get_linearization()

        self.print_output = print_output
        self.verbose = verbose
        self.num_infeas = 0
        self.num_None_output = 0

        self._init_optimization_problem()

    def _init_optimization_problem(self) -> None:
        # Define variables and dynamic parameters for the optimization problem
        self.x_var          = cp.Variable((self.n, self.K+1))
        self.i_var          = cp.Variable((self.m, self.K))
        self.x0_param       = cp.Parameter(self.n)

        # Placeholders for parametrization
        self.A_d = cp.Parameter((self.n, self.n))
        self.B_d = cp.Parameter((self.n, self.m))
        self.h_d = cp.Parameter(self.n)
        self.xss = cp.Parameter(self.n)

        # Initialize cost and constraints (parametrized)
        self.cost = 0
        self.constraints = [self.x_var[:, 0] == self.x0_param]
        
        # Iterate over the horizon K - 1 for cost and constraints
        for k in range(self.K):
            self.cost += (
                cp.quad_form(self.x_var[:, k] - self.xss, self.Q) + 
                cp.quad_form(self.i_var[k], self.R)
                )
            self.constraints += [
                self.x_var[:, k+1] == self.A_d @ self.x_var[:, k] + self.B_d * self.i_var[k] + self.h_d,
                self.i_var[0, k]   >= - self.model.I_HP_max,
                self.i_var[0, k]   <= self.model.I_HP_max,
                self.i_var[1, k]   >= 0,
                self.i_var[1, k]   <= 1,
                self.x_var[0, k]   >= 0,
                self.x_var[0, k]   <= 1,
                ]
        
        # Terminal cost or constraint for stability
        self.cost += cp.quad_form(self.x_var[:, self.K] - self.xss, self.S)
        # self.constraints += [self.x_var[:, self.K_d] == self.xss]

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)

        if self.print_output:
            print("Is the problem DPP compliant?", self.problem.is_dpp())
            print("Is the problem DCP compliant?", self.problem.is_dcp())

    def _update_linearized_matrices(self, A:np.ndarray, B:np.ndarray, h:np.ndarray) -> None:
        self.A_d.value = A
        self.B_d.value = B
        self.h_d.value = h

    def get_control_input(self, x0, time_index, xss, dss) -> tuple:
        if self.print_output:
            print(f"--  Simulation time: {time_index} min  -  Infeasible solutions: {self.num_infeas} of which None output: {self.num_None_output}  --")
        
        # Save uss and xss at the beginning of the simulation
        if time_index == 0:
            self.xss.value = xss

            # Initialize the open-loop input
            dss.basal_handler_params['open-loop input'] = np.ones(self.K)
        
        # Optimization at every sampling time
        if (time_index % self.sampling_time) == 0:

            self.x0_param.value = x0
        
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
                    
                    control_output = dss.basal_handler_params['open-loop input'][time_index % self.sampling_time]

                    if self.print_output:
                        print(f"Forced control output: {control_output}")
                    return control_output, dss
            
            # Problem solved
            control_output = self.i_var.value

            # Save open-loop input
            dss.basal_handler_params['open-loop input'] = np.repeat(self.i_var[:].value, self.discret_time)

            if self.print_output:
                print(f"Optimized control output: {control_output}")
            return control_output, dss
        
        # Open-loop control
        else:
            control_output = dss.basal_handler_params['open-loop input'][time_index % self.sampling_time]
            
            if self.print_output:
                print(f"Open-loop control output: {control_output}")
            return control_output, dss
