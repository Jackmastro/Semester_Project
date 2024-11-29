import numpy as np
from scipy.linalg import solve_continuous_are
from Classes import Model

class LQRController_cont:
    """ Infinite horizon LQR controller """
    def __init__(self, model:Model, setpoint:float, A:np.ndarray, B:np.ndarray, Q:np.ndarray, R:np.ndarray) -> None:
        # Calculate offset for converting Tc to T_cell
        R_frac = (model.R_4_lambda + model.R_5) / model.R_5
        self.Tc_inf = model.T_amb * (R_frac - 1) - setpoint * R_frac

        # Solve Riccati equation
        self.P = solve_continuous_are(A, B, Q, R)
        self.K = -np.linalg.inv(R) @ B.T @ self.P

    def get_control_input(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        x[1] = x[1] - self.Tc_inf
        return self.K @ x