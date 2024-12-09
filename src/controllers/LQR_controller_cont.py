from .controller_base import ControllerBase
import numpy as np
from scipy.linalg import solve_continuous_are
from classes import Model

class LQRController_cont(ControllerBase):
    """ Infinite horizon LQR controller """
    def __init__(self, model:Model, setpoint:float, A:np.ndarray, B:np.ndarray, Q:np.ndarray, R:np.ndarray) -> None:
        self.setpoint = setpoint
        
        # Calculate offset for converting Tc to T_cell
        R_frac = (model.R_4_lambda + model.R_5) / model.R_5
        self.Tc_inf = (setpoint - model.T_amb) * R_frac + model.T_amb

        self.x_bar = np.array([1, self.Tc_inf, model.T_amb])

        # Solve Riccati equation
        self.P = solve_continuous_are(A, B, Q, R)
        self.K = -np.linalg.inv(R) @ B.T @ self.P

    def get_control_input(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        return self.K @ (x - self.x_bar)