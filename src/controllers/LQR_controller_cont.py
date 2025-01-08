from .controller_base import ControllerBase
import numpy as np
from scipy.linalg import solve_continuous_are


class LQRControllerCont(ControllerBase):
    """ Infinite horizon LQR controller """
    def __init__(self, T_top_ref:float, T_cell_ref:float, A:np.ndarray, B:np.ndarray, Q:np.ndarray, R:np.ndarray) -> None:
        # TODO check dimensions, (semi) pos def

        self.setpoint = T_cell_ref # for plotting

        self.x_inf = np.array([1,
                               T_top_ref,
                               T_top_ref]) # TODO check if soc and bot make sense

        # Solve Riccati equation
        self.P = solve_continuous_are(A, B, Q, R)
        self.K = -np.linalg.inv(R) @ B.T @ self.P

    def get_control_input(self, current_time:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        return self.K @ (x - self.x_inf)