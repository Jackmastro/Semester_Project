from .controller_base import ControllerBase
import numpy as np
from scipy.linalg import solve_discrete_are


class LQRControllerDisc(ControllerBase):
    """ Infinite horizon discrete LQR controller """
    def __init__(self, T_top_ref:float, T_cell_ref:float, A_d:np.ndarray, B_d:np.ndarray, Q:np.ndarray, R:np.ndarray, dt:float) -> None:
        
        self.u_prev = np.zeros((2, 1))
        self.dt = dt
        self.last_update_time = None

        assert dt > 0, "dt must be positive"
        # TODO check dimensions, (semi) pos def

        self.setpoint = T_cell_ref # for plotting

        self.x_inf = np.array([1,
                               T_top_ref,
                               T_top_ref])  # TODO Check if soc and bot make sense

        # Solve Riccati equation
        self.P = solve_discrete_are(A_d, B_d, Q, R)
        self.K = -np.linalg.inv(R + B_d.T @ self.P @ B_d) @ B_d.T @ self.P @ A_d

    def get_control_input(self, current_time: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns the control input only when the time increment matches `dt`.
        """
        if self.last_update_time is None or (current_time - self.last_update_time) >= self.dt:
            self.last_update_time = current_time
            self.u_prev = self.K @ (x - self.x_inf)
        
        return self.u_prev
