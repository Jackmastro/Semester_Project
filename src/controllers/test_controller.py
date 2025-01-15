from .controller_base import ControllerBase
import numpy as np


class TestController(ControllerBase):
    """
    Test controller that returns only zero control inputs
    """
    def __init__(self) -> None:
        self.setpoint = 0.0

    def get_control_input(self, current_time:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Returns control input
        """
        u_HP = 0.0
        u_FAN = 0.0
        return np.array([u_HP, u_FAN])