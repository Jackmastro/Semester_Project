from .controller_base import ControllerBase
import numpy as np


class TestController(ControllerBase):
    def __init__(self) -> None:
        self.setpoint = 0.0

    def get_control_input(self) -> np.ndarray:
        u_HP = 0.0
        u_FAN = 0.0
        return np.array([u_HP, u_FAN])