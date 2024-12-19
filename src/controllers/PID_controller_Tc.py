from .controller_base import ControllerBase
# Simple PID library https://github.com/m-lundberg/simple-pid/tree/master (Documentation: https://pypi.org/project/simple-pid/)
from simple_pid import PID
import numpy as np

class PIDController_Tc(ControllerBase):
    def __init__(self, kp:float, ki:float, kd:float, setpoint_T_c:float, dt:float, output_limits:tuple) -> None:
        self.setpoint = setpoint_T_c
        
        self.pid = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=setpoint_T_c, sample_time=dt, output_limits=output_limits)

    def get_control_input(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        u_HP = self.pid(x[1])
        u_FAN = 1.0
        return np.array([u_HP, u_FAN])