# Simple PID library https://github.com/m-lundberg/simple-pid/tree/master (Documentation: https://pypi.org/project/simple-pid/)
from simple_pid import PID
import numpy as np

class PIDController:
    def __init__(self, kp:float, ki:float, kd:float, setpoint:float, dt:float, output_limits:tuple) -> None:
        # to work, the sample time has to be zero
        self.pid = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=setpoint, sample_time=dt, output_limits=output_limits)

    def get_control_input(self, current_output:float) -> np.ndarray:
        u_HP = self.pid(current_output)
        u_FAN = 1.0
        return np.array([u_HP, u_FAN])