from .controller_base import ControllerBase
# Simple PID library https://github.com/m-lundberg/simple-pid/tree/master (Documentation: https://pypi.org/project/simple-pid/)
from simple_pid import PID
import numpy as np


class PIDController_T_top(ControllerBase):
    def __init__(self, kp:float, ki:float, kd:float, T_top_ref:float, T_cell_ref:float, sampling_time:float, output_limits:tuple) -> None:
        self.setpoint = T_cell_ref # for plotting

        self.u_prev = np.zeros((2, 1))
        self.sampling_time = sampling_time
        self.last_update_time = None

        assert sampling_time > 0, "sampling_time must be positive"
        
        self.pid = PID(Kp=kp, Ki=ki, Kd=kd, setpoint=T_top_ref, sample_time=0.0, output_limits=output_limits)

    def get_control_input(self, current_time:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        if self.last_update_time is None or (current_time - self.last_update_time) >= self.sampling_time:
            self.last_update_time = current_time
            self.u_prev = np.array([self.pid(x[1]),
                                    1.0])

        return self.u_prev