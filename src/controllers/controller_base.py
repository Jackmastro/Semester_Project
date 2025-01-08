from abc import ABC, abstractmethod
import numpy as np


class ControllerBase(ABC):

    def __init__(self):
        self.setpoint = 0.0 # Needed for plotting
        super().__init__()
    
    @abstractmethod
    def get_control_input(self, current_time:float, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        pass