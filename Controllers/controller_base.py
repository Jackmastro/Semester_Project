from abc import ABC, abstractmethod
import numpy as np

class ControllerBase(ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_control_input(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
        pass