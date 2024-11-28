from abc import ABC, abstractmethod
import numpy as np

class ControllerBase(ABC):

    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_control_input(self, states:np.ndarray, outputs:np.ndarray) -> np.ndarray:
        raise NotImplementedError