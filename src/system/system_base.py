from abc import ABC, abstractmethod
import pandas as pd


class SystemBase(ABC):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def run(self) -> pd.DataFrame:
        pass