import os
import pandas as pd

def load_coefficients(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    return pd.read_csv(filepath)