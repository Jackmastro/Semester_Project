import os
import numpy as np
import pandas as pd


def save_matrices2csv(names:list, variables:tuple) -> None:
    directory = os.path.join(os.path.dirname(__file__), "state_space/")
    for name, variable in zip(names, variables):
        np.savetxt(f"{directory}{name}.csv", variable, delimiter=',')

def save_bounds2csv(df:pd.DataFrame, filename:str) -> None:
    directory = os.path.join(os.path.dirname(__file__), "operational_bounds/")
    df.to_csv(f"{directory}{filename}.csv", index=False)