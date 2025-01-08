import os
import matplotlib.pyplot as plt

def save_plot2pdf(filename:str, fig:plt) -> None:
    filename_w_ext = filename + ".pdf"
    filepath = os.path.join(os.path.dirname(__file__), filename_w_ext)
    fig.savefig(filepath, bbox_inches='tight')