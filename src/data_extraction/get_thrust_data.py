import numpy as np
import pandas as pd
from src.misc import get_data_file_path


def get_thrust(which=None):
    data = pd.read_csv(get_data_file_path('thrust.dat'), delimiter=' ', index_col=0, header=0)

    if which is None:
        return data

    else:
        if which == 1:
            return data.loc[:6, :]

        elif which == 2:
            return data.loc[6:13, :]

        else:
            return data.loc[13:, :]


if __name__ == "__main__":

    data = get_thrust(3)
