import numpy as np
import pandas as pd

try:
    from src.misc.data_access import get_data_file_path
    from src.data_processing.aerodynamic_coefficients import indicated_to_true_airspeed, ISA

except ImportError:
    from data_access import get_data_file_path
    from aerodynamic_coefficients import indicated_to_true_airspeed, ISA


def pfd_unit_converter(file, data_dict: dict = {}):

    temp_dict = {}

    filepath = get_data_file_path(file)
    temp_dict[file] = pd.read_csv(filepath, delimiter='\t', index_col=0, header=0)

    # Converting Units
    temp_dict[file]['hp'] *= 0.3048
    temp_dict[file]['IAS'] *= 0.514444444
    temp_dict[file]['FFL'] *= 0.000125997881
    temp_dict[file]['FFR'] *= 0.000125997881
    temp_dict[file]['F_used'] *= 0.45359237
    temp_dict[file]['TAT'] += 273.15

    TAS = [indicated_to_true_airspeed(temp_dict[file]['IAS'][i], ISA(temp_dict[file]['hp'][i])[2]) for i in range(len(temp_dict[file]['IAS']))]

    temp_dict[file]['TAS'] = TAS
    data_dict[file] = temp_dict[file]

data = {}
pfd_unit_converter(r'StatClCd.txt', data_dict=data)
pfd_unit_converter(r'StatElev.txt', data_dict=data)

