import pandas as pd
from src.misc import get_data_file_path
from src.data_processing.aerodynamics import indicated_to_true_airspeed, ISA


def pfd_time_converter(time_col, ET_col):

    new_col = []
    for time, sec in zip(time_col, ET_col):
        t = time.split(":")
        t = [float(_) for _ in t]
        new_col.append(round(t[0]*3600.0 + t[1]*60.0 + float(sec), 5))

    return new_col


def pfd_unit_converter(file, data_dict: dict = {}):
    """
    Convert all units in the Post FLight Data Sheet
    :param file: Filename
    :param data_dict: dictionary in which to store
    :return: None
    """

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
    temp_dict[file]['time'] = pfd_time_converter(temp_dict[file]['time'], temp_dict[file]['ET'])
    data_dict[file] = temp_dict[file]


# data = {}
#
# pfd_unit_converter(r'StatClCd.txt', data_dict=data)
# pfd_unit_converter(r'StatElev.txt', data_dict=data)
# pfd_unit_converter(r'GravShift.txt', data_dict=data)
