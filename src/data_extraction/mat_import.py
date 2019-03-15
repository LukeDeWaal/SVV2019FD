import scipy.io as io
import numpy as np
import sys
import os

try:
    from src.misc.data_access import get_data_file_path

except ImportError:
    from misc.data_access import get_data_file_path

dirname = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dirname, r'..\\misc')
path = os.path.abspath(os.path.realpath(path))

sys.path.append(path)


class MatFileImport(object):

    def __init__(self, filename, struct_name='flightdata', series=False, steps=3):

        self.__file = get_data_file_path(filename)      # Filepath of the .mat file
        self.__main_struct = None   # Name of the main struct, normally 'flightdata'
        self.__keys = []            # Names of the sub structs

        self.__data = {}
        self.__units = {}
        self.__descr = {}

        self.__file_import(struct_name)
        self.__get_keys()
        self.__store_data()

        if series is True:

            for key in self.get_keys():
                old_shape = self.__data[key].shape
                self.__data[key] = self.__data[key].reshape((old_shape[0],))

    def __file_import(self, struct_name='flightdata'):
        self.__main_struct = io.loadmat(self.__file)[struct_name]

    def __get_keys(self):
        for key in self.__main_struct.dtype.fields:
            self.__keys.append(key)

    def __store_data(self):
        for key in self.get_keys():
            data = self.__main_struct[key][0,0][0,0][0]
            if data.shape[1] != 1:
                data = data.reshape((data.shape[1], 1))

            units = self.__main_struct[key][0,0][0,0][1][0]

            while True:
                if units.ndim > 0:
                    try:
                        units = units[0]
                    except IndexError:
                        break

                else:
                    break

            data, units = self.__convert_unit(data, units)

            self.__units[key] = units
            self.__descr[key] = self.__main_struct[key][0,0][0,0][2]
            self.__data[key] = data

    @staticmethod
    def __convert_unit(data, unit):

        if unit == 'lbs':
            data *= 0.45359237
            unit = 'kg'

        elif unit == 'lbs/hr':
            data *= 0.000125997881
            unit = 'kg/s'

        elif unit == 'psi':
            data *= 6894.75729
            unit = 'Pa'

        elif unit == 'deg C':
            data += 273.15
            unit = 'K'

        elif unit == 'ft/min':
            data *= 0.00508
            unit = 'm/s'

        elif unit == 'ft':
            data *= 0.3048
            unit = 'm'

        elif unit == 'knots':
            data *= 0.514444444
            unit = 'm/s'

        return data, unit

    def get_keys(self):
        return self.__keys

    def get_data(self):
        return self.__data

    def get_descriptions(self):
        return self.__descr

    def get_units(self):
        return self.__units


if __name__ == "__main__":

    datafile = r'FlightData.mat'
    MatFile = MatFileImport(datafile, series=True)

    keys = MatFile.get_keys()
    descr = MatFile.get_descriptions()
    data = MatFile.get_data()
    units = MatFile.get_units()
