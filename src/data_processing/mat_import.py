import scipy.io as io
import numpy as np


class MatFileImport(object):

    def __init__(self, filepath, struct_name='flightdata'):

        self.__file = filepath      # Filepath of the .mat file
        self.__main_struct = None   # Name of the main struct, normally 'flightdata'
        self.__keys = []            # Names of the sub structs

        self.__data = {}
        self.__units = {}
        self.__descr = {}

        self.__file_import(struct_name)
        self.__get_keys()
        self.__store_data()

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
                
            self.__data[key] = data
            self.__units[key] = self.__main_struct[key][0,0][0,0][1]
            self.__descr[key] = self.__main_struct[key][0,0][0,0][2]

    def get_keys(self):
        return self.__keys

    def get_data(self):
        return self.__data

    def get_descriptions(self):
        return self.__descr


if __name__ == "__main__":

    M = MatFileImport('C:/Users/Luke/Downloads/test1.mat')

    keys = M.get_keys()
    descr = M.get_descriptions()
    data = M.get_data()