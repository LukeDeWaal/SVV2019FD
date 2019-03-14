import numpy as np
import pandas as pd
from src.data_extraction import MatFileImport, pfd_unit_converter


class Data(object):

    def __init__(self, mat_file, *pfd_files):
        """
        :param mat_file: Real Time Data file (.mat)
        :param pfd_files: Text files including the data from the PFD Excel file
        """

        self.__mat_filepath = mat_file
        self.__pfd_filepaths = pfd_files

        self.__mat_data = MatFileImport(self.__mat_filepath)
        self.__pfd_data = {}
        self.__store_pfd_data()

    def __store_pfd_data(self):
        for file in self.__pfd_filepaths:
            pfd_unit_converter(file, data_dict=self.__pfd_data)

    def get_mat(self, which=None):
        """
        Get the data from the .mat file
        :param which: WHat data needs to be returned; None will return the entire object
        :return: Requested Data
        """

        if which is None:
            return self.__mat_data

        if which.lower() == 'data':
            return self.__mat_data.get_data()

        elif which.lower() == 'keys':
            return self.__mat_data.get_keys()

        elif which.lower() == 'units':
            return self.__mat_data.get_units()

        elif which.lower() == 'descr' or which.lower() == 'descriptions':
            return self.__mat_data.get_descriptions()

        else:
            return self.__mat_data

    def get_pfd(self, which=None):
        """
        Return pfd data
        :param which: Which series of data needs to be returned (aka. the file name)
        :return: Requested pfd data
        """

        try:
            return self.__pfd_data[which]

        except KeyError:
            return self.__pfd_data





