import os


def get_data_file_path(filename):
    """
    Use this to get the absolute path to the data file in the data folder
    :param filename: name of the data file
    :return: absolute path
    """

    return "/".join(os.getcwd().split('\\')[:-2])+f"/data/{filename}"

