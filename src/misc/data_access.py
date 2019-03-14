import os


def get_data_file_path(filename):
    """
    Use this to get the absolute path to the data file in the data folder
    :param filename: name of the data file
    :return: absolute path
    """

    cwd_list = os.getcwd().split('\\')

    data_folder = []

    for folder in cwd_list:
        if folder == 'SVV2019FD':
            data_folder.append(folder)
            break
        data_folder.append(folder)

    "/".join(data_folder) + '/data/'+filename

    return "/".join(data_folder) + '/data/'+filename

