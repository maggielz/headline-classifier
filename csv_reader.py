# version 1.1

# Modified from dt_provided.py, as provided by Alice Gao, 2021

import csv
import math
import numpy as np  # numpy==1.19.2

def read_data(file_path: str):
    """
    Reads data from file_path, 

    :param file_path: The name of the data file.
    :type filename: str
    :return: A 2d data array consisting of examples 
    :rtype: List[List[int or float]]
    """
    data_array = []
    with open(file_path, 'r') as csv_file:
        # read csv_file into a 2d array
        reader = csv.reader(csv_file)
        for row in reader:
            data_array.append(row)

        # set labels
        feature_names = data_array[0]

        # exclude feature name row
        data_array = data_array[1:]

        return data_array, feature_names


def preprocess_data(data_array, folds_num=10):
    """
    Divides data_array into folds_num sets for cross validation. 
    Each fold has an approximately equal number of examples.

    :param data_array: a set of examples
    :type data_array: List[List[Any]]
    :param folds_num: the number of folds
    :type folds_num: int, default 10
    :return: a list of sets of length folds_num
    Each set contains the set of data for the corrresponding fold.
    :rtype: List[List[List[Any]]]
    """
    fold_size = math.floor(len(data_array) / folds_num)

    folds = []
    for i in range(folds_num):

        if i == folds_num - 1:
            folds.append(data_array[i * fold_size:])
        else:
            folds.append(data_array[i * fold_size: (i + 1) * fold_size])

    return folds
