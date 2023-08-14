#!/usr/bin/env python

import sys
import numpy as np

"""
    Description:
        obtains data from a file in the form of a numpy.ndarray

    params:
        filename: a string representing the filename that we wish to
                  read into memory
        dataype: a numpy generic type representing the data type of
                  the elements of the returned array
"""
def np_arr_from_file(filename: str, datatype: np.dtype) -> np.ndarray:
    try:
        with open(filename, 'rb') as fd:
            raw_data = np.fromfile(fd, dtype=datatype)
            fd.close()
    except FileNotFoundError as fnferror:
        raise FileNotFoundError(fnferror)
    else:
        return raw_data

