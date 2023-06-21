#!/usr/bin/env python

import sys
import numpy as np

def np_arr_from_file(filename, datatype):
    try:
        with open(filename, 'rb') as fd:
            raw_data = np.fromfile(fd, dtype=datatype)
            fd.close()
    except FileNotFoundError as fnferror:
        raise FileNotFoundError(fnferror)
    else:
        return raw_data

