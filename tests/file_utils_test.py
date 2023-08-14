#!/usr/bin/env python

import os

from .generate_test_data import generate_test_data, test_data_to_file
from cbm_pack.file_utils import np_arr_from_file

def np_arr_from_file_test(filename: str, datatype: np.dtype) -> None:
    raster_test_file, psth_test_file, weights_test_file = \
        "test.r", "test.p", "test.w"
    raster_test_config = {
        "data_type": "raster",
        "num_cells": 32,
        "num_trials": 500,
        "num_ts": 1300
    }
    psth_test_config = {
        "data_type": "psth",
        "num_cells": 32,
        "num_trials": 500,
        "num_ts": 1300
    }
    weights_test_config = {
        "data_type": "weights",
        "num_cells": 32,
    }
    try:
        test_data_to_file(raster_test_file, raster_test_config)
        test_data_to_file(psth_test_file, psth_test_config)
        test_data_to_file(weights_test_file, weights_test_config)
    except:
        pass # TODO

    try:
        raster_test_data = np_arr_from_file(raster_test_file, np.uint8)
        psth_test_data = np_arr_from_file(psth_test_file, np.uint8)
        weights_test_data = np_arr_from_file(weights_test_file, np.uint8)
    except:
        pass # TODO

    try:
        assert isinstance(raster_test_data, np.ndarray)
        assert isinstance(psth_test_data, np.ndarray)
        assert isinstance(weights_test_data, np.ndarray)
        
        assert raster_test_data.size == \
            raster_test_config["num_cells"] \
            * raster_test_config["num_trials"] \
            * raster_test_config["num_ts"]

        assert psth_test_data.size == \
            psth_test_config["num_cells"] \
            * psth_test_config["num_ts"]

        assert weights_test_data.size == psth_test_config["num_cells"]
    except AssertionError as ae:
        raise ae
    if os.path.isfile(raster_test_file):
        os.remove(raster_test_file)
    if os.path.isfile(psth_test_file):
        os.remove(psth_test_file)
    if os.path.isfile(weights_test_file):
        os.remove(weights_test_file)

