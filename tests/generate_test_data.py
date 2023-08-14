#!/usr/bin/env python

import numpy as np

"""
    Description:

        generate data that mimics data outputs from cbmsim for testing.

    params:
        test_config - contains information on what data type to generate
                      as well as data-type specific params, like num_ts,
                      num_trials, num_cells, etc
    returns:
        np.ndarray - contains random values with appropriate shape for
        the data type
    raises:
        TypeError - if data type is None in test_config
        AssertionError - if array dims are not ints and are less than 
            or equal to zero
"""
def generate_test_data(test_config: dict) -> np.ndarray:
    match test_config["data_type"]:
        case "raster":
            num_cells, num_trials, num_ts = \
                test_config["num_cells"], test_config["num_trials"], test_config["num_ts"]
            assert isinstance(num_cells, int) and isinstance(num_trials, int) and isinstance(num_ts, int)
            assert num_cells > 0 and num_trials > 0 and num_ts > 0
            return np.random.randint(0, high=2, size=num_ts*num_trials*num_cells, dtype=np.uint8)
        case "psth":
            num_cells, num_ts = \
                test_config["num_cells"], test_config["num_ts"]
            assert isinstance(num_cells, int) and isinstance(num_ts, int)
            assert num_cells > 0 and num_ts > 0
            return np.random.randint(0, high=2, size=num_ts*num_cells, dtype=np.uint8)
        case "weights":
            num_cells = test_config["num_cells"]
            assert isinstance(num_cells, int)
            assert num_cells > 0
            return np.random.rand(num_cells).astype(np.single)
        case None:
            raise TypeError("Data type not specified in test config")

"""
    Description:
        send the generated test data to disk

    params:
        filename: the string representation of the path to the file

        test_config: a configuration dictionary specifying parameters for
                    returned, serializable numpy array

    re-raises:
        assertion error - from generate_test_data
        type error - from generate_test_data
"""
def test_data_to_file(filename: str, test_config: dict) -> None:
    with open(filename, 'wb') as fd:
        try:
            serializable = generate_test_data(test_config)
        except AssertionError as ae:
            raise ae
        except TypeError as te:
            raise te
        else:
            serializable.tofile(fd)

