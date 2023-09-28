#!/usr/bin/env python

import numpy as np
import timeit as timeit
from cbm_pack import calc_smooth_inst_fire_rates_from_raster, \
    calc_inst_fire_rates_from

num_cells_test = 32 # to mimic pcs
num_trials_test = 500 # mimic num trials usual for isi 500
num_ts_test = 1300 # mimic pre + post + isi

num_exec = 1

def init_test_data(num_cells, num_trials, num_ts):
    return np.random.randint(
        0,
        high=2,
        size=(num_cells, num_trials, num_ts),
        dtype=np.uint8
    )

if __name__ == "__main__":
    total_exec_t = timeit.timeit(
        "calc_inst_fire_rates_from(test_data)",
        #"calc_smooth_inst_fire_rates_from_raster(test_data, kernel_type='gaussian')",
        setup="test_data = init_test_data(num_cells_test, num_trials_test, num_ts_test)",
        number=num_exec,
        globals=globals()
    )
    exec_t_per_call = total_exec_t / num_exec
    print(f"Total execution time: {total_exec_t}")
    print(f"Time per call: {exec_t_per_call}")
                      