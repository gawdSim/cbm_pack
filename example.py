#!/usr/bin/env python

import cbm_pack

NUM_CELLS = 32
NUM_TRIALS = 1000
PRE_CS_COLLECT_TS = 400
POST_CS_COLLECT_TS = 400
ISI = 1000
NUM_COLLECT_TS = PRE_CS_COLLECT_TS + ISI + POST_CS_COLLECT_TS
INPUT_FILE = "example.pcr"

import numpy as np

raw_raster_data = cbm_pack.np_arr_from_file(INPUT_FILE, np.uint8)
# returned shape will be (NUM_CELLS, NUM_TRIALS, NUM_COLLECT_TS)
raster_data = reshape_raster( \
        raw_raster_data, \
        NUM_CELLS, \
        NUM_TRIALS, \
        NUM_COLLECT_TS)

# now we have the raw firing rates. shape is (NUM_CELLS, NUM_TRIALS, NUM_COLLECT_TS)
raw_firing_rates = calc_inst_fire_rates_from(raster_data, "raster")

# we can also calculate the mean firing rates over a given tagged data type

# this gives you the average firing rate, avg'd across cells. returned shape is
# (NUM_TRIALS, NUM_COLLECT_TS). measures the cell population avg at a given trial and ts
cell_mean_firing_rates = calc_mean_inst_fire_rates_from(raster_data, "raster", "cells")

# this gives you the average firing rate, avg'd across trials. returned shape is
# (NUM_CELLS, NUM_COLLECT_TS). measures the avg firing rate of a given cell at a time pt across trials
across_trial_mean_firing_rates = calc_mean_inst_fire_rates_from(raster_data, "raster", "trials")

# this gives you the average firing rate, avg'd across ts in all trials. returned shape is
# (NUM_CELLS, NUM_TRIALS). measures the avg firing rate of a given cell on a given trial
in_trial_mean_firing_rates = calc_mean_inst_fire_rates_from(raster_data, "raster", "time_steps")

#TODO: do the same for variance and standard deviation

# now we have the smooth firing rates for all the cells,
# with final dimensions (NUM_CELLS, NUM_COLLECT_TS)
smooth_firing_rates = calc_smooth_inst_fire_rates_from( \
        raster_data, \
        "raster", \
        "half_gaussian")
