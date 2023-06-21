#!/usr/bin/env python

import numpy as np

"""
    Description:
        
        allowing user too much freedom is difficult to implement and may be
        inefficient for subsequent data analysis. Thus, I will enforce the shape
        as: (num_cells, num_trials, num_ts)
"""

def reshape_raster(data: np.ndarray, num_cells: int, num_trials: int, num_ts: int) -> np.ndarray:
    try:
        assert data.size == num_cells * num_trials * num_ts
    except AssertionError:
        raise RuntimeError(f"cannot reshape an array of size {data.size} into an array with dimensions"
                             f"({num_cells}, {num_trials}, {num_ts})")
    else:
        transformed_data = data.reshape((num_ts * num_trials, num_cells))
        transformed_data = transformed_data.transpose()
        result = np.zeros((num_cells, num_trials, num_ts), dtype=data.dtype)
        for cell_id in np.arange(num_cells):
            trial_start = 0
            trial_end = num_ts
            for trial in np.arange(num_trials):
                result[cell_id,trial,:] = transformed_data[cell_id,trial_start:trial_end]
                trial_start += num_ts
                trial_end += num_ts
        return result

"""
    Description:

        the idea here is that we want to truncate the input array, summing
        over trials and timesteps. Let:
            raster == np.ndarray with shape (32, 1000, 1400)
            dims_tags == tuple ("cell", "trial", "time_step")

        then the returned np.ndarray will have dims (32, 1400), where
        the sum will occur over trials
"""
def calc_psth_from_raster(raster: np.ndarray, dims_tags) -> np.ndarray
    try:
        assert len(raster.shape) == 3
    except AssertionError
        raise ValueError(f"The expected number of dimensions was 3. Got {len(raster.shape)}")
    else:
        return np.sum(raster,1, dtype=np.uint32) # potential memory hog
"""
    Description:

        calculates the instantaneous firing rates at
        the time points t_i for each i in the range [0, N),
        where N is the number of spikes in the input.

        The function first determines the inst firing rates
        for the isis (here interspike intervals :)), and then
        re-aligns these rates with the indices of the spikes
        in the input spike train. The output array of 
        firing rates if padded left and right with zeroes if
        necessary.

        One way to make sense of this algorithm is to construct
        an arbitrarily chosen spike train of a short length, call 
        it n. This is equivalent to choosing a random binary string
        of length n. Then, list out how this data is transformed in
        subsequent lines of the code.

"""
# TODO: redo this function over multiple cells
def calc_inst_fire_rates_from(input_data: np.ndarray, data_type: str, num_trials=0: int) -> np.ndarray:
    if data_type == "psth":
        return (input_data * 1000) / num_trials
    elif data_type == "raster":
        run_time = input_data.shape[2]
        spk_times = np.nonzero(input_data)[0]
        aligned_inst_fire_rates = []

        if spk_times.size == 0:
            return np.zeros(run_time)
        else:
            isi = np.diff(spk_times)
            inst_fire_rates = 1 / isi * 1000
            input_data_proxy = input_data[spk_times[0]+1:spk_times[-1]]
            count = 0
            for i in np.arange(input_data_proxy.size):
                aligned_inst_fire_rates.append(inst_fire_rates[count])
                if input_data_proxy[i] == 1:
                    count += 1
            aligned_inst_fire_rates = np.array(aligned_inst_fire_rates)
            prepend = np.zeros(spk_times[0])
            aligned_inst_fire_rates = np.concatenate((prepend, aligned_inst_fire_rates))
            append = np.zeros(run_time - aligned_inst_fire_rates.size)
            aligned_inst_fire_rates = np.concatenate((aligned_inst_fire_rates, append))
            return aligned_inst_fire_rates
    else:
        raise ValueError(f"unknown data type '{data_type}'")

def calc_smooth_inst_fire_rates_from(input_data: np.ndarray, data_type: str, kernel="half_gaussian": str, num_trials=0: int) -> np.ndarray:
    try:
        inst_fr = calc_inst_fire_rates_from(input_data, data_type, num_trials=num_trials)
    except ValueError as v_err:
        raise ValueError(v_err)
    else:
        #TODO convolve with kernel
        if kernel == "gaussian":
            smooth_range = np.arange(FR_SMOOTH_MIN, FR_SMOOTH_MAX + 1, 1)
            smooth_gaussian = sts.norm.pdf(smooth_range, FR_SMOOTH_MU, FR_SMOOTH_SIGMA)
            smooth_cell_inst_fr_trials[trial, :] = np.convolve(cell_inst_fr_trials[trial, :], smooth_gaussian, 'same')
            pass
        elif kernel == "half_gaussian":
            pass
