import numpy as np

"""
    Description:

        the idea here is that we want to reshape, but we want
        to label our axes with meaningful tags. The tags
        guide the implementation as well as the end-user in 
        subsequent usage of the returned array

        Example: let
            
            data == np.ndarray with shape (32 * 1000 * 1400)
            dims == tuple (1000, 1400, 32)
            dims_tags == tuple ("trial", "time_step", "cell")

        then the returned np.ndarray will have shape (1000, 1400, 32) s.t.
        if returned array is called 'reshaped_data', then the quantity

                            reshaped_data[500, 0, 16]

        will refer to data at the 500th trial, at time step 0, for cell 16.

        Mainly here so that the end-user doesn't have to think about the format
        of the incoming data. they can just write:

            data = np_arr_from_file("foo", np.ubyte)
            formatted_data = reshape_as(data, \
                    (NUM_TRIALS, NUM_TS, NUM_CELLS), \
                    ("trial", "time_step", "cell"))
        
        and be confident moving forward that they are slicing correctly
"""

def reshape_as(data: np.ndarray, dims: tuple, dims_tags: tuple) -> np.ndarray:
    # TODO: implement
    pass

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
    # TODO: implement
    pass

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
def calc_inst_fire_rates_from(spike_trains: np.ndarray, data_type: str) -> np.ndarray:
    if data_type == "psth":
        return (spike_trains * 1000) / num_trials
    elif data_type == "raster":
        run_time = spike_trains.size
        spk_times = np.nonzero(spike_trains)[0]
        aligned_inst_fire_rates = []

        if spk_times.size == 0:
            return np.zeros(run_time)
        else:
            isi = np.diff(spk_times)
            inst_fire_rates = 1 / isi * 1000
            spike_trains_proxy = spike_trains[spk_times[0]+1:spk_times[-1]]
            count = 0
            for i in np.arange(spike_trains_proxy.size):
                aligned_inst_fire_rates.append(inst_fire_rates[count]) 
                if spike_trains_proxy[i] == 1:
                    count += 1
            aligned_inst_fire_rates = np.array(aligned_inst_fire_rates)
            prepend = np.zeros(spk_times[0])
            aligned_inst_fire_rates = np.concatenate((prepend, aligned_inst_fire_rates))
            append = np.zeros(run_time - aligned_inst_fire_rates.size)
            aligned_inst_fire_rates = np.concatenate((aligned_inst_fire_rates, append))
            return aligned_inst_fire_rates
    else:
        raise ValueError(f"unknown data type '{data_type}'")

def calc_smooth_inst_fire_rates_from(spike_trains: np.ndarray, data_type: str, kernel: str) -> np.ndarray:
    try:
        inst_fr = calc_inst_fire_rates_from(spike_trains, data_type)
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
