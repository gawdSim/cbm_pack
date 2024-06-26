#!/usr/bin/env python

import numpy as np
import scipy.stats as sts

"""
    Description:
        
        allowing user too much freedom is difficult to implement and may be
        inefficient for subsequent data analysis. Thus, I will enforce the shape
        as: (num_cells, num_trials, num_ts)
"""


def reshape_raster(
    data: np.ndarray, num_cells: int, num_trials: int, num_ts: int
) -> np.ndarray:
    try:
        assert data.size == num_cells * num_trials * num_ts
    except AssertionError:
        raise RuntimeError(
            f"cannot reshape an array of size {data.size} into an array with dimensions"
            f"({num_cells}, {num_trials}, {num_ts})"
        )
    else:
        transformed_data = data.reshape((num_ts * num_trials, num_cells))
        transformed_data = transformed_data.transpose()
        result = np.zeros((num_cells, num_trials, num_ts), dtype=data.dtype)
        for cell_id in np.arange(num_cells):
            trial_start = 0
            trial_end = num_ts
            for trial in np.arange(num_trials):
                result[cell_id, trial, :] = transformed_data[
                    cell_id, trial_start:trial_end
                ]
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


def calc_psth_from_raster(raster: np.ndarray, dims_tags) -> np.ndarray:
    try:
        assert len(raster.shape) == 3
    except AssertionError:
        raise ValueError(
            f"The expected number of dimensions was 3. Got {len(raster.shape)}"
        )
    else:
        return np.sum(raster, 1, dtype=np.uint32)  # potential memory hog


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


def calc_inst_fire_rates_from_rast_1d(spike_train: np.ndarray) -> np.ndarray:
    try:
        assert len(spike_train.shape) == 1
    except AssertionError:
        raise ValueError(
            f"The expected number of dimensions was 1. Got {len(spike_train.shape)}"
        )
    else:
        run_time = spike_train.shape[0]
        spk_times = np.nonzero(spike_train)[0]

        if spk_times.size == 0:
            return np.zeros(run_time)
        else:
            isi = np.diff(spk_times)
            inst_fire_rates = 1 / isi * 1000
            spike_train_proxy = spike_train[spk_times[0] + 1 : spk_times[-1]]
            aligned_inst_fire_rates = np.zeros(spike_train_proxy.size)
            count = 0
            for i in np.arange(spike_train_proxy.size):
                aligned_inst_fire_rates[i] = inst_fire_rates[count]
                if spike_train_proxy[i] == 1:
                    count += 1
            prepend = np.zeros(spk_times[0])
            aligned_inst_fire_rates = np.concatenate((prepend, aligned_inst_fire_rates))
            append = np.zeros(run_time - aligned_inst_fire_rates.size)
            aligned_inst_fire_rates = np.concatenate((aligned_inst_fire_rates, append))
            return aligned_inst_fire_rates


"""
    Description:

        computes the instantaneous firing rates for all cells and trials within input_data.
        Expected input data can either be psth or raster data, and the respective shapes of
        either is (num_cells, num_ts) and (num_cells, num_trials, num_ts). An array of the 
        same shape is returned

        TODO: debug for psths, something is amiss >:(
"""


def calc_inst_fire_rates_from(
    input_data: np.ndarray, data_type: str = "raster", num_trials: int = 0
) -> np.ndarray:
    if data_type == "psth":
        try:
            assert num_trials > 0
        except AssertionError:
            raise ValueError(
                "Number of trials must be greater than zero for psth input data type"
            )
        else:
            return (input_data * 1000) / num_trials
    elif data_type == "raster":
        try:
            assert len(input_data.shape) == 3
        except AssertionError:
            raise ValueError(
                f"Expected input dimensions to be 3, got '{input_data.shape}'"
            )
        else:
            num_cells, num_trials, num_ts_per_trial = input_data.shape
            frs = np.zeros(input_data.shape)
            for cell_id in np.arange(num_cells):
                frs[cell_id] = calc_inst_fire_rates_from_rast_1d(
                    input_data[cell_id].reshape(num_trials * num_ts_per_trial)
                ).reshape(num_trials, num_ts_per_trial)
            return frs
    else:
        raise ValueError(f"Unknown data type '{data_type}'")


"""
    Description:
        Compute the smooth instantaneous firing rates from rasters ie raw spike data.
        This function computes the result w.r.t. individual cells, individual trials:
        the returned shape will be the same as the input, ie (num_cells, num_trials, num_ts_per_trial)
"""


def calc_smooth_inst_fire_rates_from_raster(
    input_data: np.ndarray,
    kernel_type: str = "half_gaussian",
    kernel_loc: float = 0.0,
    kernel_scale: float = 10.0,
    kernel_scale_mult: float = 8.0,
) -> np.ndarray:
    try:
        inst_fr = calc_inst_fire_rates_from(input_data)
    except ValueError as v_err:
        raise ValueError(v_err)
    else:
        if kernel_type == "gaussian":
            kernel = sts.norm.pdf(
                np.arange(
                    -kernel_scale_mult * kernel_scale,
                    kernel_scale_mult * kernel_scale + 1,
                    1,
                ),
                kernel_loc,
                kernel_scale,
            )
        elif kernel_type == "half_gaussian":
            kernel = sts.halfnorm.pdf(
                np.arange(kernel_scale_mult * kernel_scale + 1),
                kernel_loc,
                kernel_scale,
            )
        else:
            raise ValueError(
                f"Expected a kernel type of either 'gaussian' or 'half gaussian'. Got '{kernel_type}'"
            )
            return
        mode = "same" if kernel_type == "gaussian" else "full"
        convolved = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode), axis=2, arr=inst_fr
        )
        if mode == "full":
            smooth_inst_frs = convolved[:, :, : input_data.shape[2]]
        else:
            smooth_inst_frs = convolved
        return smooth_inst_frs


"""
    Description:
        Compute the smooth instantaneous firing rates from psths ie trial summed spikes
        for each cell, each time step. This function computes the result w.r.t.
        individual cells: the returned shape will be the same as the input, ie (num_cells, num_ts_per_trial)

    TODO: debug, not giving reasonable smooth frs for given psth data
"""


def calc_smooth_inst_fire_rates_from_psth(
    input_data: np.ndarray,
    num_trials: int,
    kernel_type: str = "half_gaussian",
    kernel_loc: float = 0.0,
    kernel_scale: float = 10.0,
    kernel_scale_mult: float = 8.0,
) -> np.ndarray:
    try:
        inst_fr = calc_inst_fire_rates_from(input_data, "psth", num_trials=num_trials)
    except ValueError as v_err:
        raise ValueError(v_err)
    else:
        if kernel_type == "gaussian":
            kernel = sts.norm.pdf(
                np.arange(
                    -kernel_scale_mult * kernel_scale,
                    kernel_scale_mult * kernel_scale + 1,
                    1,
                ),
                kernel_loc,
                kernel_scale,
            )
        elif kernel_type == "half_gaussian":
            kernel = sts.halfnorm.pdf(
                np.arange(kernel_scale_mult * kernel_scale + 1),
                kernel_loc,
                kernel_scale,
            )
        else:
            raise ValueError(
                f"Expected a kernel type of either 'gaussian' or 'half gaussian'. Got '{kernel_type}'"
            )
            return
        smooth_inst_frs = np.zeros(input_data.shape)
        mode = "same" if kernel_type == "gaussian" else "full"
        for cell_id in np.arange(input_data.shape[0]):
            convolved = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode), axis=0, arr=inst_fr[cell_id]
            )
            if mode == "full":
                smooth_inst_frs[cell_id] = convolved[:, : input_data.shape[1]]
            else:
                smooth_inst_frs[cell_id] = convolved
        return smooth_inst_frs


"""
    Description:
        Compute the smooth instantaneous firing rates from cell-averaged rasters
        for each trial, each time step. This function computes the result w.r.t.
        cell-averages: the returned shape will be the same as the input, ie (num_trials, num_ts_per_trial)
"""


def calc_smooth_mean_frs(
    mean_rasters: np.ndarray,
    kernel_type: str = "half_gaussian",
    kernel_loc: float = 0.0,
    kernel_scale: float = 10.0,
    kernel_scale_mult: float = 8.0,
) -> np.ndarray:
    if kernel_type == "gaussian":
        kernel = sts.norm.pdf(
            np.arange(
                -kernel_scale_mult * kernel_scale,
                kernel_scale_mult * kernel_scale + 1,
                1,
            ),
            kernel_loc,
            kernel_scale,
        )
    elif kernel_type == "half_gaussian":
        kernel = sts.halfnorm.pdf(
            np.arange(kernel_scale_mult * kernel_scale + 1), kernel_loc, kernel_scale
        )
    else:
        raise ValueError(
            f"Expected a kernel type of either 'gaussian' or 'half gaussian'. Got '{kernel_type}'"
        )
        return
    mode = "same" if kernel_type == "gaussian" else "full"
    convolved = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode), axis=0, arr=mean_rasters
    )
    if mode == "full":
        smooth_inst_frs = convolved[:, : input_data.shape[1]]
    else:
        smooth_inst_frs = convolved
    return smooth_inst_frs


"""
    Description:
        A generic version of the above three functions. May in the future be used in their place.
        Computes the smoothed data along the given axis of the input data array w.r.t. to the given
        kernel and kernel params.
"""


def calc_smooth_data_set_along_axis(
    data: np.ndarray,
    axis: int = 0,
    kernel_type: str = "half_gaussian",
    kernel_loc: float = 0.0,
    kernel_scale: float = 10.0,
    kernel_scale_mult: float = 8.0,
) -> np.ndarray:
    if kernel_type == "gaussian":
        kernel = sts.norm.pdf(
            np.arange(
                -kernel_scale_mult * kernel_scale,
                kernel_scale_mult * kernel_scale + 1,
                1,
            ),
            kernel_loc,
            kernel_scale,
        )
    elif kernel_type == "half_gaussian":
        kernel = sts.halfnorm.pdf(
            np.arange(kernel_scale_mult * kernel_scale + 1), kernel_loc, kernel_scale
        )
    else:
        raise ValueError(
            f"Expected a kernel type of either 'gaussian' or 'half gaussian'. Got '{kernel_type}'"
        )
        return
    mode = "same" if kernel_type == "gaussian" else "full"
    convolved = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode), axis=axis, arr=data
    )
    if mode == "full":
        smooth_inst_frs = convolved[:, : input_data.shape[1]]
    else:
        smooth_inst_frs = convolved
    return smooth_inst_frs
