#!/usr/bin/env python

import numpy as np
from .transform import calc_inst_fire_rates_from
from .transform import calc_smooth_mean_frs
from .transform import calc_smooth_inst_fire_rates_from_raster

MS_PER_TIME_STEP = 1

"""
    Description:

        computes the cr amplitudes (on trials that they appear in) as a function
        of time within the trial. The returned array shape is (num_trials, num_ts_per_trial)
        
        TODO: make base interval low and high kwargs
"""


def pcs_to_crs(
    pc_rasters: np.ndarray, pre_cs_collect: int, post_cs_collect: int, isi: int
) -> np.ndarray:
    num_cells, num_trials, num_ts_per_trial = pc_rasters.shape
    base_interval_low = int(0.25 * pre_cs_collect)
    base_interval_high = int(0.75 * pre_cs_collect)
    cr_entrap_win = 10  # time step window to average over to detect cr
    crs = np.zeros((num_trials, num_ts_per_trial), dtype=np.single)

    smooth_inst_frs = calc_smooth_inst_fire_rates_from_raster(
        pc_rasters, kernel_type="gaussian"
    )
    mean_smooth_inst_frs = np.mean(smooth_inst_frs, axis=0)
    amp_ests = np.zeros(num_trials, dtype=np.single)
    onset_times = np.zeros(num_trials, dtype=np.uint32)
    for trial in np.arange(num_trials):
        cr_exists = False
        # get base rate in middle of pre-cs period as convolving depresses the tails of the interval
        response_onset = 0.8 * np.mean(
            mean_smooth_inst_frs[trial, base_interval_low:base_interval_high]
        )
        min_rate = np.min(
            mean_smooth_inst_frs[trial, pre_cs_collect : pre_cs_collect + isi]
        )
        amp_est = response_onset - min_rate
        for ts in np.arange(pre_cs_collect, pre_cs_collect + isi, 1):
            if (
                amp_est > 5
                and mean_smooth_inst_frs[trial, ts] > response_onset
                and np.mean(mean_smooth_inst_frs[trial, ts : ts + cr_entrap_win])
                <= response_onset
            ):
                onset_times[trial] = np.uint32(ts - pre_cs_collect + cr_entrap_win / 2)
                cr_exists = True
                break

        if cr_exists:
            crs[trial, :] = response_onset - mean_smooth_inst_frs[trial, :]
            amp_ests[trial] = np.max(crs[trial, :])
        cr_exists = False

    norm = np.max(amp_ests)
    crs = crs / norm
    crs *= 6.0
    crs[crs < 0.01] = 0.0
    crs[:, : int(0.05 * pre_cs_collect)] = 0.0
    crs[:, int(-0.05 * post_cs_collect) :] = 0.0
    return crs, amp_ests, onset_times


"""
    Description:

        computes the membrane current of a simulated red nucleus (RN)
        cell which receives input from all of our deep nucleus cells,
        then transforms that current into an amplitude in the range [0, 6]

        NOTE: we initialize the first time step of EVERY trial to 
        the same value for both the excitatory conductance and
        the membrane potential. This may not be what we want.
        we may want to keep a running record of each for all ts
        in every trial. this is in conflict with the idea that we
        will collect rasters at only probe trials rather than all
        trials. May want to just collect rasters at all trials and
        then run them through this alg, updating across all trials
"""


def ncs_to_cr_mike(nc_rasters: np.ndarray) -> np.ndarray:
    g_exc_tau = 15.0
    e_leak = 0.0
    g_leak = 0.025 / (6 - MS_PER_TIME_STEP)
    g_exc_inc = 0.012
    g_exc_dec = np.exp(-MS_PER_TIME_STEP / g_exc_tau)
    num_cells, num_trials, num_ts_per_trial = nc_rasters.shape
    g_exc = np.zeros((num_trials, num_ts_per_trial), dtype=np.single)
    for ts in np.arange(1, num_ts_per_trial):
        exc_sum = 0
        for cell_id in np.arange(num_cells):
            exc_sum += nc_rasters[cell_id, :, ts]
        g_exc[:, ts] = exc_sum * g_exc_inc + g_exc[:, ts - 1] * g_exc_dec

    g_exc[g_exc < 0.0] = 0.0  # get rid of negative values
    g_exc = 5.0 * np.power(g_exc, 3)  # make the shape look more like a CR
    g_exc[g_exc < 0.02] = 0.0  # threshold small squigglies

    v_m = np.zeros((num_trials, num_ts_per_trial), dtype=np.single)
    v_m[:, 0] = e_leak
    for ts in np.arange(1, num_ts_per_trial):
        v_m[:, ts] = (
            v_m[:, ts - 1]
            + g_leak * (-v_m[:, ts - 1])
            + g_exc[:, ts] * (80 - v_m[:, ts - 1])
        )
    return v_m * 0.075


"""
    Description:

        Calculates the voltage of a simulated red nucleus downstream
        from the nucleus cells of the cbm_sim, summing over the 
        activity of all the nucleus cell inputs

        Notice that I am parallelizing over trials: I am assuming
        the rasters come from probe trials interspersed within a 
        training run. Thus, we cannot keep a running sum of the current
        membrane potential as we don't have all of the nc spike information
        to do so.
"""


def rn_integrator_gelson(nc_rasters: np.ndarray) -> np.ndarray:
    g_exc_tau = 15.0
    e_leak = -60.0
    g_leak = 0.01
    g_exc_inc = 0.005
    g_exc_dec = np.exp(-MS_PER_TIME_STEP / g_exc_tau)
    num_cells, num_trials, num_ts_per_trial = nc_rasters.shape
    g_exc = np.zeros((num_trials, num_ts_per_trial), dtype=np.single)
    v_m = np.zeros((num_trials, num_ts_per_trial))
    v_m[:, 0] = e_leak
    for trial in np.arange(num_trials):
        for ts in np.arange(1, num_ts_per_trial):
            exc_sum = 0
            for cell_id in np.arange(num_cells):
                exc_sum += nc_rasters[cell_id, :, ts]
            g_exc[:, ts] = exc_sum * g_exc_inc + g_exc[:, ts - 1] * g_exc_dec
            v_m[:, ts] = (
                v_m[:, ts - 1]
                + g_leak * (e_leak - v_m[:, ts - 1])
                - g_exc[:, ts] * v_m[:, ts - 1]
            )
    return v_m


"""
    Description

        Computes a red nucleus threshold that gives cr onset times whose mean is 
        equal to the mean obtained from the cr onset times computed from pc cells
        up to an error of loss_cutoff.

        NOTE: it is assumed that for rasters obtained from ISIs less than 1000 the 
        number of trials ran was 250. For greater ISIs, it is assumed that the number
        of trials ran was 500

        TODO: make loss_cutoff, trial_cutoff_high an input-able parameter
"""


def calc_rn_thresh(
    pc_onset_times: np.ndarray, nc_rasters: np.ndarray, pre_cs_collect: int, isi
) -> float:
    int_max = -10.0
    int_min = -30.0
    # play with these cut offs
    onset_time_cutoff = int(0.1 * isi)
    _, num_trials, num_ts_per_trial = nc_rasters.shape
    # play with this number, esp if we collect all trials
    trial_cutoff_low = 50
    if num_trials <= 250:
        trial_cutoff_high = 200
    else:
        trial_cutoff_high = 450  # assuming all other trials == 500

    rn_thresh = (int_min + int_max) / 2
    rn_vms = rn_integrator_gelson(nc_rasters)
    rn_onset_times = calc_cr_onsets_from_rn(rn_vms, pre_cs_collect, isi, rn_thresh)
    quote_un_quote_true_mean = np.mean(
        pc_onset_times[trial_cutoff_low:trial_cutoff_high]
    )
    loss = np.abs(
        np.mean(rn_onset_times[trial_cutoff_low:trial_cutoff_high])
        - quote_un_quote_true_mean
    )
    loss_cutoff = 0.05
    i = 0
    while loss > loss_cutoff:
        print(f"iteration: {i}, loss: {loss}, rn_thresh: {rn_thresh}")
        if (
            np.mean(rn_onset_times[trial_cutoff_low:trial_cutoff_high])
            > quote_un_quote_true_mean
        ):
            int_max = rn_thresh
        else:
            int_min = rn_thresh
        rn_thresh = (int_min + int_max) / 2
        rn_onset_times = calc_cr_onsets_from_rn(rn_vms, pre_cs_collect, isi, rn_thresh)
        loss = np.abs(
            np.mean(rn_onset_times[trial_cutoff_low:trial_cutoff_high])
            - quote_un_quote_true_mean
        )
        i += 1
    return rn_thresh


"""
    Description:

        calculates the CRs from ncs by computing the membrane potential of a hypothetical excitatory
        red nucleus cell. Uses the CR onset times computed from the purkinje cells in order to produce
        a red nucleus threshold that produces CR onset times whose mean is "close" to that obtained from
        the purkinje cells. The red nucleus membrane potential is then modulated by obtaining
        values equal to the deviation of the membrane potential from the threshold, normalizing, scaling by
        the maximal eyelid closure in behavioural studies, and then cutting off values below a certain threshold

        NOTE: this function is dependent upon the tuning of the integrated red nucleus membrane potential and 
        hyper-parameters for obtaining the rn threshold. These params and hyper-params may also have an ISI dependence,
        so care must be taken to obtain params that give *reasonable* results

        TODO: make the cutoff threshold and max amplitude scaling kwargs
"""


def ncs_to_cr_gelson(
    pc_rasters: np.ndarray, nc_rasters: np.ndarray, pre_cs_collect: int, isi: int
) -> np.ndarray:
    pc_onsets = calc_cr_onsets_from_pc(pc_rasters, pre_cs_collect, isi)
    rn_thresh = calc_rn_thresh(pc_onsets, nc_rasters, pre_cs_collect, isi)
    rn_v_m = rn_integrator_gelson(nc_rasters)
    norm = np.abs(
        rn_v_m[:, pre_cs_collect : pre_cs_collect + isi]
        - rn_thresh[:, pre_cs_collect : pre_cs_collect + isi]
    ).max()
    crs = (rn_v_m - rn_thresh) / norm
    crs *= 6.0
    crs[crs < 0.01] = 0.0
    return crs


"""
    Description:

        calculates the CRs directly from the ncs by calculating
        the per cell smooth fr and the taking the mean of that,
        then normalizing, scaling, and applying a threshold for values
        very close to zero

        NOTE: Updated 07/05/2023 by including threshold of 10Hz above base rate
        for CRs. From Medina et al 2000 (Timing Mechanisms in the Cerebellum)
"""


def ncs_to_cr_sean(
    nc_rasters: np.ndarray, pre_cs_collect: int, post_cs_collect: int, isi: int
) -> np.ndarray:
    num_cells, num_trials, num_ts_per_trial = nc_rasters.shape
    base_interval_low = int(0.25 * pre_cs_collect)
    base_interval_high = int(0.75 * pre_cs_collect)
    inst_frs = calc_inst_fire_rates_from(nc_rasters)
    mean_inst_frs = np.mean(inst_frs, axis=0)
    smooth_mean_inst_frs = calc_smooth_mean_frs(mean_inst_frs, kernel_type="gaussian")
    crs = np.zeros((num_trials, num_ts_per_trial))
    amp_ests = np.zeros(num_trials)
    for trial in np.arange(num_trials):
        trial_max_fr = np.max(
            smooth_mean_inst_frs[trial, pre_cs_collect : pre_cs_collect + isi]
        )
        trial_base_fr = np.mean(
            smooth_mean_inst_frs[trial, base_interval_low:base_interval_high]
        )
        response_criterion = trial_base_fr + 10
        amp_est = trial_max_fr - trial_base_fr
        if amp_est > 0:
            crs[trial, :] = smooth_mean_inst_frs[trial, :] - response_criterion
            amp_ests[trial] = amp_est
    crs[crs < 0.0] = 0.0
    norm = np.max(amp_ests)
    crs = crs / norm
    crs *= 6.0
    # some tail action: might be a result of smoothing: look into
    crs[:, : int(0.05 * pre_cs_collect)] = 0.0
    crs[:, int(-0.05 * post_cs_collect) :] = 0.0
    return crs


"""
    Description:

        calculates the CR onset times obtained from purkinje cells. uses the following criterion:

            1) a CR is given only if the maximal deviation from the base firing rate is greater than 5
            2) the point in simulated time when the pc firing rate falls below 80% of baseline
"""


def calc_cr_onsets_from_pc(
    pc_rasters: np.ndarray, pre_cs_collect: int, isi: int
) -> np.ndarray:
    num_cells, num_trials, num_ts_per_trial = pc_rasters.shape
    onset_times = -10 * np.ones(num_trials)
    base_interval_low = int(0.25 * pre_cs_collect)
    base_interval_high = int(0.75 * pre_cs_collect)

    inst_frs = calc_inst_fire_rates_from(pc_rasters)
    mean_inst_frs = np.mean(inst_frs, axis=0)
    smooth_mean_inst_frs = calc_smooth_mean_frs(mean_inst_frs, kernel_type="gaussian")
    for trial in np.arange(num_trials):
        # get base rate in middle of pre-cs period as convolving depresses the tails of the interval
        base_pc_rate = np.mean(
            smooth_mean_inst_frs[trial, base_interval_low:base_interval_high]
        )
        min_cs_pc_rate = np.min(
            smooth_mean_inst_frs[trial, pre_cs_collect : pre_cs_collect + isi]
        )
        response_onset = base_pc_rate * 0.8  # criterion is 80% of base rate
        amp_est = response_onset - min_cs_pc_rate
        if amp_est > 5:
            for ts in np.arange(pre_cs_collect, pre_cs_collect + isi):
                if (
                    smooth_mean_inst_frs[trial, ts] > response_onset
                    and smooth_mean_inst_frs[trial, ts + 1] <= response_onset
                ):
                    onset_times[trial] = ts - pre_cs_collect
                    break
    return onset_times


"""
    Description:
        calculates the cr onset times directly from nucleus cell rasters by computing
        cell-averaged, smoothed firing rates then checking whether these frs (as a function of
        trial-time) ever reach criterion, and if so, keeping track of the time step
"""


def calc_cr_onsets_from_nc(
    nc_rasters: np.ndarray, pre_cs_collect: int, isi: int
) -> np.ndarray:
    num_cells, num_trials, num_ts_per_trial = nc_rasters.shape
    onset_times = -10 * np.ones(num_trials)
    base_interval_low = int(0.25 * pre_cs_collect)
    base_interval_high = int(0.75 * pre_cs_collect)

    inst_frs = calc_inst_fire_rates_from(nc_rasters)
    mean_inst_frs = np.mean(inst_frs, axis=0)
    smooth_mean_inst_frs = calc_smooth_mean_frs(mean_inst_frs, kernel_type="gaussian")
    for trial in np.arange(num_trials):
        # get base rate in middle of pre-cs period as convolving depresses the tails of the interval
        trial_max_fr = np.max(
            smooth_mean_inst_frs[trial, pre_cs_collect : pre_cs_collect + isi]
        )
        trial_base_fr = np.mean(
            smooth_mean_inst_frs[trial, base_interval_low:base_interval_high]
        )
        response_criterion = trial_base_fr + 10
        if trial_max_fr > response_criterion:
            for ts in np.arange(pre_cs_collect, pre_cs_collect + isi):
                if (
                    smooth_mean_inst_frs[trial, ts] < response_criterion
                    and smooth_mean_inst_frs[trial, ts + 1] >= response_criterion
                ):
                    onset_times[trial] = ts - pre_cs_collect
                    break
    return onset_times


"""
    Description:
        calculates the CR onset times obtained from red nucleus membrane potential. uses the following criterion:

            1) a CR is given only if the maximal deviation from the input threshold is greater than 1
            2) the point in simulated time when the red nucleus membrane potential passes threshold

"""


def calc_cr_onsets_from_rn(
    rn_vms: np.ndarray, pre_cs_collect: int, isi: int, rn_thresh: float
) -> np.ndarray:
    num_trials, num_ts_per_trial = rn_vms.shape
    onset_times = np.zeros(num_trials)
    for trial in np.arange(num_trials):
        max_rn_vm = np.max(rn_vms[trial, pre_cs_collect : pre_cs_collect + isi])
        amp_est = max_rn_vm - rn_thresh
        if amp_est > 1:  # this is a free param
            for ts in np.arange(pre_cs_collect, pre_cs_collect + isi):
                if rn_vms[trial, ts] < rn_thresh and rn_vms[trial, ts + 1] >= rn_thresh:
                    onset_times[trial] = ts - pre_cs_collect
                    break
    return onset_times


"""
    Description:
        computes the probability of a CR by averaging over an array indicating the presence
        of a CR in a trial or not. This function computes the presence of a CR from the
        activity of DCN cells (straight from their cell rasters)
"""


def calc_cr_probs_from_pc(
    pc_rasters: np.ndarray, pre_cs_collect: int, isi: int, num_avg_over: int
) -> np.ndarray:
    num_cells, num_trials, num_ts_per_trial = pc_rasters.shape
    assert num_avg_over < num_trials
    cr_bools = np.zeros(num_trials, dtype=np.single)
    base_interval_low = int(0.25 * pre_cs_collect)
    base_interval_high = int(0.75 * pre_cs_collect)

    inst_frs = calc_inst_fire_rates_from(pc_rasters)
    mean_inst_frs = np.mean(inst_frs, axis=0)
    smooth_mean_inst_frs = calc_smooth_mean_frs(mean_inst_frs, kernel_type="gaussian")
    for trial in np.arange(num_trials):
        trial_base_fr = np.mean(
            smooth_mean_inst_frs[trial, base_interval_low:base_interval_high]
        )
        trial_min_fr = np.min(
            smooth_mean_inst_frs[trial, pre_cs_collect : pre_cs_collect + isi]
        )
        response_criterion = trial_base_fr * 0.8  # criterion is 80% of base rate
        amp_est = response_criterion - trial_min_fr
        if amp_est > 5:
            cr_bools[trial] = 1.0
    cr_probs = np.zeros(num_trials // num_avg_over, dtype=np.single)
    avg_start = 0
    for id, trial in np.ndenumerate(np.arange(avg_start, num_trials, num_avg_over)):
        cr_probs[id] = np.mean(cr_bools[trial : trial + num_avg_over])
    return cr_probs


"""
    Description:
        computes the probability of a CR by averaging over an array indicating the presence
        of a CR in a trial or not. This function computes the presence of a CR from the
        activity of DCN cells (straight from their cell rasters)
"""


def calc_cr_probs_from_nc(
    nc_rasters: np.ndarray, pre_cs_collect: int, isi: int, num_avg_over: int
) -> np.ndarray:
    num_cells, num_trials, num_ts_per_trial = nc_rasters.shape
    assert num_avg_over < num_trials
    cr_bools = np.zeros(num_trials, dtype=np.single)
    base_interval_low = int(0.25 * pre_cs_collect)
    base_interval_high = int(0.75 * pre_cs_collect)

    inst_frs = calc_inst_fire_rates_from(nc_rasters)
    mean_inst_frs = np.mean(inst_frs, axis=0)
    smooth_mean_inst_frs = calc_smooth_mean_frs(mean_inst_frs, kernel_type="gaussian")
    for trial in np.arange(num_trials):
        # get base rate in middle of pre-cs period as convolving depresses the tails of the interval
        trial_max_fr = np.max(
            smooth_mean_inst_frs[trial, pre_cs_collect : pre_cs_collect + isi]
        )
        trial_base_fr = np.mean(
            smooth_mean_inst_frs[trial, base_interval_low:base_interval_high]
        )
        print(f"trial {trial} max fr: {trial_max_fr}")
        response_criterion = trial_base_fr + 10.0
        print(f"trial {trial} criterion: {response_criterion}")
        if trial_max_fr > response_criterion:
            cr_bools[trial] = 1.0
    cr_probs = np.zeros(num_trials // num_avg_over, dtype=np.single)
    avg_start = 0
    for id, trial in np.ndenumerate(np.arange(avg_start, num_trials, num_avg_over)):
        cr_probs[id] = np.mean(cr_bools[trial : trial + num_avg_over])
    return cr_probs


"""
    Description:

        we might want a single number, in the case where we sum over dcn output
        into a single RN cell...check back in on this
"""


def calc_trials_to_criterion(dcn_rasters: np.ndarray, criterion: str) -> int:
    pass
