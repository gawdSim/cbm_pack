#!/usr/bin/env python

import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error
from .transform import calc_inst_fire_rates_from
from .transform import calc_smooth_mean_frs
from .transform import calc_smooth_inst_fire_rates_from_raster

MS_PER_TIME_STEP = 1

"""
    Description:

        computes the cr amplitudes (on trials that they appear in) as a function
        of time within the trial. The returned array shape is (num_trials, num_ts_per_trial)
"""
def pcs_to_crs(pc_rasters: np.ndarray, \
    pre_cs_collect: int, \
    post_cs_collect: int, \
    isi: int) -> np.ndarray:
    num_cells, num_trials, num_ts_per_trial = pc_rasters.shape
    base_interval_low = int(0.25 * pre_cs_collect)
    base_interval_high = int(0.75 * pre_cs_collect)
    crs = np.zeros((num_trials, num_ts_per_trial))

    smooth_inst_frs = calc_smooth_inst_fire_rates_from_raster(pc_rasters, kernel_type="gaussian")
    mean_smooth_inst_frs = np.mean(smooth_inst_frs, axis=0)
    amp_ests = np.zeros(num_trials)
    for trial in np.arange(num_trials):
        # get base rate in middle of pre-cs period as convolving depresses the tails of the interval
        response_onset = 0.8 * np.mean(mean_smooth_inst_frs[trial, base_interval_low:base_interval_high])
        amp_est = response_onset - np.min(mean_smooth_inst_frs[trial, pre_cs_collect:pre_cs_collect+isi])
        if amp_est > 5:
            crs[trial, :] = response_onset - mean_smooth_inst_frs[trial, :]
            amp_ests[trial] = np.max(crs[trial, :])
    norm = np.max(amp_ests)
    crs = crs / norm
    crs *= 6.0
    crs[crs < 0.01] = 0.0
    crs[:, :int(0.05 * pre_cs_collect)] = 0.0
    crs[:, int(-0.05 * post_cs_collect):] = 0.0
    return crs

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

def nc_to_cr_mike(nc_rasters: np.ndarray) -> np.ndarray:
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
        g_exc[:, ts] = exc_sum * g_exc_inc + g_exc[:, ts-1] * g_exc_dec

    g_exc[g_exc < 0.] = 0. # get rid of negative values
    g_exc = 5.0 * np.power(g_exc, 3) # make the shape look more like a CR
    g_exc[g_exc < 0.02] = 0. # threshold small squigglies

    v_m = np.zeros((num_trials, num_ts_per_trial), dtype=np.single)
    v_m[:, 0] = e_leak
    for ts in np.arange(1, num_ts_per_trial):
        v_m[:, ts] = v_m[:, ts-1] \
                   + g_leak * (-v_m[:, ts-1]) \
                   + g_exc[:, ts] * (80 - v_m[:, ts-1])
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
            g_exc[:, ts] = exc_sum * g_exc_inc + g_exc[:, ts-1] * g_exc_dec
            v_m[:, ts] = v_m[:, ts-1] \
                       + g_leak * (e_leak - v_m[:, ts-1]) \
                       - g_exc[:, ts] * v_m[:, ts-1]
    return v_m

"""
    Description:

        Computes...things
"""
def slope(x):
    return np.array([x[2]-x[0],x[1]-x[0]])

def calc_rn_thresh(pc_onset_times: np.ndarray, \
        nc_rasters: np.ndarray, \
        pre_cs_collect: int, \
        isi) -> float:
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
        trial_cutoff_high = 450 # assuming all other trials == 500

    rn_thresh = (int_min + int_max) / 2
    rn_vms = rn_integrator_gelson(nc_rasters)
    rn_onset_times = calc_cr_onsets_from_rn(rn_vms, pre_cs_collect, isi, rn_thresh)
    quote_un_quote_true_mean = np.mean(pc_onset_times[trial_cutoff_low:trial_cutoff_high])
    loss = np.abs(np.mean(rn_onset_times[trial_cutoff_low:trial_cutoff_high]) - quote_un_quote_true_mean)
    loss_cutoff = 0.05
    i = 0
    while loss > loss_cutoff:
        print(f"iteration: {i}, loss: {loss}, rn_thresh: {rn_thresh}")
        if np.mean(rn_onset_times[trial_cutoff_low:trial_cutoff_high]) > quote_un_quote_true_mean:
            int_max = rn_thresh
        else:
            int_min = rn_thresh
        rn_thresh = (int_min + int_max) / 2
        rn_onset_times = calc_cr_onsets_from_rn(rn_vms, pre_cs_collect, isi, rn_thresh)
        loss = np.abs(np.mean(rn_onset_times[trial_cutoff_low:trial_cutoff_high]) - quote_un_quote_true_mean)
        i += 1
    return rn_thresh


def nc_to_cr_gelson(nc_rasters: np.ndarray) -> np.ndarray:
    v_m = rn_integrator_gelson(nc_rasters)
# PConsetTimes = PConsetTime_calc(ISI,meanFiringPC_df)
#     PCmu,PCsigma,PConset_pdf = onset_pdf(PConsetTimes)
# 
#     print('Calculating Red Nuclei Threshold')
#     RN_thres = RN_thres_calc(ISI,DCN_RN_df,PConsetTimes)
#     
#     norm_factor = np.abs((DCN_RN_df-RN_thres).max().max())
#     CR = 6 * (DCN_RN_df-RN_thres)/norm_factor
#     CR[CR<0] = 0

"""
    Description:

        want to return an array of (possibly) length (num_trials)
        where the value at a given trial number is the time-point
        at which the cr is initiated
"""
def calc_cr_onsets_from_pc(
        pc_rasters: np.ndarray, \
        pre_cs_collect: int, \
        isi: int) -> np.ndarray:
    num_cells, num_trials, num_ts_per_trial = pc_rasters.shape
    onset_times = -10 * np.ones(num_trials)
    base_interval_low = int(0.25 * pre_cs_collect)
    base_interval_high = int(0.75 * pre_cs_collect)

    inst_frs = calc_inst_fire_rates_from(pc_rasters)
    mean_inst_frs = np.mean(inst_frs, axis=0)
    smooth_mean_inst_frs = calc_smooth_mean_frs(mean_inst_frs, kernel_type="gaussian")
    for trial in np.arange(num_trials):
        # get base rate in middle of pre-cs period as convolving depresses the tails of the interval
        base_pc_rate      = np.mean(smooth_mean_inst_frs[trial, base_interval_low:base_interval_high])
        min_cs_pc_rate    = np.min(smooth_mean_inst_frs[trial, pre_cs_collect:pre_cs_collect+isi])
        response_onset    = base_pc_rate * 0.8 # criterion is 80% of base rate
        amp_est           = response_onset - min_cs_pc_rate
        if amp_est > 5:
            for ts in np.arange(pre_cs_collect, pre_cs_collect + isi):
                if smooth_mean_inst_frs[trial, ts] > response_onset \
                    and smooth_mean_inst_frs[trial, ts+1] <= response_onset:
                    onset_times[trial] = ts - pre_cs_collect
                    break
    return onset_times

"""
    Description:

        want to return an array of (possibly) length (num_trials)
        where the value at a given trial number is the time-point
        at which the cr is initiated
"""
def calc_cr_onsets_from_rn(rn_vms: np.ndarray, \
    pre_cs_collect: int, \
    isi: int, \
    rn_thresh: float) -> np.ndarray:
    num_trials, num_ts_per_trial = rn_vms.shape
    onset_times = np.zeros(num_trials)
    for trial in np.arange(num_trials):
        max_rn_vm = np.max(rn_vms[trial,pre_cs_collect:pre_cs_collect+isi])
        amp_est = max_rn_vm - rn_thresh
        if amp_est > 1: # this is a free param
            for ts in np.arange(pre_cs_collect, pre_cs_collect + isi):
                if rn_vms[trial, ts] < rn_thresh and rn_vms[trial, ts+1] >= rn_thresh:
                    onset_times[trial] = ts - pre_cs_collect
                    break
    return onset_times

"""
    Description:

        want to return an array of (possibly) length (num_trials)
        where the value at a given trial number is the probability
        of a cr, as measured by... well if we're reading off a cr, 
        how do we determine this? should we just return a single number,
        which is the fraction of number of crs found over total number
        of trials? check back into this
"""
def calc_cr_prob(dcn_rasters: np.ndarray, criterion: str) -> float:
    pass

"""
    Description:

        we might want a single number, in the case where we sum over dcn output
        into a single RN cell...check back in on this
"""
def calc_trials_to_criterion(dcn_rasters: np.ndarray, criterion: str) -> int:
    pass
