#!/usr/bin/env python

import numpy as np

MS_PER_TIME_STEP = 1

"""
    Description:

        computes the membrane current of a simulated red nucleus (RN)
        cell which receives input from all of our deep nucleus cells

        NOTE: we initialize the first time step of EVERY trial to 
        the same value for both the excitatory conductance and
        the membrane potential. This may not be what we want.
        we may want to keep a running record of each for all ts
        in every trial. this is in conflict with the idea that we
        will collect rasters at only probe trials rather than all
        trials. May want to just collect rasters at all trials and
        then run them through this alg, updating across all trials
"""

def rn_integrator(nc_rasters: np.ndarray) -> np.ndarray:
    g_exc_tau = 15.0
    e_leak = -60.0
    g_leak = 0.01
    g_exc_inc = 0.005
    g_exc_dec = np.exp(-MS_PER_TIME_STEP / g_exc_tau)
    num_cells, num_trials, num_ts_per_trial = nc_rasters.shape
    g_exc_sums = np.sum(nc_rasters, axis=0)
    g_exc = np.zeros(num_trials, num_ts_per_trials, dtype=np.single)
    v_m = np.zeros(num_trials, num_ts_per_trials, dtype=np.single)
    g_exc[:, 0] = g_exc_sums[:, 0] * g_exc_inc
    v_m[:, 0] = e_leak
    for ts in np.arange(num_ts_per_trial):
        g_exc[:, ts] = g_exc_sums[:, ts] * g_exc_inc + g_exc[:, ts-1] * g_exc_dec
        v_m[:, ts] = v_m[:, ts-1] \
                   + g_leak * (e_leak - v_m[:, ts-1]) \
                   + g_exc[:, ts] * v_m[:, ts-1]
    return v_m

"""
    Description:
        
        want an array with shape (num_trials, num_ts_collect) to
        get the values necessary for a waterfall plot of CRs

        assume the input has shape (num_cells, num_trials, num_ts_collect)
"""
def calc_cr_amp(nc_rasters: np.ndarray) -> np.ndarray:
    pass

"""
    Description:

        want to return an array of (possibly) length (num_trials)
        where the value at a given trial number is the time-point
        at which the cr is initiated
"""
def calc_cr_onsets(dcn_rasters: np.ndarray, criterion: "") -> np.ndarray:
    pass

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
