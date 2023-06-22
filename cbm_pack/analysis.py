#!/usr/bin/env python

import numpy as np

"""
    Description:
        
        want an array with shape (num_trials, num_ts_collect) to
        get the values necessary for a waterfall plot of CRs

        assume the input has shape (num_cells, num_trials, num_ts_collect)
"""
def calc_cr_amp(dcn_rasters: np.ndarray) -> np.ndarray:
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
