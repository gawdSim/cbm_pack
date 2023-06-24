#!/usr/bin/env python

import numpy as np

MS_PER_TIME_STEP = 1

"""
Public Const gETauRN = 15
Public Const ELeakRN = 0
Public gLeakRN As Single
Public gEDecayRN As Single

Public Type RedNucleusCell
  Vm As Single
  gE(NCNUMBER) As Single
End Type

Public RN As RedNucleusCell

Public Sub RN_TDV()
Dim X As Integer
    gLeakRN = 0.025 / (6 - Time_step_size)
    gEDecayRN = Exp(-Time_step_size / gETauRN)
End Sub

Public Sub DoWorkRN()
Dim i As Integer
Dim gE As Single
    gE = 0
    For i = 1 To NCNUMBER
        RN.gE(i) = RN.gE(i) * gEDecayRN
        RN.gE(i) = RN.gE(i) + (Nc(i).act * 0.012)
        gE = gE + RN.gE(i)
    Next i
    gE = gE - 0.05
    If gE < 0 Then gE = 0
    gE = gE * gE * gE * 5#
    If gE < 0.02 Then gE = 0
    RN.Vm = RN.Vm - (gLeakRN * (RN.Vm)) + (gE * (80 - RN.Vm))
    RN_Histo(Bincounter) = RN.Vm * 100
End Sub
"""

"""
    Description:

        computes the value of the excitatory conductance
"""

#TODO: finish
def rn_integrator(nc_rasters: np.ndarray) -> np.ndarray:
    g_exc_tau_rn = 15.0
    e_leak_rn = 0.0
    g_leak_rn = 0.025 / 5.0
    g_exc_dec_rn = np.exp(-MS_PER_TIME_STEP / g_exc_tau_rn)
    num_cells, num_trials, num_ts_per_trial = nc_rasters.shape
    g_exc_sums = np.sum(nc_rasters, axis=0)
    g_exc = np.zeros(num_trials, num_ts_per_trials, dtype=np.single)
    for ts in np.arange(num_ts_per_trial):
        if ts == 0:
            g_exc[:, ts] = g_exc_sums[:, ts] * 0.012
        else:
            g_exc[:, ts] += g_exc[:, ts-1]
            g_exc[:, ts] += g_exc_sums[:, ts] * 0.012
        
        g_exc[:, ts] *= g_exc_dec_rn


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
