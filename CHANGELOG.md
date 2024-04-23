# (04-23-2024)

This is not a release, but I put it here for clarity, as there  
breaking changes with some commits that I have made.

Breaking Changes:

transform.pcs_to_crs now returns three things: the crs, the  
cr amplitudes, and the cr onset times. Their shapes are  
(num_trials, num_ts), (num_trials), (num_trials), respectively.

Other Changes:

I updated [example.py](https://github.com/gawdSim/cbm_pack/blob/main/example.py) to reflect the changes made to the packaging structure of cbm_pack  
