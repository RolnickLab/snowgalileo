## NOTES / TODOS:

Ideally the addition / removal of channels should be dynamic at some point.
For now, active channels and their belonging to different shape groups (s_t_h_x, sp_x, t_x) needs to be adjusted both in `.config.py` and in `.earthengine/eo.py` (group indeces according to the channel group).

If at some point, different s_t_h_x groups should be added, the branch `space_time_distinction` offers a starting point.
If at some point, we want to resample / reproject data, the branch `resampling` offers a starting point.


Current dataset structure:

- 35 exportet time varying channels + 3 cloud channels
- 8 timesteps
- 2 non-exportet time varying channels (NDVI, NDSI)
- 4 space-only channels

The end.