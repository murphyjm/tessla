import numpy as np

def time_delta_to_data_delta(x, time_window=1) -> int:
    '''
    Convert a difference in time to a difference in the spacing of elements in an array.
    Used for e.g., picking how large of a window to use for the SG filter for the initial outlier removal.

    Args
    ----------
    x (Iterable): The array of data to use. Usually an array of times in units of days.
        target_time_delta (float): The window to use for the smoothing. Default = 0.25 days.

    Returns
    ----------
    int: The (median) number of array elements corresponding to the time window.
    '''
    med_time_delta = np.median(np.ediff1d(x)).value # HACK
    window_size = int(time_window / med_time_delta) # Points
    return window_size