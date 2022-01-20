import numpy as np
from astropy import units

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

def get_density(mass, radius, input_mass_units, intput_radius_units, output_mass_units, output_radius_units):
    '''
    Convenience function for calculating bulk density.
    '''
    density = mass / (4/3 * np.pi * radius**3) * getattr(units, input_mass_units) / getattr(units, intput_radius_units)**3
    return density.to(getattr(units, output_mass_units) / getattr(units, output_radius_units)**3).value

def convert_negative_angles(omega):
    if omega < 0:
        omega += 2 * np.pi
    return omega

'''
TODO: Clean these functions and standardize.
'''
def get_luminosity(teff_samples, rstar_samples):
    '''
    Return L in units of L_sun
    '''
    TEFF_SOL = 5772 # K.
    teff_samples /= TEFF_SOL
    return np.square(rstar_samples) * np.power(teff_samples, 4)

def get_semimajor_axis(period_samples, mstar_samples):
    '''
    Return a in units of AU.
    '''
    period_samples /= 365.25 # Convert JD to years
    return np.cbrt(np.square(period_samples) * mstar_samples)

def get_sinc(teff_samples, rstar_samples, a_samples):
    '''
    Return insolation flux in units of S_earth
    '''
    luminosity_samples = get_luminosity(teff_samples, rstar_samples)
    return luminosity_samples / np.square(a_samples)

def get_aor(a_samples, rstar_samples):
    return (a_samples.values * units.AU).to(units.R_sun).value / rstar_samples

def get_teq(a_samples, teff_samples, rstar_samples, bond_albedo=0):
    '''
    Return planet equilibrium temperature in units of Kelvin
    '''
    a_samples_sun = (a_samples.values * units.AU).to(units.R_sun).value
    return teff_samples * (1 - bond_albedo)**(0.25) * np.sqrt(rstar_samples / (2 * a_samples_sun))