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

def get_semimajor_axis(period_samples, mstar, mstar_err):
    '''
    Return a in units of AU.
    '''
    mstar_samples = np.random.normal(mstar, mstar_err, len(period_samples))
    period_samples /= 365.25 # Convert JD to years
    return np.cbrt(np.square(period_samples) * mstar_samples)

def get_luminosity(teff, teff_err, rstar, rstar_err, n_samples):
    '''
    Return L in units of L_sun
    '''
    TEFF_SOL = 5772 # K.
    rstar_samples = np.random.normal(rstar, rstar_err, n_samples)
    teff_samples = np.random.normal(teff/TEFF_SOL, teff_err/TEFF_SOL, n_samples)
    return np.square(rstar_samples) * np.power(teff_samples, 4)

def get_sinc(teff, teff_err, rstar, rstar_err, a_samples):
    '''
    Return insolation flux in units of S_earth
    '''
    luminosity_samples = get_luminosity(teff, teff_err, rstar, rstar_err, len(a_samples))
    return luminosity_samples / np.square(a_samples)

def get_aor(a_samples, r_star_samples):
    return (a_samples.values * units.AU).to(units.R_sun).value / r_star_samples

def get_teq(a_samples, teff, teff_err, r_star_samples, bond_albedo=0):
    '''
    Return planet equilibrium temperature in units of Kelvin
    '''
    n_samples = len(r_star_samples)
    teff_samples = np.random.normal(teff, teff_err, n_samples)
    a_samples_sun = (a_samples.values * units.AU).to(units.R_sun).value
    return teff_samples * (1 - bond_albedo)**(0.25) * np.sqrt(r_star_samples / (2 * a_samples_sun))