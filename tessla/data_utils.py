import numpy as np
from astropy import units
import pandas as pd

def find_breaks(time, diff_threshold=10, verbose=False):
    '''
    Identify breaks between non-consecutive sectors of data (identified by gaps larger than diff_threshold days)
    '''
    diffs = np.ediff1d(time)
    diffs = np.append(diffs, 0) # To make it the same length for mask
    break_inds = np.arange(len(time))[diffs > diff_threshold]

    # Verify that the breaks seem correct
    if verbose:
        for break_ind in break_inds:
            print(f"break_ind = {break_ind}")
            print(f"Gap between {time[break_ind]:.2f} and {time[break_ind + 1]:.2f} BTJD")
            print("=======")
    return break_inds

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

def get_luminosity(teff_samples, rstar_samples):
    '''
    Return L in units of L_sun
    '''
    TEFF_SOL = 5772 # K.
    teff_samples_sol = teff_samples / TEFF_SOL
    return np.square(rstar_samples) * np.power(teff_samples_sol, 4)

def get_semimajor_axis(period_samples, mstar_samples):
    '''
    Return a in units of AU.
    '''
    period_samples_yr = period_samples / 365.25 # Convert JD to years
    return np.cbrt(np.square(period_samples_yr) * mstar_samples)

def get_sinc(a_samples, teff_samples, rstar_samples):
    '''
    Return insolation flux in units of S_earth
    '''
    luminosity_samples = get_luminosity(teff_samples, rstar_samples)
    return luminosity_samples / np.square(a_samples)

def get_aor(a_samples, rstar_samples):
    return (a_samples * units.AU).to(units.R_sun).value / rstar_samples

def get_teq(a_samples, teff_samples, rstar_samples, bond_albedo=0):
    '''
    Return planet equilibrium temperature in units of Kelvin
    '''
    a_samples_sun = (a_samples * units.AU).to(units.R_sun).value
    return teff_samples * (1 - bond_albedo)**(0.25) * np.sqrt(rstar_samples / (2 * a_samples_sun))

def __get_summary_info(chain):
    median = np.median(chain)
    std = np.std(chain)
    q = np.quantile(chain, [0.16, 0.84]) - median
    err16, err84 = q[0], q[-1]
    min = np.min(chain)
    max = np.max(chain)
    return [median, std, err16, err84, min, max]

def quick_look_summary(toi, df_derived_chains):
    columns = ['median', 'std', 'err16', 'err84', 'min', 'max']
    df = pd.DataFrame(columns=columns)
    for letter in toi.transiting_planets.keys():
        for param in ['period', 't0', 'rp', 'dur_hr', 'b', 'ecc', 'omega_folded_deg']:
            df.loc[f"{param}_{letter}"] = __get_summary_info(df_derived_chains[f"{param}_{letter}"])
    df.to_csv(toi.output_dir, f"{toi.name.replace(' ', '_')}_quick_look_summary.csv")