import numpy as np
from astropy import units
import pandas as pd
import os

def get_t0s_in_range(xstart, xstop, per, t0):
    '''
    Return a list of transit times between xstart and xstop for a planet with period per and reference transit epoch t0.
    '''
    first_transit_in_range = t0 + int((xstart - t0)/per) * per
    t0s = []
    t0_curr = first_transit_in_range
    # Definitely a quicker way to do this, but this is fine for now.
    while t0_curr < xstop:
        t0s.append(t0_curr)
        t0_curr += per
    if len(t0s) == 0:
        print("No transits in range...")
    return np.array(t0s)

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

def get_inclination(b_samples, a_samples, rstar_samples):   
    inclination_samples_rad = np.arccos(b_samples / ((a_samples * units.AU).to(units.R_sun).value) * rstar_samples)
    inclination_samples_deg = inclination_samples_rad * 180 / np.pi
    
    return inclination_samples_rad, inclination_samples_deg

def get_dur(period_samples, aor_samples, b_samples, i_samples, ecc_samples, omega_samples):
    '''
    Transit duration. Equation 14 from Winn 2010
    '''
    factor_one = period_samples / np.pi
    arcsin_arg = (1/aor_samples) * np.sqrt(1 - b_samples**2) / np.sin(i_samples)
    arcsin_arg = arcsin_arg % 1 # arcsin only defined on [-1, 1]
    factor_two = np.arcsin(arcsin_arg)
    factor_three = np.sqrt(1 - ecc_samples**2) / (1 + ecc_samples * np.sin(omega_samples))
    return factor_one * factor_two * factor_three

def get_dur_circ(period_samples, aor_samples):
    '''
    Transit duration assuming a circular orbit and b = 0. See Equation 4 in Petigura 2020.

    Returns in units of days.
    '''
    return period_samples / np.pi * np.arcsin(1 / aor_samples)

def get_tsm(pl_rade, pl_masse, pl_aor, rstar, teff, jmag):
    '''
    Calculate TSM from derived chains.
    '''
    pl_rade_med = np.median(pl_rade) # Determine the scale factor using the median of the planet's radius measurement.
    scale_factor = None
    if pl_rade_med < 1.5:
        scale_factor = 0.190
    elif pl_rade_med >= 1.5 and pl_rade_med < 2.75:
        scale_factor = 1.26
    elif pl_rade_med >= 2.75 and pl_rade_med < 4.0:
        scale_factor = 1.28
    elif pl_rade_med >= 4.0 and pl_rade_med < 10:
        scale_factor = 1.15
    else:
        scale_factor = -1 # Planet too large
    teq = teff * (np.sqrt(1/pl_aor)*(0.25**0.25))

    numerator = scale_factor * pl_rade**3 * teq * 10**(-1 * jmag / 5)
    denominator = pl_masse * rstar**2

    return numerator / denominator

def __get_summary_info(chain):
    median = np.median(chain)
    mean = np.mean(chain)
    std = np.std(chain)
    q = np.quantile(chain, [0.16, 0.84]) - median
    err16, err84 = q[0], q[-1]
    min = np.min(chain)
    max = np.max(chain)
    return [median, std, err16, err84, mean, min, max]

def quick_look_summary(toi, df_derived_chains):
    columns = ['median', 'std', 'err16', 'err84', 'mean', 'min', 'max'] # Labels must correspond to what's returned by __get_summary_info()
    df = pd.DataFrame(columns=columns)

    # Stellar properties
    stellar_params = ['mstar', 'rstar']
    for param in stellar_params:
        df.loc[param] = __get_summary_info(df_derived_chains[param])

    # Planet properties
    if not toi.is_joint_model:
        params = ['period', 't0', 'rp', 'dur_hr', 'b', 'ecc', 'omega_folded_deg']
        for letter in toi.transiting_planets.keys():
            prefix = ''
            for param in params:
                df.loc[f"{prefix}{param}_{letter}"] = __get_summary_info(df_derived_chains[f"{prefix}{param}_{letter}"])
    else:
        for letter in toi.transiting_planets.keys():
            params = ['period', 't0', 'rp', 'b', 'ecc', 'omega', 'K', 'msini', 'mp', 'rho', 'a', 'teq', 'tsm', 'dur_hr', 'dur_circ_hr', 'Rtau_Petigura2020']
            prefix = ''
            for param in params:
                df.loc[f"{prefix}{param}_{letter}"] = __get_summary_info(df_derived_chains[f"{prefix}{param}_{letter}"])
        for letter in toi.nontransiting_planets.keys():
            params = ['period', 't0', 'ecc', 'omega', 'K', 'msini', 'a', 'teq']
            prefix = 'nontrans_'
            for param in params:
                df.loc[f"{prefix}{param}_{letter}"] = __get_summary_info(df_derived_chains[f"{prefix}{param}_{letter}"])
    save_fname = f"{toi.name.replace(' ', '_')}_quick_look_summary.csv"
    save_path = os.path.join(toi.output_dir, save_fname)
    df.to_csv(save_path)
    print(f"Quick look summary table saved to {save_path}.")