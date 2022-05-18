# Core packages
import os
import warnings
import numpy as np
import pandas as pd
import lightkurve as lk
from astropy import units
import pickle

# Data utils
from tessla.data_utils import get_inclination, time_delta_to_data_delta, convert_negative_angles, get_semimajor_axis, get_sinc, get_aor, get_teq, get_density, get_inclination, get_t0s_in_range
from scipy.signal import savgol_filter

# Enables sampling with multiple cores on Mac.
import platform
import multiprocessing as mp

# Plotting utils
from tessla.plotting_utils import plot_periodogram

from tessla.planet import Planet

# Exoplanet, pymc3, theano imports. 
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing" # Not exactly sure what this flag change does.
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess

import arviz as az

# Radvel
from radvel.utils import Msini

'''
NOTE: Right now, nontransiting planets won't work, so don't add them!

Problem: How to have different length period vectors for the exoplanet orbit model. E.g. 3 transiting planet periods and 2 additional nontransiting planets. 
Might just make sense to use RadVel in that case.
'''

RV_INST_NAME_MAPPER = {
    'apf':'APF',
    'j':'HIRES',
    'hires_j':'HIRES'
}

T0_BTJD = 2457000

class TessSystem:
    '''
    Container object that holds meta information about a system.
    '''
    def __init__(self, 
                # System stuff
                name, # CPS ID or common name if not in Jump.
                tic=None, # TIC ID
                toi=None, # TOI Number
                star=None, # Star object containing stellar properties

                # Photometry stuff
                mission='TESS', # Only use TESS data by default. Could also specify other missions like "Kepler" or "all".
                cadence=120, # By default, extract the 2-minute cadence data.
                use_long_cadence_data=False, # By default, don't use the 30-min cadence data if it's the only data available for a sector. If true, treat it as a different instrument.
                flux_origin='sap_flux', # By default, use the SAP flux. Can also specify "pdcsap_flux" but this may not be available for all sectors.
                n_transiting=1, # Number of transiting planets
                bjd_ref=2457000, # BJD offset. BTJD as default
                phot_gp_kernel='exp_decay', # What GP kernel to use to flatten the light curve. 
                                            # Options are: ['activity', 'exp_decay', 'rotation']. Activity is exp_decay + rotation. See celerite2 documentation.
                # RV stuff
                rv_data_path=None, # Path to RV data .csv. Must contain columns: "time", "mnvel" [m/s], "errvel" [m/s], "tel"
                mnvel_cut=None,
                errvel_cut=None, # If a float is provided, cut all mnvel and errvel absolute values above this limit. To get rid of bad data.
                # General stuff
                verbose=True, # Print out messages
                plotting=True, # Create plots as you go
                output_dir=None, # Output directory. Will default to CPS Name if non is provided.
                output_dir_suffix='') -> None:  # Suffix for output directory. E.g. TOI-1824_test_01
        
        # System-level attributes
        self.name = name
        self.tic = tic
        self.toi = toi
        self.mission = mission
        self.cadence = cadence
        self.flux_origin = flux_origin
        self.star = star
        self.toi_catalog_csv_path = '~/code/tessla/data/toi_list.csv'
        self.use_long_cadence_data = use_long_cadence_data

        # Planet-related attributes
        self.n_transiting = n_transiting # Should get rid of this attribute eventually, and just go by the length of the dictionary self.transiting_planets.
        self.transiting_planets = {}
        self.all_transits_mask = None
        self.toi_catalog = None

        self.bjd_ref = bjd_ref
        self.phot_gp_kernel = phot_gp_kernel
        self.verbose = verbose
        self.plotting = plotting

        # Set this at the start
        self.chains_path = None # TODO: Could instantiate the object with a chains path from a previous run
        self.chains_derived_path = None
        self.map_soln = None

        # Organize the output directory structure
        if output_dir is None:
            self.output_dir = self.name.replace(' ', '_') + output_dir_suffix
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # If RVs were included so creating a joint model
        self.rv_data_path = rv_data_path
        self.rv_df = None
        if self.rv_data_path is not None:
            if self.verbose:
                print("RV dataset detected. This will be a joint photometry-RV model.")
            model_out_dir = os.path.join(self.output_dir, 'joint_model')
            sampling_out_dir = os.path.join(self.output_dir, 'joint_sampling')
            self.is_joint_model = True

            # Read in the RV data from file.
            rv_df = pd.read_csv(self.rv_data_path, comment='#')
            cols = rv_df.columns.tolist()
            msg = 'RV .csv file must have the following columns: ["time", "mnvel", "errvel", "tel"], where "mnvel" and "errvel" are in m/s.'
            assert all([col in cols for col in ['time', 'mnvel', 'errvel', 'tel']]), msg
            # Standardize 'j' and 'hires_j' to 'HIRES' and 'apf' to 'APF'
            bad_mask = np.zeros(len(rv_df), dtype=bool)
            if mnvel_cut is not None:
                bad_mask |= np.abs(rv_df.mnvel) > mnvel_cut
            if errvel_cut is not None:
                bad_mask |= np.abs(rv_df.errvel) > errvel_cut
            rv_df = rv_df[~bad_mask].reset_index(drop=True)
            if self.verbose:
                print(f"Removed {np.sum(bad_mask)} RVs for being outliers.")
            rv_df['tel'] = rv_df['tel'].map(RV_INST_NAME_MAPPER).fillna(rv_df['tel'])
            self.rv_df = rv_df
            self.rv_inst_names = np.unique(rv_df['tel'])
            self.num_rv_inst = len(self.rv_inst_names)

        # Photometry-only
        else:
            # Sub-directory for photometry
            model_out_dir = os.path.join(self.output_dir, 'photometry')
            sampling_out_dir = os.path.join(self.output_dir, 'phot_sampling')
            self.is_joint_model = False

        # Make these subdirectories
        if not os.path.isdir(model_out_dir):
            os.makedirs(model_out_dir) # TODO: Some sort of warning re: overwriting.
        if not os.path.isdir(sampling_out_dir):
            os.makedirs(sampling_out_dir)

        self.model_dir = model_out_dir
        self.sampling_dir = sampling_out_dir

    def add_transiting_planet(self, planet) -> None:
        '''
        Add a transiting planet to the TessSystem object.
        '''
        self.transiting_planets[planet.pl_letter] = planet
        self.n_transiting = len(self.transiting_planets)
    
    def remove_transiting_planet(self, pl_letter):
        '''
        Remove a transiting planet.
        '''
        return self.transiting_planets.pop(pl_letter)

    def update_transiting_planet_props_to_map_soln(self):
        '''
        Update the transiting planet properties to the MAP solution values. Is optionally automatically run after the MAP fitting procedure.
        '''
        assert self.map_soln is not None, "MAP Solution is none. Must run MAP fitting procedure first."
        for i,planet in enumerate(self.transiting_planets.values()):
            planet.per = self.map_soln["period"][i]
            planet.t0 = self.map_soln["t0"][i]
            planet.depth = (self.map_soln["ror"][i])**2 * 1e3 # Places in units of PPT
            if not self.is_joint_model:
                planet.dur = self.map_soln["dur"][i]

    def search_for_tois(self):
        '''
        Look for TOIs in the TOI catalog.
        '''
        catalog = pd.read_csv(self.toi_catalog_csv_path, comment='#')
        result = catalog[catalog['TIC'] == self.tic]
        if self.verbose:
            print(f"Identified {len(result)} TOI(s) for TIC {self.tic}.")
            print("----------")
            for i,row in result.iterrows():
                print(row)
                print("----------")
        self.toi_catalog = result
    
    def add_tois_from_catalog(self):
        '''
        Add TOIs from the TOI catalog.
        '''
        assert self.toi_catalog is not None, "No TOIs found in TOI list or you have not searched for them yet."

        # HACK
        pl_letter_mapper = {
            '.01':'b',
            '.02':'c',
            '.03':'d',
            '.04':'e',
            '.05':'f',
            '.06':'g'
        }

        for i,row in self.toi_catalog.iterrows():
            pl_toi_suffix = str(row['Full TOI ID'])[-3:]
            pl_letter = pl_letter_mapper[pl_toi_suffix]
            planet = Planet(self.bjd_ref,
                            pl_letter=pl_letter, 
                            pl_toi_suffix=pl_toi_suffix, 
                            per=row['Orbital Period Value'], 
                            t0=row['Epoch Value'], 
                            dur=row['Transit Duration Value']/24, 
                            depth=row['Transit Depth Value'] * 1e-3)
            self.add_transiting_planet(planet)
        
        self.create_transit_mask()

    def update_t0s_to_near_data_middle(self, buffer=100): # If t0 is already within buffer days of the data middle, that's fine.
        '''
        To reduce the covariance bewteen period and t0, use a t0 initial guess that is close to the middle of the photometry timeseries.
        '''
        phot_data_middle = 0.5 * (np.max(self.lc.time.value) - np.min(self.lc.time.value))
        for planet in self.transiting_planets.values():
            if np.abs(phot_data_middle - planet.t0) < buffer:
                continue
            else:
                t0s = get_t0s_in_range(np.min(self.lc.time.value), np.max(self.lc.time.value), planet.per, planet.t0)
                middle_ind = int(np.median(np.arange(len(t0s))))
                middle_t0 = t0s[middle_ind]
                assert np.abs(phot_data_middle - middle_t0) < buffer, "New t0 is still not within buffer."
                if self.verbose:
                    print(f"Planet {planet.pl_letter} t0 updated from {planet.t0:.2f} to {middle_t0:.2f} [BJD - {self.bjd_ref:.2f}] to reduce P and t0 covariance.")
                planet.t0 = middle_t0

    def add_star_props(self, star):
        '''
        Add star properties.
        '''
        self.star = star

    def set_phot_gp_kernel(self, kernel_name):
        '''
        Set the phot_gp_kernel attribute.
        '''
        assert kernel_name in ['exp_decay', 'activity', 'rotation'], 'That kernel name is not recognized'
        self.phot_gp_kernel = kernel_name

    def get_tess_phot(self) -> lk.LightCurve:
        '''
        Download the TESS photometry. 
        
        Use SAP flux by default, but can specify to use PDCSAP flux instead, 
            though PDCSAP flux might not be available for all sectors.
        
        Returns
        ----------
        lc (lk.lightcurve): Stitched, normalized, and cleaned light curve object.
        '''
        # Download the photometry
        collection = self.__download_tess_phot()
        
        # Clean and normalize the photometry 
        lc = self.__clean_and_normalize_tess_phot(collection)
        
        # Set t=0 to the start of the data.
        t_start = np.min(lc['time'])
        lc['time'] -= t_start.value
        self.bjd_ref += t_start.value

        if self.rv_df is not None:
            self.rv_df.time = self.rv_df.time.values - self.bjd_ref

        self.lc = lc

        return lc
        
    def __download_tess_phot(self) -> lk.LightCurveCollection:
        '''
        Download the photometry.

        Only use lightcurves collected with the same exposure time. 
            I.e., don't combine a sector of 30-min cadence FFI data with a 2-min cadence light curve. 
            Could adjust this in the future, but strict use of same cadence data for now.

        Args
        ----------

        Returns
        ----------
        The collection of lk.Light
        '''
        search_result = lk.search_lightcurve(f"TIC {self.tic}")
        if self.verbose:
            print(f"Found {len(search_result)} data products for TIC {self.tic}:")
            print(search_result)
            print(f"Extracting all {self.cadence} s cadence data...")
        
        # Pick out the data of the correct cadence and mission.
        mask = search_result.exptime.value == self.cadence
        mask &= search_result.author == 'SPOC'
        # If self.use_long_cadence_data is True, download 30-min cadence data if there is no 2-min cadence data for that sector.
        if self.use_long_cadence_data:
            long_cadence_mask = (search_result.exptime.value == 1800) & (search_result.author == "TESS-SPOC")
            for i in range(len(mask)):
                mission_str = search_result[i].mission[0]
                if mission_str in search_result[mask].mission:
                    continue
                elif long_cadence_mask[i]:
                    mask[i] = True
        mission_series = pd.Series(search_result.mission)
        mask &= mission_series.str.contains(self.mission, case=False, regex=False).values # Only use data from the same mission.
        assert np.sum(mask) > 0, "No data fits the cadence and mission criteria for this target."

        # Isolate the lightkurve objects that we want based on the mask above and download the data.
        collection = search_result[mask].download_all()
        assert collection is not None, "Trouble downloading the data"
        if self.verbose:
            print("----------")
            print("Data that will be used for modeling:")
            print(search_result[mask])
        return collection

    def __stitch_corrector(self, lc) -> lk.lightcurve:
        '''
        Corrector function to be passed to lk.LightCurveCollection.stitch().

        Args
        ----------
        lc (lk.lightcurve): Lightcurve object to be cleaned/normalized.

        Returns
        ----------
        lc (lk.lightcurve): The cleaned and normalized light curve to be stitched together.
        '''
        # Clean the data
        lc = lc.remove_nans().normalize().remove_outliers()
        lc = lc[lc.quality == 0]

        # Normalize the flux
        flux = getattr(lc, self.flux_origin)
        flux_normed = (flux.value / np.median(flux.value) - 1) * 1e3 # Puts flux in units of PPT
        flux_normed_err = getattr(lc, f"{self.flux_origin}_err")
        flux_normed_err = flux_normed_err.value / np.median(flux.value) * 1e3 # Puts flux error in units of PPT
        
        # Add these columns to the lightcurve object.
        lc['norm_flux'] = flux_normed # PPT
        lc['norm_flux_err'] = flux_normed_err # PPT

        lc['sector'] = lc.sector

        return lc

    def __clean_and_normalize_tess_phot(self, collection) -> lk.LightCurve:
        '''
        Do some initial cleaning and normalizing of the TESS photometry. Stitch all of the lightcurves together.

        Args
        -----
        collection (lk.LightCurveCollection): Light curve collection containing the data we'll use.

        Returns
        -----
        lk.LightCurve: The stitched, cleaned, and normalized light curve.
        '''
        return collection.stitch(self.__stitch_corrector)

    def create_transit_mask(self):
        '''
        Create a mask for the in-transit data.

        Args
        ----------
        transiting_planets (Iterable): An iterable of tessla.Planet objects containing estimates of the planet duration  

        Returns
        ----------
        all_transit_mask (ndarray): A boolean mask where True/1 means the data is in-transit and False/0 means the data is out-of-transit for all transiting planets.
        '''
        assert self.n_transiting == len(self.transiting_planets), "Number of transiting planets for tessla.TessSystem does not match length of transiting_planets argument."
        assert all([self.bjd_ref == planet.bjd_ref for planet in self.transiting_planets.values()]), "Not all of the tessla.Planet objects are using the same bjd_ref as the tessla.TessSystem object."

        self.all_transits_mask = np.zeros(len(self.lc.time), dtype=bool)

        for planet in self.transiting_planets.values():
            transit_mask = planet.get_transit_mask(self.lc.time)
            self.all_transits_mask |= transit_mask
        
        return self.all_transits_mask

    def reset_transit_mask(self):
        '''
        Reset the transit mask if you add/remove planets basd on the 
        '''
        self.all_transits_mask = np.zeros(len(self.lc.time), dtype=bool)

    def __sg_smoothing(self, positive_outliers_only=False, max_iters=10, sigma_thresh=3):
        '''
        Before using a GP to flatten the light curve, remove outliers using a Savitzky-Golay filter.
        '''
        m = np.ones(len(self.lc.time), dtype=bool)
        
        for i in range(max_iters):
            # HACK: Why is this a hack? I forget...
            norm_flux_prime = np.interp(self.lc.time.value, self.lc.time[m].value, self.lc.norm_flux[m].value)
            if (self.sg_window_size % 2) == 0:
                if self.verbose:
                    print(f"Must use an odd window size. Changing window size from {self.sg_window_size} to {self.sg_window_size + 1}.")
                self.sg_window_size += 1
            smooth = savgol_filter(norm_flux_prime, self.sg_window_size, polyorder=3)
            resid = self.lc.norm_flux - smooth
            sigma = np.sqrt(np.mean(resid ** 2))
            m0 = np.abs(resid) < sigma_thresh * sigma
            if m.sum() == m0.sum():
                m = m0
                if self.verbose:
                    print(f"SG smoothing and outlier removal converged after {i+1} iterations.")
                self.sg_iters = i + 1
                if positive_outliers_only:
                    m = resid < sigma_thresh * sigma
                    if self.verbose:
                        print(f"Note: Only removing positive outliers.")

                break
            m = m0
        
        # Don't remove in-transit data.
        if self.all_transits_mask is None:
            self.all_transits_mask = np.zeros(len(m), dtype=bool)
        m = ~(~m & ~self.all_transits_mask) # Some bitwise operator kung-fu, but it makes sense.
        if self.verbose:
            print(f"{len(self.lc.time) - m.sum()} {sigma_thresh}-sigma outliers identified.")

        self.sg_outlier_mask = m
        return self.sg_outlier_mask

    def initial_outlier_removal(self, positive_outliers_only=False, max_iters=10, sigma_thresh=3, time_window=1):
        '''
        Smooth the initial light curve with a Savitzky-Golay filter and remove outliers before fitting the GP to flatten the light curve.

        Args
        ----------
        positive_outliers_only (bool): If False, remove both positive and negative outliers. If True, only remove positive outliers.
        max_iters (int): Maximum number of iterations for the loop to converge.
        sigma_thresh (float): Sigma threshold for identifying an outlier.
        time_window (float): Window in days to use for the SG filter.

        Returns
        ----------
        
        '''
        if self.all_transits_mask is None and self.n_transiting > 0 and self.verbose:
            print("Warning: No transit mask has been created, so this initial outlier removal step may flag in-transit data as outliers.")
        if self.n_transiting != len(self.transiting_planets):
            print("Warning: self.n_transiting does not equal len(self.transiting_planets)")
        self.sg_window_size = time_delta_to_data_delta(self.lc.time, time_window=time_window)
        self.__sg_smoothing(positive_outliers_only=positive_outliers_only, max_iters=max_iters, sigma_thresh=sigma_thresh)

    def oot_periodogram(self, min_per=1, max_per=50, samples_per_peak=1000, **kwargs):
        '''
        Use a LS periodogram of the out-of-transit flux to estimate the stellar rotation period.

        Args
        -----
        min_per (float): Minimum period in days to use for periodogram.
        max_per (float): Maximum period in days to use for periodogram.
        samples_per_peak (float): Number of samples to generate per peak. 
        plot (bool): Plot the periodogram if true.
        **kwargs (unpacked dictionary): Keyword arguments to send to plot_periodogram().

        Returns
        -----
        self.rot_per (float): Estimated rotation period of the star based on the OoT photometry. Peak of the periodogram.
        (fig, ax) (tuple): Figure and axes objects of the plot. Both None if plot=False.
        '''
        oot_mask = self.sg_outlier_mask & ~self.all_transits_mask
        xo_ls = xo.estimators.lomb_scargle_estimator(self.lc.time[oot_mask].value, self.lc.norm_flux[oot_mask].value, self.lc.norm_flux_err[oot_mask].value, 
                                                    min_period=min_per, max_period=max_per, samples_per_peak=samples_per_peak)
        try:
            peak = xo_ls['peaks'][0]
            rot_per_guess = peak['period']
            self.rot_per = rot_per_guess
        except IndexError:
            print("There were no peaks detected in the LS periodogram of the OoT data.")
            self.rot_per = None

        fig, ax = None, None
        if self.plotting:
            fig, ax = plot_periodogram(self.output_dir, f"{self.name} OoT Photometry LS Periodogram", 
                        xo_ls, 
                        self.transiting_planets,
                        verbose=self.verbose,
                        **kwargs) # What to do with figure and ax that is returned?
        return self.rot_per, (fig, ax)

    def __get_exp_decay_kernel(self):
        '''
        Create an exponentially-decaying SHOTerm GP kernel.
        '''
        log_sigma_dec_gp = pm.Normal("log_sigma_dec_gp", mu=0., sigma=10)
        BoundedNormal = pm.Bound(pm.Normal, lower=np.log(0.5), upper=np.log(50)) # Bounded normal for the periodic length scale so it's forced to be longer than 12 hours so it doesn't interfere with transit fitting. 
        log_rho_gp = BoundedNormal("log_rho_gp", mu=np.log(10), sd=np.log(50))
        kernel = terms.SHOTerm(sigma=tt.exp(log_sigma_dec_gp), rho=tt.exp(log_rho_gp), Q=1/np.sqrt(2))
        noise_params = [log_sigma_dec_gp, log_rho_gp]
        return noise_params, kernel

    def __get_rotation_kernel(self):
        '''
        Create a rotation term for the GP kernel.
        '''
        # TODO:
        # sigma_rot_gp = pm.InverseGamma("sigma_rot_gp", alpha=3.0, beta=2*np.std(y[mask]))
        # CONFUSED
        sigma_rot_gp = pm.InverseGamma(
            "sigma_rot_gp", **pmx.estimate_inverse_gamma_parameters(1, 5) # MAGIC NUMBERS FROM EXOPLANET TUTORIAL
        )
        log_prot = pm.Normal("log_prot", mu=np.log(self.rot_per), sd=np.log(5))
        prot = pm.Deterministic("prot", tt.exp(log_prot))
        log_Q0 = pm.Normal("log_Q0", mu=0, sd=2)
        log_dQ = pm.Normal("log_dQ", mu=0, sd=2)
        f = pm.Uniform("f", lower=0.01, upper=1)
        kernel = terms.RotationTerm(sigma=sigma_rot_gp, period=prot, Q0=tt.exp(log_Q0), dQ=tt.exp(log_dQ), f=f)
        noise_params = [sigma_rot_gp, log_prot, prot, log_Q0, log_dQ, f]
        return noise_params, kernel

    def __build_full_phot_model(self, x, y, yerr, mask=None, start=None, phase_lim=0.3, n_eval_points=500):
        '''
        Build the photometric model used to flatten the light curve.
        '''
        if mask is None:
            mask = np.ones(len(x), dtype=bool)

        with pm.Model() as model:

            # Parameters for the star
            mean_flux = pm.Normal("mean_flux", mu=0.0, sd=10.0)
            u = xo.QuadLimbDark("u")
            xo_star = xo.LimbDarkLightCurve(u)

            # Orbital parameters for the transiting planets
            # This transit fitting parameterization follows the methodology in the "Quick Fits for TESS Light Curves" Exoplanet tutorial.
            # Fit transits in terms of: P, t0, Rp/R*, transit duration (a form of stellar density), and b.
            assert_msg = "Not all of the transiting planets use the same BJD reference date as the TOI object so the t0 value will be incorrect."
            assert all([planet.bjd_ref == self.bjd_ref for planet in self.transiting_planets.values()]), assert_msg
            t0 = pm.Normal("t0", mu=np.array([planet.t0 for planet in self.transiting_planets.values()]), sd=1, shape=self.n_transiting)
            log_period = pm.Normal("log_period", mu=np.log(np.array([planet.per for planet in self.transiting_planets.values()])), sd=1, shape=self.n_transiting)
            period = pm.Deterministic("period", tt.exp(log_period))
            log_ror = pm.Normal("log_ror", mu=0.5 * np.log(1e-3 * np.array([planet.depth for planet in self.transiting_planets.values()])), sigma=np.log(10), shape=self.n_transiting)
            ror = pm.Deterministic("ror", tt.exp(log_ror))
            b = pm.Uniform("b", 0, 1, shape=self.n_transiting)
            log_dur = pm.Normal("log_dur", mu=np.log([planet.dur for planet in self.transiting_planets.values()]), sigma=np.log(10), shape=self.n_transiting)
            dur = pm.Deterministic("dur", tt.exp(log_dur))
            
            # Light curve jitter
            log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.median(yerr.values[mask])), sd=2)

            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, ror=ror, duration=dur)

            # Track the implied stellar density
            pm.Deterministic("rho_circ", orbit.rho_star)

            # Light curves
            light_curves = xo_star.get_light_curve(orbit=orbit, r=ror, t=x.values[mask], texp=self.cadence/60/60/24) * 1e3 # Converts self.cadence from seconds to days.
            light_curve = tt.sum(light_curves, axis=-1) + mean_flux
            resid = y.values[mask] - light_curve

            # Build the GP kernel
            gp_params = None
            kernel = None
            if self.phot_gp_kernel == 'exp_decay':
                exp_decay_params, exp_decay_kernel = self.__get_exp_decay_kernel()
                gp_params = exp_decay_params
                kernel = exp_decay_kernel
            elif self.phot_gp_kernel == 'rotation':
                rotation_params, rotation_kernel = self.__get_rotation_kernel()
                gp_params = rotation_params
                kernel = rotation_kernel
            elif self.phot_gp_kernel == 'activity':
                exp_decay_params, exp_decay_kernel = self.__get_exp_decay_kernel()
                gp_params = exp_decay_params
                kernel = exp_decay_kernel

                rotation_params, rotation_kernel = self.__get_rotation_kernel()
                gp_params += rotation_params
                kernel += rotation_kernel

            self.phot_gp_params = gp_params
            # Compute the GP model for the light curve
            gp = GaussianProcess(kernel, t=x.values[mask], diag=yerr.values[mask]**2 + tt.exp(2 * log_sigma_lc)) # diag= is the variance of the observational model.
            gp.marginal("gp", observed=resid)

            # Compute and save the phased light curve models
            phase_lc = np.linspace(-phase_lim, phase_lim, n_eval_points)
            lc_phase_pred = 1e3 * tt.stack(
                                    [
                                        xo_star.get_light_curve(
                                            orbit=orbit, r=ror, t=t0[n] + phase_lc, texp=self.cadence/60/60/24)[..., n]
                                            for n in range(self.n_transiting)
                                    ],
                                    axis=-1,
            )

            # Perform the MAP fitting
            if start is None:
                start = model.test_point
            # Order of parameters to be optimized is a bit arbitrary
            map_soln = pmx.optimize(start=start, vars=[log_sigma_lc])
            map_soln = pmx.optimize(start=map_soln, vars=gp_params)
            map_soln = pmx.optimize(start=map_soln, vars=[log_ror])
            map_soln = pmx.optimize(start=map_soln, vars=[b])
            map_soln = pmx.optimize(start=map_soln, vars=[log_dur])
            map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
            map_soln = pmx.optimize(start=map_soln, vars=[u])
            map_soln = pmx.optimize(start=map_soln, vars=[log_ror])
            map_soln = pmx.optimize(start=map_soln, vars=[b])
            map_soln = pmx.optimize(start=map_soln, vars=[log_dur])
            map_soln = pmx.optimize(start=map_soln, vars=[mean_flux])
            map_soln = pmx.optimize(start=map_soln, vars=[log_sigma_lc])
            map_soln = pmx.optimize(start=map_soln, vars=gp_params)
            map_soln = pmx.optimize(start=map_soln)

            extras = dict(
                zip(
                    ["light_curves", "gp_pred", "lc_phase_pred"],
                    pmx.eval_in_model([light_curves, gp.predict(resid), lc_phase_pred], map_soln)
                )
            )
            return model, map_soln, extras

    def __mark_resid_outliers(self, y, old_mask, map_soln, extras, sigma_thresh=7):
        '''
        Mark outliers and create a mask that removes them.
        '''
        mod = extras["gp_pred"] + np.sum(extras["light_curves"], axis=-1) + map_soln["mean_flux"]
        resid = y.values[old_mask] - mod
        rms = np.sqrt(np.median(resid ** 2))
        mask = np.abs(resid) < sigma_thresh * rms
        return mask

    def flatten_light_curve(self, phase_lim=0.3, n_eval_points=500, sigma_thresh=7, max_iters=10, update_planet_props_to_map_soln=True):
        '''
        Produce MAP fit of the photometry data, iteratively removing outliers until convergence to produce the flattened light curve.
        Will then extract the flattened data around each transit to produce a model of just the transits themselves to use during sampling. This should hopefully speedup sampling.
        '''

        map_soln = None
        old_mask = self.sg_outlier_mask # Start with the mask from the Savitzky-Golay filtering.
        x, y, yerr = pd.Series(self.lc.time.value), pd.Series(self.lc.norm_flux), pd.Series(self.lc.norm_flux_err) # Make these series objects so that we can keep track of the indices of the data that remain and be able to map their indices back on to the original dataset.
        tot_map_outliers = 0 # Does not count outliers removed from SG filtering.
        if self.verbose:
            print("Entering optimization and outlier rejection loop...")
        for i in range(max_iters):
            if self.verbose:
                print("====================")
                print(f"Optimization and outlier rejection iteration number {i + 1}.")
                print("====================")
            model, map_soln, extras = self.__build_full_phot_model(x, y, yerr, mask=old_mask, start=map_soln, phase_lim=phase_lim, n_eval_points=n_eval_points)
            new_mask = self.__mark_resid_outliers(y, old_mask, map_soln, extras, sigma_thresh=sigma_thresh)

            # Remove the outliers from the previous step.
            x, y, yerr = x[old_mask], y[old_mask], yerr[old_mask]
            
            tot_map_outliers += np.sum(~new_mask)
            
            if np.sum(~new_mask) == 0:
                break
            old_mask = new_mask

        if self.verbose:
            if i == max_iters - 1:
                print("Maximum number of iterations reached. MAP fitting loop did not converge.")
            print(f"MAP fitting and {sigma_thresh}-sigma outlier removal converged in {i + 1} iterations.")
            print(f"{tot_map_outliers} outlier(s) removed.")

        # Save the cleaned timestamps and flux as attributes
        self.cleaned_time, self.cleaned_flux, self.cleaned_flux_err = x, y, yerr

        # Save the cleaned and flattened flux as an attribute by subtracting out the GP and offset.
        self.cleaned_flat_flux = y - (extras["gp_pred"] + map_soln["mean_flux"])
        
        # Save these dictionaries as attributes. TODO: Also save model object?
        self.map_soln = map_soln
        self.extras = extras

        # Update planet properties with MAP values
        if update_planet_props_to_map_soln:
            if self.verbose:
                print("Updating transiting planet properties to MAP solution values")
            self.update_transiting_planet_props_to_map_soln()

        return model

    def __get_rv_model(self, t, K, orbit, trend_rv):

        planet_rv = orbit.get_radial_velocity(t, K=K)

        # Background model
        bkg = tt.zeros(len(t))
        if self.rv_trend:
            A = np.vander(t - self.rv_trend_time_ref, self.rv_trend_order + 1, increasing=True)# [:, :-1] # Don't use the offset term, we already have instrument offsets.
            bkg = tt.dot(A, trend_rv)
            
        return planet_rv, bkg, tt.sum(planet_rv, axis=-1) + bkg
        
    def __build_joint_model(self, x_phot, y_phot, yerr_phot, start=None, phase_lim=0.3, n_eval_points=500, t_rv_buffer=5):
        '''
        '''
        t_rv = np.linspace(self.rv_df.time.min() - t_rv_buffer, self.rv_df.time.max() + t_rv_buffer, n_eval_points)
        self.t_rv = t_rv
        with pm.Model() as model:
            mean_flux = pm.Normal("mean_flux", 0.0, sd=10.)
            u = xo.QuadLimbDark("u")
            xo_star = xo.LimbDarkLightCurve(u)
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
            mstar = BoundedNormal("mstar", mu=self.star.mstar, sd=self.star.mstar_err)
            rstar = BoundedNormal("rstar", mu=self.star.rstar, sd=self.star.rstar_err)
            
            # Orbital parameters for the transiting planets
            assert_msg = "Not all of the transiting planets use the same BJD reference date as the TOI object so the t0 value will be incorrect."
            assert all([planet.bjd_ref == self.bjd_ref for planet in self.transiting_planets.values()]), assert_msg
            t0 = pm.Normal("t0", mu=np.array([planet.t0 for planet in self.transiting_planets.values()]), sd=1, shape=self.n_transiting)
            log_K = pm.Normal("log_K", mu=np.log(np.std(self.rv_df.mnvel) * np.ones(self.n_transiting)), sigma=10, shape=self.n_transiting)
            K = pm.Deterministic("K", tt.exp(log_K))
            log_period = pm.Normal("log_period", mu=np.log(np.array([planet.per for planet in self.transiting_planets.values()])), sd=1, shape=self.n_transiting)
            period = pm.Deterministic("period", tt.exp(log_period))
            log_ror = pm.Normal("log_ror", mu=0.5 * np.log(1e-3 * np.array([planet.depth for planet in self.transiting_planets.values()])), sigma=np.log(10), shape=self.n_transiting)
            ror = pm.Deterministic("ror", tt.exp(log_ror))
            b = pm.Uniform("b", 0, 1, shape=self.n_transiting)
            
            # Eccentricity and omega
            ecs = pmx.UnitDisk("ecs", shape=(self.n_transiting, self.n_transiting), testval=0.01 * np.ones((self.n_transiting, self.n_transiting)))
            ecc = pm.Deterministic("ecc", tt.sum(ecs**2, axis=0))
            omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
            xo.eccentricity.vaneylen19("ecc_prior", multi=(self.n_transiting > 1), shape=self.n_transiting, fixed=True, observed=ecc)

            # Light curve jitter
            log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.median(yerr_phot.values)), sd=2)

            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(r_star=rstar, m_star=mstar, period=period, t0=t0, b=b, ecc=ecc, omega=omega)

            # Light curves
            light_curves = xo_star.get_light_curve(orbit=orbit, r=ror, t=x_phot.values, texp=self.cadence/60/60/24) * 1e3 # Converts self.cadence from seconds to days.
            light_curve = tt.sum(light_curves, axis=-1) + mean_flux
            resid_phot = y_phot.values - light_curve

            # Build the GP kernel
            gp_params = None
            kernel = None
            if self.phot_gp_kernel == 'exp_decay':
                exp_decay_params, exp_decay_kernel = self.__get_exp_decay_kernel()
                gp_params = exp_decay_params
                kernel = exp_decay_kernel
            elif self.phot_gp_kernel == 'rotation':
                rotation_params, rotation_kernel = self.__get_rotation_kernel()
                gp_params = rotation_params
                kernel = rotation_kernel
            elif self.phot_gp_kernel == 'activity':
                exp_decay_params, exp_decay_kernel = self.__get_exp_decay_kernel()
                gp_params = exp_decay_params
                kernel = exp_decay_kernel

                rotation_params, rotation_kernel = self.__get_rotation_kernel()
                gp_params += rotation_params
                kernel += rotation_kernel

            self.phot_gp_params = gp_params
            # Compute the GP model for the light curve
            gp = GaussianProcess(kernel, t=x_phot.values, diag=yerr_phot.values**2 + tt.exp(2 * log_sigma_lc)) # diag= is the variance of the observational model.
            gp.marginal("gp", observed=resid_phot)

            # Build the RV model
            if self.rv_trend:
                # Optionally add a polynomial trend as the RV background
                trend_rv = pm.Normal("trend_rv", mu=0, sd=10, shape=self.rv_trend_order + 1)
            else:
                trend_rv = None
            gamma_rv_list = []
            for tel in self.rv_inst_names:
                mask = self.rv_df['tel'] == tel
                gamma_rv_list.append(np.median(self.rv_df.loc[mask, 'mnvel']))
            gamma_rv = pm.Normal("gamma_rv", mu=np.array(gamma_rv_list), sigma=10, shape=self.num_rv_inst)
            sigma_rv = pm.HalfNormal("sigma_rv", sigma=10, shape=self.num_rv_inst)
            mean_rv = tt.zeros(len(self.rv_df))
            diag_rv = tt.zeros(len(self.rv_df))
            for i, tel in enumerate(self.rv_inst_names):
                mean_rv += gamma_rv[i] * np.array(self.rv_df['tel'] == tel, dtype=int)
                diag_rv += ((self.rv_df.errvel.values)**2 + sigma_rv[i]**2) * np.array(self.rv_df['tel'] == tel, dtype=int)
            
            # RV model
            planet_rv, bkg_rv, full_rv_model = self.__get_rv_model(self.rv_df.time, K, orbit, trend_rv)
            planet_rv_pred, bkg_rv_pred, full_rv_model_pred = self.__get_rv_model(t_rv, K, orbit, trend_rv)
            resid_rv = self.rv_df.mnvel.values - mean_rv - full_rv_model
            pm.Normal("obs_rv", mu=0, sd=tt.sqrt(diag_rv), observed=resid_rv)

            # Compute and save the phased light curve models
            phase_lc = np.linspace(-phase_lim, phase_lim, n_eval_points)
            lc_phase_pred = 1e3 * tt.stack(
                                    [
                                        xo_star.get_light_curve(
                                            orbit=orbit, r=ror, t=t0[n] + phase_lc, texp=self.cadence/60/60/24)[..., n]
                                            for n in range(self.n_transiting)
                                    ],
                                    axis=-1,
            )

            # Perform the MAP fitting
            if start is None:
                start = model.test_point
            # Order of parameters to be optimized is a bit arbitrary
            map_soln = pmx.optimize(start=start, vars=[log_sigma_lc])
            map_soln = pmx.optimize(start=map_soln, vars=[sigma_rv])
            if self.rv_trend:
                map_soln = pmx.optimize(start=map_soln, vars=[trend_rv])
            map_soln = pmx.optimize(start=map_soln, vars=[gamma_rv])
            map_soln = pmx.optimize(start=map_soln, vars=[log_K])
            map_soln = pmx.optimize(start=map_soln, vars=[log_ror])
            map_soln = pmx.optimize(start=map_soln, vars=[b])
            map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
            map_soln = pmx.optimize(start=map_soln, vars=[u])
            map_soln = pmx.optimize(start=map_soln, vars=[log_ror])
            map_soln = pmx.optimize(start=map_soln, vars=[b])
            map_soln = pmx.optimize(start=map_soln, vars=[mean_flux])
            map_soln = pmx.optimize(start=map_soln, vars=[log_sigma_lc])
            map_soln = pmx.optimize(start=map_soln, vars=gp_params)
            map_soln = pmx.optimize(start=map_soln)
            
            extras = dict(
                zip(
                    ["light_curves", 
                    "gp_pred",
                    "lc_phase_pred",
                    "mean_rv",
                    "err_rv",
                    "planet_rv",
                    "planet_rv_pred",
                    "bkg_rv",
                    "bkg_rv_pred",
                    "full_rv_model", 
                    "full_rv_model_pred"],
                    pmx.eval_in_model([light_curves, 
                                        gp.predict(resid_phot), 
                                        lc_phase_pred,
                                        mean_rv,
                                        np.sqrt(diag_rv),
                                        planet_rv,
                                        planet_rv_pred,
                                        bkg_rv,
                                        bkg_rv_pred,
                                        full_rv_model,
                                        full_rv_model_pred], 
                                        map_soln)
                )
            )
            return model, map_soln, extras

    def fit_phot_and_rvs(self, rv_trend=False, rv_trend_order=1, update_planet_props_to_map_soln=True):
        '''
        Build a joint RV and photometry model.
        '''
        self.rv_trend = rv_trend
        self.rv_trend_order = rv_trend_order
        self.rv_trend_time_ref = 0.5 * (np.max(self.rv_df.time) - np.min(self.rv_df.time)) # Reference time for the background trend model, if needed.
        # You may want to call self.flatten_light_curve() first because it will remove photometric outliers.
        model, map_soln, extras = self.__build_joint_model(self.cleaned_time, self.cleaned_flux, self.cleaned_flux_err, start=self.map_soln)

        self.map_soln = map_soln
        self.extras = extras

        # Update planet properties with MAP values
        if update_planet_props_to_map_soln:
            if self.verbose:
                print("Updating transiting planet properties to MAP solution values")
            self.update_transiting_planet_props_to_map_soln()

        return model

    def __flat_samps_to_csv(self, model, flat_samps, chains_output_fname):
        '''
        HACK

        Ugly hacky function to extract the concatenated chains and save them to a compressed .csv file.

        Something is going wrong in this function, at least when running on Expanse. 
        Not correctly converting from flat samples to .csv files.
        '''
        df_chains = pd.DataFrame()
        for param in model.named_vars.keys():
            try:
                num_dim = len(flat_samps[param].shape)
            except KeyError:
                continue
            if num_dim > 1 and param != 'u' and param != 'gamma_rv' and param != 'sigma_rv' and param != 'trend_rv':
                msg = "Chains and number of transiting planets have shape mismatch."
                assert len(self.transiting_planets) == flat_samps[param].shape[0], msg
                for i, pl_letter in enumerate(self.transiting_planets.keys()):
                    try:
                        df_chains[f"{param}_{pl_letter}"] = flat_samps[param][i, :].data
                    except ValueError:
                        continue
            elif param == 'u':
                for i in range(flat_samps[param].shape[0]):
                    df_chains[f"u_{i}"] = flat_samps[param][i, :].data
            elif param == 'gamma_rv':
                msg = "Chains and number of RV instruments have shape mismatch."
                assert flat_samps[param].shape[0] == len(self.rv_inst_names), msg
                for i,tel in enumerate(self.rv_inst_names):
                    df_chains[f"gamma_rv_{tel}"] = flat_samps[param][i, :].data
            elif param == 'sigma_rv':
                msg = "Chains and number of RV instruments have shape mismatch."
                assert flat_samps[param].shape[0] == len(self.rv_inst_names), msg
                for i,tel in enumerate(self.rv_inst_names):
                    df_chains[f"sigma_rv_{tel}"] = flat_samps[param][i, :].data
            elif param == 'trend_rv':
                for i in range(self.rv_trend_order + 1):
                    df_chains[f"trend_rv_{i}"] = flat_samps[param][i, :].data
            else:
                try:
                    df_chains[param] = flat_samps[param].data
                except ValueError:
                    continue
            
        df_chains.to_csv(chains_output_fname, index=False, compression="gzip")

    def run_sampling(self, model, map_soln, tune=6000, draws=4000, chains=8, cores=None, init_method='adapt_full', output_fname_suffix='', overwrite=False):
        '''
        Run the HMC sampling.
        '''
        
        # Enables parallel processing on Mac OS. 
        if platform.system().lower() == 'darwin':
            try:
                mp.set_start_method("fork")
            except RuntimeError:
                if self.verbose:
                    print("Multiprocessing context has already been set. Continuing.")
                else:
                    pass
        
        # Do some output directory housekeeping
        assert os.path.isdir(self.sampling_dir), "Output directory does not exist." # This should be redundant, but just in case.
        chains_output_fname = os.path.join(self.sampling_dir, f"{self.name.replace(' ', '_')}_phot_chains{output_fname_suffix}.csv.gz")
        if not overwrite and os.path.isfile(chains_output_fname):
            warnings.warn("Exiting before starting the sampling to avoid overwriting exisiting chains file.")
            return None, None

        trace_summary_output_fname = os.path.join(self.sampling_dir, f"{self.name.replace(' ', '_')}_trace_summary{output_fname_suffix}.csv")
        if not overwrite and os.path.isfile(trace_summary_output_fname):
            warnings.warn("Exiting before starting the sampling to avoid overwriting exisiting trace summary file.")
            return None, None

        if cores is None:
            cores = chains
        
        with model:
            trace = pmx.sample(
                            tune=tune,
                            draws=draws,
                            start=map_soln,
                            chains=chains,
                            cores=cores,
                            return_inferencedata=True,
                            init=init_method
            )
        
        # Save the trace summary which contains convergence information
        summary_df = az.summary(trace, round_to=5)
        summary_df.to_csv(trace_summary_output_fname)

        # Save the concatenated samples.
        flat_samps =  trace.posterior.stack(sample=("chain", "draw"))
        with open(os.path.join(self.sampling_dir, f"{self.name.replace(' ', '_')}_flat_samples.pkl"), "wb") as flat_samples_fname:
            pickle.dump(flat_samps, flat_samples_fname, protocol=pickle.HIGHEST_PROTOCOL)
        self.__flat_samps_to_csv(model, flat_samps, chains_output_fname)
        self.chains_path = chains_output_fname

        return flat_samps

    def add_ecc_and_omega_to_chains(self, flat_samps, rho_circ_param_name='rho_circ'):
        '''
        Estimate eccentricity and omega using the photoeccentric effect.
        '''

        # Flat priors
        ecc = np.random.uniform(0, 1, size=flat_samps[rho_circ_param_name].shape)
        omega = np.random.uniform(-np.pi, np.pi, size=flat_samps[rho_circ_param_name].shape) # Radians

        g = (1 + ecc * np.sin(omega)) / np.sqrt(1 - ecc**2)
        rho = flat_samps[rho_circ_param_name].data / g**3

        log_weights = -0.5 * ((rho - self.star.rhostar) / self.star.rhostar_err) **2 # Like a chi-square likelihood
        weights = np.exp(log_weights - np.max(log_weights))

        df_chains = pd.read_csv(self.chains_path)
        for i,letter in enumerate(self.transiting_planets.keys()):
            df_chains[f"ecc_{letter}"] = ecc[i, :]
            df_chains[f"omega_{letter}"] = omega[i, :]
            df_chains[f"ecc_omega_weights_{letter}"] = weights[i, :]

        # Don't overwrite the original chains file, just in case something goes wrong in adding columns for derived parameters.
        extension = '.csv.gz'
        chains_derived_path = self.chains_path[:self.chains_path.find(extension)] + '_derived' + extension
        self.chains_derived_path = chains_derived_path
        df_chains.to_csv(self.chains_derived_path, index=False, compression="gzip")

    def add_derived_quantities_to_chains(self):
        '''
        Add some useful derived quantities to the chains.
        '''
        if self.chains_derived_path is None:
            extension = '.csv.gz'
            chains_derived_path = self.chains_path[:self.chains_path.find(extension)] + '_derived' + extension
            self.chains_derived_path = chains_derived_path
            df_chains = pd.read_csv(self.chains_path)
        else:
            try:
                df_chains = pd.read_csv(self.chains_derived_path)
            except FileNotFoundError:
                df_chains = pd.read_csv(self.chains_path)
        N = len(df_chains)
        if self.is_joint_model:
            mstar_samples = df_chains['mstar'].values
            rstar_samples = df_chains['rstar'].values
        else:
            mstar_samples = np.random.normal(self.star.mstar, self.star.mstar_err, N)
            rstar_samples = np.random.normal(self.star.rstar, self.star.rstar_err, N)
        teff_samples = np.random.normal(self.star.teff, self.star.teff_err, N)
        for letter in self.transiting_planets.keys():
            df_chains[f"rp_{letter}"] = units.R_sun.to(units.R_earth, df_chains[f"ror_{letter}"] * rstar_samples) # Planet radius in earth radius
            df_chains[f"t0_BTJD_{letter}"] = df_chains[f"t0_{letter}"] + self.bjd_ref - T0_BTJD # T0 in BTJD

            if not self.is_joint_model:
                df_chains[f"dur_hr_{letter}"] = df_chains[f"dur_{letter}"] * 24 # Transit duration in hours

            df_chains[f"omega_folded_{letter}"] = df_chains[f"omega_{letter}"].apply(convert_negative_angles)
            df_chains[f"omega_folded_deg_{letter}"] = df_chains[f"omega_folded_{letter}"].values * 180 / np.pi # Convert omega from radians to degrees to have for convenience
            df_chains[f"a_{letter}"] = get_semimajor_axis(df_chains[f"period_{letter}"].values, mstar_samples)
            df_chains[f"aor_{letter}"] = get_aor(df_chains[f"a_{letter}"].values, rstar_samples)
            df_chains[f"sinc_{letter}"] = get_sinc(df_chains[f"a_{letter}"].values, teff_samples, rstar_samples)
            df_chains[f"teq_{letter}"] = get_teq(df_chains[f"a_{letter}"].values, teff_samples, rstar_samples) # Calculated assuming zero Bond albedo
            
            df_chains[f"i_rad_{letter}"], df_chains[f"i_deg_{letter}"] = get_inclination(df_chains[f"b_{letter}"].values, df_chains[f"a_{letter}"].values, rstar_samples)
            if self.is_joint_model:
                df_chains[f"msini_{letter}"] = Msini(df_chains[f"K_{letter}"], df_chains[f"period_{letter}"], mstar_samples, df_chains[f"ecc_{letter}"], Msini_units='earth')
                df_chains[f"mp_{letter}"] = df_chains[f"msini_{letter}"] / np.sin(df_chains[f"i_rad_{letter}"])
                df_chains[f"rho_{letter}"] = get_density(df_chains[f"mp_{letter}"].values, df_chains[f"rp_{letter}"].values, 'earthMass', 'earthRad', 'g', 'cm')

        df_chains.to_csv(self.chains_derived_path, index=False, compression="gzip")