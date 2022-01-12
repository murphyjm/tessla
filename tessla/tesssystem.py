# Core packages
import os
import numpy as np
import pandas as pd
import lightkurve as lk

# Data utils
from tessla.data_utils import time_delta_to_data_delta
from scipy.signal import savgol_filter

# Enables sampling with multiple cores.
import multiprocessing as mp
# mp.set_start_method("fork") # Add this at the start of the sampling portion of this script.

# Plotting utils
from tessla.plotting_utils import plot_periodogram

# Exoplanet, pymc3, theano imports. 
import exoplanet as xo
import pymc3_ext as pmx
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing" # Not exactly sure what this flag change does.
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess

class TessSystem:
    '''
    Container object that holds meta information about a system.
    '''
    def __init__(self, 
                name, # CPS ID or common name if not in Jump.
                tic=None, # TIC ID
                toi=None, # TOI Number
                mission='TESS', # Only use TESS data by default. Could also specify other missions like "Kepler" or "all".
                cadence=120, # By default, extract the 2-minute cadence data, as opposed to the 20 s cadence, if both are available for a TOI.
                flux_origin='sap_flux', # By default, use the SAP flux. Can also specify "pdcsap_flux" but this may not be available for all sectors.
                n_transiting=1, # Number of transiting planets
                n_keplerians=None, # Number of Keplerian signals to include in models of the RVs. This will include transiting planets and any potentially non-transiting planet signals.
                bjd_ref=2457000, # BJD offset
                phot_gp_kernel='activity', # What GP kernel to use to flatten the light curve. 
                                            # Options are: ['activity', 'exp_decay', 'rotation']. Activity is exp_decay + rotation. See celerite2 documentation.
                verbose=True, # Print out messages
                plotting=True, # Create plots as you go
                output_dir=None) -> None: # Output directory. Will default to CPS Name if non is provided.
        self.name = name
        self.tic = tic
        self.toi = toi
        self.mission = mission
        self.cadence = cadence
        
        # Planet-related attributes
        self.n_transiting = n_transiting # Should get rid of this attribute eventually, and just go by the length of the dictionary self.transiting_planets.
        self.n_keplerians = n_transiting
        self.transiting_planets = {}
        self.all_transits_mask = None

        self.bjd_ref = bjd_ref
        self.phot_gp_kernel = phot_gp_kernel
        self.verbose = verbose
        self.plotting = plotting

       # Default flux origin to use. SAP flux used by default.
        self.flux_origin = flux_origin

        # Organize the output directory structure
        if output_dir is None:
            self.output_dir = self.name
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Sub-directory for photometry
        phot_out_dir = os.path.join(self.output_dir, 'photometry')
        if not os.path.isdir(phot_out_dir):
            os.makedirs(phot_out_dir) # TODO: Some sort of warning re: overwriting.
        self.phot_dir = phot_out_dir

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
        
        # Pick out the data of the correct cadence and mission. Could update both of these options later.
        # E.g. at the moment, you can't ask for data from both Kepler and TESS, and/or 2-min data and 30-min data.
        mask = search_result.exptime.value == self.cadence # Only use data with the same exposure time. Could change this if wanted.
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
        flux_err = lc.flux_err.value / np.median(flux.value) * 1e3   # Puts flux error in units of PPT

        # Add these columns to the lightcurve object.
        lc['norm_flux'] = flux_normed # PPT
        lc['norm_flux_err'] = flux_err # PPT

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
            print("Warning: The number ")
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
                        {planet.pl_letter:planet.per for planet in self.transiting_planets.values()}, 
                        verbose=self.verbose,
                        **kwargs) # What to do with figure and ax that is returned?
        return self.rot_per, (fig, ax)

    def flatten_light_curve(self, sigma_thresh=7):
        '''
        Produce MAP fit of the photometry data, iteratively removing outliers until convergence to produce the flattened light curve.
        Will then extract the flattened data around each transit to produce a model of just the transits themselves to use during sampling. This should hopefully speedup sampling.
        '''
        pass