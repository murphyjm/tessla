import os
import numpy as np
import pandas as pd
import lightkurve as lk
from tessla.data_utils import time_delta_to_data_delta
from scipy.signal import savgol_filter

from tessla.plotting_utils import sg_smoothing_plot

class TessSystem:
    '''
    Container object that holds meta information about a system.
    '''
    def __init__(self, 
                name, # CPS ID or common name if not in Jump.
                tic=None, 
                toi=None, 
                mission='TESS', # Only use TESS data by default. Could also specify other missions like "Kepler" or "all".
                cadence=120, # By default, extract the 2-minute cadence data, as opposed to the 20 s cadence, if both are available for a TOI.
                flux_origin='sap_flux', # By default, use the SAP flux. Can also specify "pdcsap_flux" but this may not be available for all sectors.
                n_transiting=1, # Number of transiting planets
                n_keplerians=None, # Number of Keplerian signals to include in models of the RVs. This will include transiting planets and any potentially non-transiting planet signals.
                bjd_ref=2457000, # BJD offset
                phot_gp_kernel='activity', # What GP kernel to use to flatten the light curve. 
                                            # Options are: ['activity', 'exp_decay', 'rotation']. Activity is exp_decay + rotation. See celerite2 documentation.
                verbose=True,
                plotting=True,
                output_dir=None) -> None:
        self.name = name
        self.tic = tic
        self.toi = toi
        self.mission = mission
        self.cadence = cadence
        
        self.n_transiting = n_transiting
        self.n_keplerians = n_transiting
        self.transiting_planets = {}

        self.bjd_ref = bjd_ref
        self.phot_gp_kernel = phot_gp_kernel
        self.verbose = verbose
        self.plotting = plotting

       # Default flux origin to use. SAP flux used by default.
        self.flux_origin = flux_origin

        # Organize the output directory structure
        save_dir_prefix = ''
        if output_dir is not None:
            save_dir_prefix = output_dir
        phot_out_dir = os.path.join(save_dir_prefix, 'photometry')
        if not os.path.isdir(phot_out_dir):
            os.makedirs(phot_out_dir) # TODO: Some sort of warning re: overwriting.
        self.phot_dir = phot_out_dir

    def add_transiting_planet(self, planet) -> None:
        '''
        Add a transiting planet to the TessSystem object.
        '''
        self.transiting_planets[planet.pl_letter] = planet
    
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

        Returns: The collection of lk.Light
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

    def __sg_smoothing(self, window_size, positive_outliers_only=False, max_iters=10, sigma_thresh=3):
        '''
        Before using a GP to flatten the light curve, remove outliers using a Savitzky-Golay filter.
        '''
        m = np.ones(len(self.lc.time), dtype=bool)
        
        for i in range(max_iters):
            # import pdb; pdb.set_trace()
            # HACK
            norm_flux_prime = np.interp(self.lc.time.value, self.lc.time[m].value, self.lc.norm_flux[m].value)
            if (window_size % 2) == 0:
                if self.verbose:
                    print(f"Must use an odd window size. Changing window size from {window_size} to {window_size + 1}.")
                window_size += 1
            smooth = savgol_filter(norm_flux_prime, window_size, polyorder=3)
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
        m &= ~self.all_transits_mask
        if self.verbose:
            print(f"{len(self.lc.time) - m.sum()} {sigma_thresh}-sigma outliers identified.")

        # Should these be attributes or just returned by this function?
        self.sg_outlier_mask = m
        self.sg_smoothed_flux = smooth

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
        window_size = time_delta_to_data_delta(self.lc.time, time_window=time_window)
        sg_outlier_mask = self.__sg_smoothing(window_size, positive_outliers_only=positive_outliers_only, max_iters=max_iters, sigma_thresh=sigma_thresh)
        if self.plotting:
            sg_smoothing_plot(self)