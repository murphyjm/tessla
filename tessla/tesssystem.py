from re import search
from astroquery.utils import download_file_list
from lightkurve import lightcurve
import numpy as np
import pandas as pd
import lightkurve as lk

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
                ntransiting=1,
                verbose=True) -> None:
        self.name = name
        self.tic = tic
        self.toi = toi
        self.mission = mission
        self.cadence = cadence
        self.ntransiting = ntransiting
        self.verbose = verbose

       # Default flux origin to use. SAP flux used by default.
        self.flux_origin = flux_origin

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

        return lc

    def __clean_and_normalize_tess_phot(self, collection) -> lk.LightCurve:
        '''
        Do some initial cleaning and normalizing of the TESS photometry.

        Args
        -----
        collection (lk.LightCurveCollection): Light curve collection containing the data we'll use.
        '''
        return collection.stitch(self.__stitch_corrector)

    def get_rvs(self, rv_fname):
        pass