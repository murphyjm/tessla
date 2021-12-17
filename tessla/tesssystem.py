from astroquery.utils import download_file_list
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
                author=None, # SPOC or QLP
                sectors=np.array([], dtype=int), # Sectors you'd like to include. Can also pass string 'all' to use all available sectors.
                cadence=120, # By default, extract the 2-minute cadence data, as opposed to the 20 s cadence, if both are available for a TOI.
                default_flux_origin='pdcsap_flux', # By default, try to use the PDCSAP flux. If this isn't available for all sectors, the SAP flux will be used for consistency.
                ntransiting=1) -> None: # Annotation denotes that __init__ should always return None
        self.name = name
        self.tic = tic
        self.toi = toi
        self.author = author
        self.cadence = cadence
        self.ntransiting = ntransiting

        # Private attributes
        self._all_sectors_flag = False # Want to download all available sectors of data? Underscore at start since this attribute isn't necessarily needed by the client.
        self._default_flux_origin = default_flux_origin

        # Handle which sectors the user wants to use.
        # NOTE: Easier/cleaner way to do this?
        if type(sectors) == str:
            if sectors.lower() == 'all':
                self._all_sectors_flag = True
        if not self._all_sectors_flag:
            try:
                len(sectors)
            except TypeError:
                sectors = np.array([sectors], dtype=int) # If only a single sector passed as an int or float, then make it a single-element array.
            self.sectors = sectors

    def get_tess_phot(self) -> pd.DataFrame:
        '''
        Download the TESS photometry. 
        
        Use PDCSAP flux by default, but can specify to use SAP flux instead.
        If multiple sectors of photometry are requested and one sector has e.g. SAP flux but no PDCSAP flux, 
        then use SAP flux for all sectors, even if PDCSAP is available for some, for consistency. An example of this is TOI-1824, 
        which has 30-min cadence Sector 15 SAP light curve, but no PDCSAP light curve for Sector 15 (as opposed to its other sectors,
        which are 2-min cadence and have both PDCSAP and SAP.) 
        '''
        lc_dict = self.__download_tess_phot(self)
        assert None not in lc_dict.values(), "Problem downloading the TESS photometry."
        flux_origins = [lc.FLUX_ORIGIN for lc in lc_dict.values()]
        if all(flux_origins == 'pdcsap_flux'):
            # If PDCSAP flux is available for all sectors, the use whatever the user requested as the flux type. 
            # Could have a scenario where PDCSAP is available for all sectors but the user specifies to use SAP flux instead. 
            # This is fine because if SAP flux should be available regardless of whether PDCSAP is available or not. 
            flux_origin = self._default_flux_origin
        elif any(flux_origins == 'sap_flux'):
            # If PDCSAP flux is *not* available for all sectors e.g. for TOI-1824, then use SAP flux for all sectors for consistency.
            flux_origin = 'sap_flux'
            if self._default_flux_origin != 'sap_flux':
                # If the user requested to use PDCSAP flux for all sectors but this is not possible, let them know about the change.
                print(f"Requested to use {self._default_flux_origin} but this is not available for all sectors.\n Using {flux_origin} instead for consistency.")
        
        # Add this attribute to the TessSystem object specifying whether we're using the PDCSAP or SAP light curve.
        self.flux_origin = flux_origin

        self.__clean_and_normalize_tess_photometry(self, lc_dict)

        return self.lc_df
        
    def __download_tess_phot(self) -> dict:
        '''
        Download the lightkurve objects, if available. 
        Private method, since the user shouldn't need to access this one specifically.
        '''
        lc_dict = {}
        for sector in self.sectors:
            print(f"Downloading Sector {sector} data...")
            success = False
            for author in ['SPOC', 'QLP']:
                if not success:
                    lc = lk.search_lightcurve(f"TIC {self.tic}", author=author, sector=sector)
                if not lc:
                    print(f"No {author} light curve found for Sector {sector}...")
                else:
                    lc = lc.download()
                    lc_dict[sector] = lc
                    success = True
                    print(f"Download successful for Sector {sector} {author} light curve...")
        return lc_dict

    def __clean_and_normalize_tess_photometry(self, lc_dict) -> None:
        '''
        Do some initial cleaning and normalizing of the TESS photometry.
        '''
        df = pd.DataFrame([], columns=['time', 'flux', 'flux_err'])

        # Iterate over each sector of photometry.
        for lc in lc_dict.values():
            lc = lc.remove_nans().normalize().remove_outliers()
            lc = lc[lc.quality == 0]
            flux = getattr(lc, self.flux_origin)
            flux_normed = (flux.value / np.median(flux.value) - 1) * 1e3 # Puts flux in units of PPT
            flux_err = lc.flux_err.value / np.median(flux.value) * 1e3   # Puts flux error in units of PPT
            lc_sector_df = pd.DataFram(data={'time':lc.time.value, 'flux':flux_normed, 'flux_err':flux_err})
            df = pd.concat((df, lc_sector_df))
        
        # Add the concatenated dataframe containing the light curve for all sectors as an attribute.
        self.lc_df = df

    def get_rvs(self, rv_fname):
        pass
        
