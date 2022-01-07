import numpy as np
from tesssystem import TessSystem

class Planet(TessSystem):
    '''
    Object to house data related to a planet in a tess system.
    '''
    def __init__(self, name, tic=None, toi=None, mission='TESS', cadence=120, flux_origin='sap_flux', ntransiting=1, phot_gp_kernel='activity', verbose=True,
                pl_letter='b',
                pl_toi_suffix='.01',
                per=None,
                per_err=None,
                t0=None,
                t0_err=None,
                bjd_offset=2457000, # By default, assume that t0 is reported in BTJD i.e., BJD - 2457000. If you want to start the light curve modeling at t = 0, then there will be an additional offset.
                dur=None,
                dur_err=None,
                depth=None,
                depth_err=None
                ) -> None:
        super().__init__(name, tic=tic, toi=toi, mission=mission, cadence=cadence, flux_origin=flux_origin, ntransiting=ntransiting, phot_gp_kernel=phot_gp_kernel, verbose=verbose)
    
        self.pl_letter = pl_letter
        self.pl_toi_suffix = pl_toi_suffix
        self.per = per
        self.per_err = per_err
        self.t0 = t0
        self.t0_err = t0_err
        self.bjd_offset = bjd_offset
        self.dur = dur
        self.dur_err = dur_err
        self.depth = depth
        self.depth_err = depth_err

    def __create_transit_mask(self, time):
        '''
        Create a transit mask.
        '''
        return np.abs((time - self.t0 + 0.5 * self.per) % self.per - 0.5 * self.per) < self.dur

    def get_transit_mask(self, time):
        '''
        Create a mask for the in-transit data for this planet.
        '''
        self.transit_mask = self.__create_transit_mask(time)
        return self.transit_mask