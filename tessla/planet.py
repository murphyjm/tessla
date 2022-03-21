import numpy as np

# What color to associate with each planet in plots.
# Can change/set these as needed.
planet_color_dict = {
    'b':'cornflowerblue',
    'c':'orange',
    'd':'forestgreen'
}

class Planet:
    '''
    Object to house data related to a planet in a tess system.
    '''
    def __init__(self, 
                toi_bjd_ref,
                pl_letter='b',
                pl_toi_suffix='.01',
                per=None,
                per_err=None,
                t0=None,
                t0_err=None,
                bjd_ref=2457000, # By default, assume that t0 is reported in BTJD i.e., BJD - 2457000. If you want to start the light curve modeling at t = 0, then there will be an additional offset.
                dur=None,
                dur_err=None,
                depth=None,
                depth_err=None
                ) -> None:
    
        self.pl_letter = pl_letter
        self.pl_toi_suffix = pl_toi_suffix
        self.per = per
        self.per_err = per_err

        # A little janky. But make sure that t0 uses the same time reference offset.
        self.t0 = t0
        self.t0_err = t0_err
        if bjd_ref != toi_bjd_ref:
            self.bjd_ref = toi_bjd_ref
            self.t0 = self.t0 + bjd_ref - toi_bjd_ref

        self.dur = dur
        self.dur_err = dur_err
        self.depth = depth
        self.depth_err = depth_err

        assert pl_letter in planet_color_dict.keys(), "Planet letter {pl_letter} not in planet_color_dict"
        self.color = planet_color_dict[pl_letter]

    def __create_transit_mask(self, time):
        '''
        Create a transit mask.
        '''
        # HACK
        return np.abs((time.value - self.t0 + 0.5 * self.per) % self.per - 0.5 * self.per) < self.dur

    def get_transit_mask(self, time):
        '''
        Create a mask for the in-transit data for this planet.
        '''
        self.transit_mask = self.__create_transit_mask(time)
        return self.transit_mask
    
    