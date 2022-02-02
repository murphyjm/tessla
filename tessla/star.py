import numpy as np
from tessla.data_utils import get_density

class Star:
    '''
    Object to hold properties about the stellar characterization.
    '''
    def __init__(self, 
                toi,
                mstar=None,
                mstar_err=None,
                mstar_prov=None,
                rstar=None,
                rstar_err=None,
                rstar_prov=None,
                rhostar=None,
                rhostar_err=None,
                rhostar_prov=None,
                teff=None,
                teff_err=None,
                teff_prov=None,
                feh=None,
                feh_err=None,
                feh_prov=None,
                vmag=None,
                gmag=None,
                jmag=None,
                hmag=None,
                kmag=None):

        self.mstar = mstar
        self.mstar_err = mstar_err
        self.mstar_prov = mstar_prov
        self.rstar = rstar
        self.rstar_err = rstar_err
        self.rstar_prov = rstar_prov

        N = 1000
        # If given stellar mass and radius but no density, compute it
        if mstar is not None and rstar is not None and rhostar is None:
            mstar_chain = np.random.normal(mstar, mstar_err, N)
            rstar_chain = np.random.normal(rstar, rstar_err, N)
            rhostar_chain = get_density(mstar_chain, rstar_chain, 'solMass', 'solRad', 'g', 'cm')
            rhostar = np.median(rhostar_chain)
            rhostar_err = np.median(np.abs(np.quantile(rhostar_chain, [0.16, 0.84]) - rhostar))
            assert mstar_prov == rstar_prov, "Difference provenance values for mstar and rstar"
            rhostar_prov = mstar_prov

        self.rhostar = rhostar
        self.rhostar_err = rhostar_err
        self.rhosstar_prov = rhostar_prov

        self.teff = teff
        self.teff_err = teff_err
        self.teff_prov = teff_prov
        self.feh = feh
        self.feh_err = feh_err
        self.feh_prov = feh_prov
        self.vmag = vmag
        self.gmag = gmag
        self.jmag = jmag
        self.hmag = hmag
        self.kmag = kmag
    
    def set_teff(self, teff, teff_err, teff_prov):
        self.teff = teff
        self.teff_err = teff_err
        self.teff_prov = teff_prov

    def set_mstar(self, mstar, mstar_err, mstar_prov):
        self.mstar = mstar
        self.mstar_err = mstar_err
        self.mstar_prov = mstar_prov

    def set_rstar(self, rstar, rstar_err, rstar_prov):
        self.rstar = rstar
        self.rstar_err = rstar_err
        self.rstar_prov = rstar_prov

    def set_feh(self, feh, feh_err, feh_prov):
        self.feh = feh
        self.feh_err = feh_err
        self.feh_prov = feh_prov
    