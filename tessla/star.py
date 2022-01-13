from tesssystem import TessSystem

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
        self.teff = teff
        self.teff_err = teff_err
        self.teff_prov = teff_prov
        self.feh = feh
        self.feh_err = feh_err
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