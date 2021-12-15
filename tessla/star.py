from tesssystem import TessSystem

class Star(TessSystem):

    def __init__(self, name, tic=None, toi=None, source=None, sectors=np.array([], dtype=int), ntransiting=1):
        super().__init__(name, tic=tic, toi=toi, source=source, sectors=sectors, ntransiting=ntransiting)
    
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

    def set_met(self, met, met_err, met_prov):
        self.met = met
        self.met_err = met_err
        self.met_prov = met_prov