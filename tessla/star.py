import os
import pickle
import numpy as np
import pandas as pd
from tessla.data_utils import get_density

def pickle_star_from_isoclassify_output(id_starname_list, isoclassify_csv_path, output_dir, specmatch_version='emp'):
    '''
    Create a star object using output from isoclassify.
    '''
    assert os.path.isfile(isoclassify_csv_path), "Invalid isoclassify output file path."
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(isoclassify_csv_path)
    assert all([starname in df.id_starname.values for starname in id_starname_list]), "At least one of the star id's you requested is not in the output .csv file."

    for starname in id_starname_list:
        star_props = {}

        iso_output = df[df['id_starname'] == starname].copy()
        
        # SpecMatch parameters
        assert specmatch_version in ['emp', 'syn'], "Valid inputs for specmatch_version are 'emp' and 'syn'."
        specmatch_version_str = f'specmatch-{specmatch_version}'
        for prop in ['teff', 'feh']:
            star_props[prop] = iso_output[prop].values[0]
            star_props[prop + '_err'] = iso_output[prop + '_err'].values[0]
            star_props[prop + '_prov'] = specmatch_version_str
        
        if specmatch_version == 'syn':
            prop = 'logg'
            star_props[prop] = iso_output[prop].values[0]
            star_props[prop + '_err'] = iso_output[prop + '_err'].values[0]
            star_props[prop + '_prov'] = specmatch_version_str
        
        # Isoclassify parameters
        # Age
        prop = 'age'
        star_props[prop] = iso_output['iso_' + prop].values[0]
        star_props[prop + '_err'] = np.median([iso_output['iso_' + prop + '_err1'].values[0], np.abs(iso_output['iso_' + prop + '_err2'].values[0])])
        star_props[prop + '_prov'] = 'isoclassify'
        # Mass
        prop = 'mass'
        star_props['mstar'] = iso_output['iso_' + prop].values[0]
        star_props['mstar' + '_err'] = np.median([iso_output['iso_' + prop + '_err1'].values[0], np.abs(iso_output['iso_' + prop + '_err2'].values[0])])
        star_props['mstar' + '_prov'] = 'isoclassify'
        # Radius
        prop = 'rad'
        star_props['rstar'] = iso_output['iso_' + prop].values[0]
        star_props['rstar' + '_err'] = np.median([iso_output['iso_' + prop + '_err1'].values[0], np.abs(iso_output['iso_' + prop + '_err2'].values[0])])
        star_props['rstar' + '_prov'] = 'isoclassify'
        # Rho
        prop = 'rho'
        star_props['rhostar'] = iso_output['iso_' + prop].values[0]
        star_props['rhostar' + '_err'] = np.median([iso_output['iso_' + prop + '_err1'].values[0], np.abs(iso_output['iso_' + prop + '_err2'].values[0])])
        star_props['rhostar' + '_prov'] = 'isoclassify'

        star = Star(**star_props)

        with open(os.path.join(output_dir, f"{starname}_iso_star_obj.pkl"), "wb") as star_fname:
            pickle.dump(star, star_fname, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Pickled star objects saved to {output_dir}")

class Star:
    '''
    Object to hold properties about the stellar characterization.
    '''
    def __init__(self,
                age=None,
                age_err=None,
                age_prov=None,
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
                logg=None,
                logg_err=None,
                logg_prov=None,
                vmag=None,
                gmag=None,
                jmag=None,
                hmag=None,
                kmag=None):

        self.age = age
        self.age_err = age_err
        self.age_prov = age_prov

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
        self.logg = logg
        self.logg_err = logg_err
        self.logg_prov = logg_prov

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

    def set_logg(self, logg, logg_err, logg_prov):
        self.logg = logg
        self.logg_err = logg_err
        self.logg_prov = logg_prov
    