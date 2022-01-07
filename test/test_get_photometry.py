import numpy as np
import pytest
from tessla.tesssystem import TessSystem

def test_downlaod_phot_simple():
    '''
    Simple test to see if downloading the photometry works properly.
    '''
    toi = TessSystem('HIP8152', 
                    tic=164767175, 
                    toi=266,
                    mission="TESS",
                    cadence=120,
                    flux_origin='sap_flux', 
                    ntransiting=2, 
                    verbose=True)
    lc = toi.get_tess_phot()
    break_ind = np.argmax(np.ediff1d(lc.time))

    # There should be sectors of data spaced by a wide time gap.
    assert lc.time[break_ind + 1] - lc.time[break_ind] > 10, "Something went wrong with the data download."

def test_normalized_flux():
    '''
    Check that the normalization seems to work properly.
    '''
    toi = TessSystem('HIP8152', 
                    tic=164767175, 
                    toi=266,
                    mission="TESS",
                    cadence=120,
                    flux_origin='sap_flux', 
                    ntransiting=2, 
                    verbose=True)
    lc = toi.get_tess_phot()
    assert np.abs(np.median(lc.flux) - 1) < 0.1, "Something is wrong with the flux normalization."

def test_no_data_available():
    '''
    Try to download data for a bogus TIC number.
    '''
    foo = TessSystem('Foobar', 
                    tic=1234567890,
                    toi=123)
    with pytest.raises(AssertionError):
        foo.get_tess_phot()