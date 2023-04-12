'''
This testing module is modeled after the tests Dan Foreman-Mackey has written for the exoplanet source code as accessed on Github.
'''
import logging
import pymc3 as pm
import pymc3_ext as pmx
import numpy as np

class _Base:
    
    def __init__(self):
        self.random_seed = 509

    def _sample(self, **kwargs):
        logger = logging.getLogger("pymc3")
        logger.propagate = False
        logger.setLevel(logging.ERROR)
        kwargs["draws"] = kwargs.get("draws", 1000)
        kwargs["progressbar"] = kwargs.get("progressbar", False)
        kwargs["return_inferencedata"] = kwargs.get("return_inferencedata", False)
        return pmx.sample(**kwargs)

    def _model(self, **kwargs):
        np.random.seed(self.random_seed)
        return pm.Model(**kwargs)


class TestNoncentered(_Base):
    
    def __init__(self, random_seed=509):
        self.random_seed = random_seed
        super().__init__()
    
    def test_mstar_rstar(self):
        
        mu, sigma = 1, 0.05

        with self._model() as centered_model:

            # A centered implementation
            mu = pm.Normal("mu", mu, sigma)

            trace = self._sample()

            trace_cen = trace['mu']
            
        with self._model() as noncentered_model:
            
            # Noncentered implementation
            mu_offset = pm.Normal("mu_offset", 0, 1)

            mu = pm.Deterministic("mu", mu + (mu_offset * sigma))

            trace = self._sample()

            trace_nc = trace['mu']

        for f in [np.median, np.std]:
            assert np.allclose(f(trace_cen), f(trace_nc), rtol=1e-3, atol=1e-3)