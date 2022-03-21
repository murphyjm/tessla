from importlib_metadata import entry_points
from setuptools import setup

setup(name='tessla',
      version='0.1',
      # list folders, not files
      packages=['tessla',
                'test'],
      package_data={'tessla': ['data/toi_list.csv']},
      # scripts=['bin/fit_phot.py'],
      entry_points={
            'console_scripts':['tessla_fit_phot=tessla.fit_phot:main'],
      }

)