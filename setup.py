from setuptools import setup

setup(name='tessla',
      version='0.1',
      # list folders, not files
      packages=['tessla',
                'test'],
      scripts=['bin/fit_phot.py'],
      package_data={'tessla': ['data/toi_list.csv']},
      )