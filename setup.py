from setuptools import setup

setup(name='tessla',
      version='0.1',
      # list folders, not files
      packages=['tessla',
                'test'],
      scripts=['bin/foo.py'],
      package_data={'tessla': ['data/toi_list.csv']},
      )