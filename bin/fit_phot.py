'''
Fit the TESS photometry for a specific system
'''

from email.policy import default
from parso import parse
from tessla.tesssystem import TessSystem
from tessla.planet import Planet
from tessla.star import Star
from tessla.plotting_utils import sg_smoothing_plot, quick_transit_plot

import argparse

def parse_args():
    '''
    Take in command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Model the photometry for a system. Use this script with a shell script to model photometry for a list of planets.")
    parser.add_argument("name", type=str, help="Reference name for the system.")
    parser.add_argument("tic", type=int, help="TIC number.")
    parser.add_argument("toi", type=int, help="TOI number.")
    parser.add_argument("--phot_gp_kernel", type=str, default="exp_decay", help="Kernel to use for the photometry flattening.")
    