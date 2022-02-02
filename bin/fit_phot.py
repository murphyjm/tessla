'''
Fit the TESS photometry for a specific system.

This script uses a lot of default and/or pre-filled optional arguments. 
If something in the results looks funky, refer to the example notebooks for a more by-hand approach.
'''
# tessla imports
from tessla.tesssystem import TessSystem
from tessla.planet import Planet
from tessla.star import Star
from tessla.plotting_utils import sg_smoothing_plot, quick_transit_plot

# Script imports
import os
import argparse

def parse_args():
    '''
    Take in command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Model the photometry for a system. Use this script with a shell script to model photometry for a list of planets.")
    parser.add_argument("name", type=str, help="Reference name for the system.")
    parser.add_argument("tic", type=int, help="TIC number.")
    parser.add_argument("toi", type=int, help="TOI number.")
    parser.add_argument("star_props_path", type=str, help="Path to the .json file with the star properties for loading into the tessla.Star object.")
    parser.add_argument("--num_non_toi_planets", default=0, type=int, help="Number of transiting planets in the system that are not TOIs or have incorrect orbital properties in the TOI catalog. E.g. single-transit planets.")
    parser.add_argument("--planet_props_path", type=str, default=None, help="If there are transiting planets that are not TOIs in the system or the TOIs in the catalog have incorrect properties, this is the path to the .json file that contains the transit properties for the non-TOI planets.")
    parser.add_argument("--phot_gp_kernel", type=str, default="exp_decay", help="Kernel to use for the photometry flattening.")
    parser.add_argument("--verbose", action="store_true", help="Print out helpful progress statements during the analysis.")
    return parser.parse_args()

def main():
    args = parse_args()
    toi = TessSystem(args.name, tic=args.tic, toi=args.toi, phot_gp_kernel=args.phot_gp_kernel)
    lc = toi.get_tess_phot()
    toi.search_for_tois()
    toi.add_tois_from_catalog()

    # If there are TOIs that are not in the catalog or the TOIs in the catalog have incorrect properties, fix them manually.
    if args.num_non_toi_planets > 0 and args.planet_props_path is not None:
        '''
        # TODO: Will have to fix the case where the TOI catalog has incorrect orbital properties for a TOI. For now just implementing the manual addition of extra planet candidates that are non-TOIs. E.g. like in the case of TOI-266 and TOI-1471.
        '''
        assert os.path.exists(args.planet_props_path), f"{args.planet_props_path} is not a valid path."
        

if __name__ == "__main__":
    main()