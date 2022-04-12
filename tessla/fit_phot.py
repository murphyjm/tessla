'''
Fit the TESS photometry for a specific system.

This script uses a lot of default and/or pre-filled optional arguments. 
If something in the results looks funky, refer to the example notebooks for a more by-hand approach.
'''
# tessla imports
from tessla.tesssystem import TessSystem
from tessla.planet import Planet
from tessla.star import Star
from tessla.data_utils import find_breaks, quick_look_summary
from tessla.plotting_utils import sg_smoothing_plot, quick_transit_plot, plot_individual_transits, plot_corners
from tessla.threepanelphotplot import ThreePanelPhotPlot

# Script imports
import os
import pickle
import argparse

import pandas as pd

def parse_args():
    '''
    Take in command line arguments.
    '''
    parser = argparse.ArgumentParser(description="Model the photometry for a system. Use this script with a shell script to model photometry for a list of planets.")
    
    # System information and setup
    parser.add_argument("name", type=str, help="Reference name for the system.")
    parser.add_argument("tic", type=int, help="TIC number.")
    parser.add_argument("toi", type=int, help="TOI number.")
    parser.add_argument("star_obj_fname", type=str, help="Path to the .pkl file containing the tessla.star.Star object.")
    parser.add_argument("--planet_objs_dir", type=str, default=None, help="If there are transiting planets that are not TOIs in the system or the TOIs in the catalog have incorrect properties, this is the path to the directory with the .pkl files that contain the tessla.planet.Planet objects.")
    parser.add_argument("--output_dir_suffix", type=str, default='', help="Suffix for output directory. Default is empty string. E.g. '_test_01' for TOI-1824_test_01.")

    # Data
    parser.add_argument("--flux_origin", type=str, default="sap_flux", help="Either pdcsap_flux or sap_flux. Default is SAP.")

    # Model hyperparameters
    parser.add_argument("--phot_gp_kernel", type=str, default="exp_decay", help="Kernel to use for the photometry flattening.")

    # Sampling hyperparameters
    parser.add_argument("--no_sampling", action="store_true", help="If given then skip the HMC sampling and only run the MAP fitting.")
    parser.add_argument("--ntune", type=int, default=1000, help="Number of tuning steps per HMC chain.")
    parser.add_argument("--draws", type=int, default=1000, help="Number of draws per HMC chain.")
    parser.add_argument("--nchains", type=int, default=2, help="Number of chains for the HMC sampler.")

    # Plotting hyperparameters
    parser.add_argument("--no_plotting", action="store_true", help="If included, don't do any of the plotting.")
    parser.add_argument("--num_transit_draws", type=int, default=25, help="Number of random transit draws to plot in the 3-panel plot.")
    parser.add_argument("--plot_fname_suffix", type=str, default='', help="Suffix to append to the save name of the 3-panel plot.")
    parser.add_argument("--overwrite_plot", action="store_true", help="If included, overwrite an existing file when saving the 3-panel and corner plots.")
    # Bonus stuff
    parser.add_argument("--quiet", action="store_true", help="Disable print statements.")
    return parser.parse_args()

def fix_tois(toi, args):
    '''
    Fix incorrect entries in the TOI catalog or manually add planets that don't appear there.
    '''
    planet_dir = args.planet_objs_dir
    if not os.path.exists(planet_dir):
        print(f"{planet_dir} is not a valid path. Assuming there are no TOI entries to fix or manual planets to add. Continuing.")
    else:
        if not args.quiet:
            print(f"Loading {len(os.listdir(planet_dir))} non-TOI transiting planet(s) from {planet_dir}")
        for fname in os.listdir(planet_dir):
            f = os.path.join(planet_dir, fname)
            with open(f, 'rb') as planet_fname:
                # Load the planet from the pickled file.
                planet = pickle.load(planet_fname)
                # If we're replacing a planet remove that planet first.
                if planet.pl_letter in toi.transiting_planets.keys():
                    toi.remove_transiting_planet(planet.pl_letter)
                # Add the planet.
                toi.add_transiting_planet(planet)
        
        # Reset the transit mask and create the new one with the correct transiting planets.
        toi.reset_transit_mask()
        toi.create_transit_mask()

def main():
    args = parse_args()

    assert os.path.isfile(args.star_obj_fname), f"Invalid Star object file name: {args.star_obj_fname}"

    # Create the TessSystem object, download the photometry, and add TOIs that appear in the TOI catalog.
    toi = TessSystem(args.name, 
                    tic=args.tic, 
                    toi=args.toi, 
                    phot_gp_kernel=args.phot_gp_kernel, 
                    plotting=(not args.no_plotting), 
                    flux_origin=args.flux_origin, 
                    output_dir_suffix=args.output_dir_suffix)
    toi.get_tess_phot()
    toi.search_for_tois()
    toi.add_tois_from_catalog()
    
    # If there are TOIs that are not in the catalog or the TOIs in the catalog have incorrect properties, fix them manually.
    if args.planet_objs_dir is not None:
        fix_tois(toi, args)

    # Initial outlier removal with a SG filter
    toi.initial_outlier_removal(positive_outliers_only=False, max_iters=10, sigma_thresh=3, time_window=1)

    # Plot and save the initial outlier removal
    if toi.plotting:
        sg_smoothing_plot(toi)
    
    # This isn't strictly necessary, but estimate the rotation period using a periodogram OoT photometry
    toi.oot_periodogram(**{'label_peaks':False})

    # Add the star properties
    with open(args.star_obj_fname, 'rb') as star_fname:
        star = pickle.load(star_fname)
        toi.add_star_props(star)
    
    # Run the MAP fitting loop
    model = toi.flatten_light_curve()
    quick_transit_plot(toi)

    # Make plots of the individual transits
    plot_individual_transits(toi)
    
    # Run the sampling
    flat_samps = None
    if not args.no_sampling:
        flat_samps = toi.run_sampling(model, 
                                    toi.map_soln, 
                                    tune=args.ntune, 
                                    draws=args.draws, 
                                    chains=args.nchains)

        # Add eccentricity and omega to the chains
        toi.add_ecc_and_omega_to_chains(flat_samps)
        toi.add_derived_quantities_to_chains()

    if toi.plotting:
        # Plot the results
        use_broken_x_axis = len(find_breaks(toi.cleaned_time.values)) > 0 # If breaks in the x-axis, then use the broken x-axis plot
        phot_plot = ThreePanelPhotPlot(toi,
                                    use_broken_x_axis=use_broken_x_axis, 
                                    plot_random_transit_draws=(not args.no_sampling),
                                    num_random_transit_draws=args.num_transit_draws)
        phot_plot.plot(save_fname=f"{toi.name.replace(' ', '_')}_phot_model" + args.plot_fname_suffix, overwrite=args.overwrite_plot)

    # If the sampling was run...
    if flat_samps is not None:

        # Read this data for derived corner plots and output summary table.
        df_derived_chains = pd.read_csv(toi.chains_derived_path)

        # Make the corner plots
        if toi.plotting:
            plot_corners(toi, df_derived_chains, overwrite=args.overwrite_plot)
        
        # Save an output table with derived physical parameters in useful units for quickly checking on the sampling results.
        quick_look_summary(toi, df_derived_chains)
        
    # Save specific attributes as a pickled object or json file e.g. the dictionary with the MAP values: toi.map_soln. 
    # Can add to this list if wanted.
    with open(os.path.join(toi.output_dir, f"{toi.name.replace(' ', '_')}_map_soln.pkl"), "wb") as map_soln_fname:
        pickle.dump(toi.map_soln, map_soln_fname, protocol=pickle.HIGHEST_PROTOCOL)

    # For now: pickle the toi object
    # Can maybe get rid of this step since it takes up a lot of memory?
    with open(os.path.join(toi.output_dir, f"{toi.name.replace(' ', '_')}_toi_obj.pkl"), "wb") as toi_fname:
        pickle.dump(toi, toi_fname, protocol=pickle.HIGHEST_PROTOCOL)
    
    if not args.quiet:
        print(f"Analysis of {toi.name} photometry complete. Results stored in output directory: {toi.output_dir}.")

if __name__ == "__main__":
    main()