'''
Fit the TESS photometry for a specific system. Optionally add in an RV data set and build a joint model.

This script uses a lot of default and/or pre-filled optional arguments. 
If something in the results looks funky, refer to the example notebooks for a more by-hand approach.
'''
# tessla imports
from tessla.tesssystem import TessSystem
from tessla.data_utils import find_breaks, quick_look_summary
from tessla.plotting_utils import sg_smoothing_plot, quick_transit_plot, plot_individual_transits, phase_plot, plot_phot_only_corners, plot_joint_corners
from tessla.threepanelphotplot import ThreePanelPhotPlot
from tessla.rvplot import RVPlot

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
    
    # Joint photometry-RV model?
    parser.add_argument("--rv_data_path", type=str, default=None, help="Path to RV data set to include in modeling.")
    parser.add_argument("--rv_trend", action="store_true", help="Include a linear trend in the background RV model.")

    # Data
    parser.add_argument("--flux_origin", type=str, default="sap_flux", help="Either pdcsap_flux or sap_flux. Default is SAP.")
    parser.add_argument("--use_long_cadence_data", action="store_true", help="WARNING: Don't use this feature quite yet save for testing. If included, it's okay to use 30-minute cadence data if no other data is available for that sector.")
    
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

def main():
    args = parse_args()

    assert os.path.isfile(args.star_obj_fname), f"Invalid Star object file name: {args.star_obj_fname}"

    # Create the TessSystem object, download the photometry, and add TOIs that appear in the TOI catalog.
    toi = TessSystem(args.name, 
                    tic=args.tic, 
                    toi=args.toi, 
                    phot_gp_kernel=args.phot_gp_kernel,
                    rv_data_path=args.rv_data_path,
                    plotting=(not args.no_plotting), 
                    flux_origin=args.flux_origin, 
                    use_long_cadence_data=args.use_long_cadence_data,
                    output_dir_suffix=args.output_dir_suffix)
    toi.get_tess_phot()
    toi.search_for_tois()
    toi.add_tois_from_catalog()
    
    # If there are TOIs that are not in the catalog or the TOIs in the catalog have incorrect properties, fix them manually.
    if args.planet_objs_dir is not None:
        toi.fix_planets(args.planet_objs_dir)

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
        star.inflate_star_mass_and_rad_errs() # Inflate the error bars on stellar mass and radius according to Tayar et al. 2022
        toi.add_star_props(star)

    # Before MAP fitting, update t0s to near middle of phot data to reduce covariance between P and t0
    toi.update_t0s_to_near_data_middle()
    
    # Run the MAP fitting loop
    model = toi.flatten_light_curve()

    # If there's an RV dataset, make a joint model of the photometry and RVs.
    if args.rv_data_path is not None:
        model = toi.fit_phot_and_rvs(rv_trend=args.rv_trend)
    quick_transit_plot(toi)
    
    # Make plots of the individual transits
    plot_individual_transits(toi)

    # Create a phase-folded plot of the light curve at the presumed stellar rotation period
    phase_plot(os.path.join(toi.model_dir, 'plotting'), 
                f"{toi.name} {toi.flux_origin.replace('_', ' ')}", 
                'Relative flux [ppt]', 
                toi.cleaned_time.values, toi.cleaned_flux.values, toi.rot_per, 0)

    # Plot the MAP solutions for inspection
    if toi.plotting:
        use_broken_x_axis = len(find_breaks(toi.cleaned_time.values)) > 0 # If breaks in the x-axis, then use the broken x-axis plot
        phot_plot = ThreePanelPhotPlot(toi,
                                    use_broken_x_axis=use_broken_x_axis, 
                                    plot_random_transit_draws=False)
        phot_plot.plot(save_fname=f"{toi.name.replace(' ', '_')}_phot_model" + args.plot_fname_suffix, overwrite=args.overwrite_plot)
        # Create a LS periodogram of the residuals about the full model. This should hopefully be white noise
        phot_plot.residuals_periodogram(overwrite=args.overwrite_plot)

        if toi.is_joint_model:
            rv_plot = RVPlot(toi, 
                             plot_random_orbit_draws=False)
            rv_plot.plot(save_fname=f"{toi.name.replace(' ', '_')}_rv_model" + args.plot_fname_suffix, overwrite=args.overwrite_plot)
        
    # Run the sampling
    flat_samps = None
    if not args.no_sampling:
        flat_samps = toi.run_sampling(model, 
                                    toi.map_soln, 
                                    tune=args.ntune, 
                                    draws=args.draws, 
                                    chains=args.nchains)
        
        if not toi.is_joint_model:
            # Add eccentricity and omega to the chains
            toi.add_ecc_and_omega_to_chains(flat_samps)
        toi.add_derived_quantities_to_chains()

        # Plot the results
        use_broken_x_axis = len(find_breaks(toi.cleaned_time.values)) > 0 # If breaks in the x-axis, then use the broken x-axis plot
        phot_plot = ThreePanelPhotPlot(toi,
                                    use_broken_x_axis=use_broken_x_axis, 
                                    plot_random_transit_draws=True,
                                    num_random_transit_draws=args.num_transit_draws)
        phot_plot.plot(save_fname=f"{toi.name.replace(' ', '_')}_phot_model_w_draws" + args.plot_fname_suffix, overwrite=args.overwrite_plot)

        if toi.is_joint_model:
            # Make RV plot
            rv_plot = RVPlot(toi, 
                             plot_random_orbit_draws=True, 
                             num_random_orbit_draws=(args.num_transit_draws))
            rv_plot.plot(save_fname=f"{toi.name.replace(' ', '_')}_rv_model_w_draws" + args.plot_fname_suffix, overwrite=args.overwrite_plot)
        
        # Read this data for derived corner plots and output summary table.
        df_derived_chains = pd.read_csv(toi.chains_derived_path)

        # Make the corner plots
        if toi.plotting:
            if not toi.is_joint_model:
                plot_phot_only_corners(toi, df_derived_chains, overwrite=args.overwrite_plot)
            else:
                plot_joint_corners(toi, df_derived_chains, overwrite=args.overwrite_plot)
        
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