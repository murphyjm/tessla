import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
import numpy as np
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter, find_peaks
from tqdm import tqdm
import datetime
from astropy.time import Time

from tessla.tesslacornerplot import TesslaCornerPlot

def sg_smoothing_plot(toi, legend_outside=False):
    '''
    Make a sector-by-sector plot of the light curve and initial SG filter smoothing/outlier rejection.
    '''
    for sector in np.unique(toi.lc.sector):

        fig, ax = plt.subplots()
        ax.set_title(f'{toi.name} Sector {sector} Initial Outlier Rejection')

        sector_mask = toi.lc.sector == sector

        # Plot the data
        ax.plot(toi.lc.time[sector_mask].value, toi.lc.norm_flux[sector_mask].value, '.k', label='Data')

        # Plot the smoothed data
        sg_outlier_mask = toi.sg_outlier_mask & sector_mask
        norm_flux_prime = np.interp(toi.lc.time.value, toi.lc.time[sg_outlier_mask].value, toi.lc.norm_flux[sg_outlier_mask].value)[sg_outlier_mask]
        ax.plot(toi.lc.time[sg_outlier_mask].value, savgol_filter(norm_flux_prime, toi.sg_window_size, polyorder=3), color='C2', label='Smoothed Data', zorder=100)

        # Mark the in-transit data for each planet.
        for pl_letter, planet in toi.transiting_planets.items():
            transit_mask = planet.transit_mask & sector_mask
            ax.plot(toi.lc.time[transit_mask].value, toi.lc.norm_flux[transit_mask], '.', color=planet.color, label=f"Planet {pl_letter} in-transit ($P =$ {planet.per:.1f} d)")
        
        sg_outlier_mask_inv = ~toi.sg_outlier_mask & sector_mask
        ax.plot(toi.lc.time[sg_outlier_mask_inv].value, toi.lc.norm_flux[sg_outlier_mask_inv], 'xr', label="Discarded outliers")

        if legend_outside:
            ax.legend(fontsize=10, bbox_to_anchor=[1,1])
        else:
            ylim_bottom, ylim_top = ax.get_ylim()
            ax.set_ylim([ylim_bottom - 0.5, ylim_top]) # Make room for the legend below the data
            ax.legend(fontsize=10, loc='lower center', ncol=2)
        ax.set_xlabel(f'Time [BJD - {toi.bjd_ref:.1f}]')
        ax.set_ylabel(f'Relative flux [ppt]')

        ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(2.5))

        ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)
        # ax.yaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_minor_locator(MultipleLocator(2.5))

        out_dir = os.path.join(toi.model_dir, 'plotting')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        fig.savefig(os.path.join(out_dir, f"{toi.name.replace(' ', '_')}_sg_filtering_sector_{sector:02}.png"), facecolor='white', bbox_inches='tight', dpi=300)
        plt.close()
    
    if toi.verbose:
        print(f"SG smoothing plots saved to {out_dir}")


def plot_periodogram(out_dir, title, xo_ls, transiting_planets, label_peaks=True, peak_thresh=1.0, verbose=True):
    '''
    Plot a periodogram given data and some plot hyperparameters.

    Args
    -----
    periodogram_dir (str): Output directory for where to save the plot.
    title (str): Title for the plot
    xo_ls (exoplanet.estimators.lomb_scargle_estimator): LS periodogram object from exoplanet. 
    transiting_planets (dict): e.g., Dictionary with keys being planet letters and values Planet objects.
    label_peaks (bool): Default=True. If True, label peaks in LS power that rise above the FAP level specified by peak_thresh.
    peak_thresh (float): Default=1.0. If find_peaks=True, use this as the FAP percent threshold above which to label the peaks. Note: 1.0 == FAP level of 1%. 
    verbose (bool): If verbose, output print statements.
    Returns
    -----
    fig, ax: The figure and axis objects of the plot.
    '''
    fig, ax = plt.subplots()

    # Plot the periodogram
    freq, power = xo_ls['periodogram']
    periods = 1/freq
    ax.plot(periods, power, 'k', lw=0.5)

    # Label the periods of transiting planets.
    for pl_letter, planet in transiting_planets.items():
        ax.axvline(planet.per, label=f"Planet {pl_letter}, $P =$ {planet.per:.2f} d", color=planet.color, zorder=0)
    
    # Label the highest peak and its first harmonic
    top_peak_period = periods[np.argmax(power)]
    ax.axvline(top_peak_period, alpha=0.3, lw=5, color='purple', label=f"Highest peak, $P =$ {top_peak_period:.2f} d")
    ax.axvline(top_peak_period/2, alpha=0.3, lw=5, color='gray', label=f"Highest peak harmonic, $P =$ {top_peak_period/2:.2f} d")

    # Plot the false alarm probabilities
    ls = xo_ls['ls'] # Extract the astropy periodogram object
    fap_point01_percent = ls.false_alarm_level(0.0001) # 0.01 % level
    fap_point1_percent = ls.false_alarm_level(0.001) # 0.1 % level
    fap_1percent = ls.false_alarm_level(0.01) # 1 % level
    fap_10percent = ls.false_alarm_level(0.1) # 10 % level
    for fap,level,linestyle in zip([fap_point01_percent, fap_point1_percent, fap_1percent, fap_10percent], ['0.01%', '0.1%', '1%', '10%'], ['-', '--', '-.', ':']):
        ax.axhline(fap, label=f'FAP {level} level', ls=linestyle)

    # Optionally label periodogram peaks that rise above peak_thresh (which is given in units of FAP percentage! 
    # i.e., peak_thresh=1 corresponds to the FAP 1% level.
    if label_peaks:
        peak_inds, peak_props = find_peaks(power, height=float(ls.false_alarm_level(peak_thresh * 1e-2)))
        
        # Sort the peak periods according to highest to lowest power.
        peak_periods = periods[peak_inds]
        sorted_inds = np.argsort(peak_props['peak_heights'])[-5:]
        peak_heights = peak_props['peak_heights'][sorted_inds]
        peak_periods = peak_periods[sorted_inds]
        for i,per in enumerate(peak_periods):
            ax.text(per + per*0.05, peak_heights[i], f"$P =$ {per:.1f} d", fontsize=12)

    # Plot housekeeping
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("LS power")
    ax.legend(bbox_to_anchor=(1, 1), fontsize=12)
    ax.set_title(title)

    out_dir = os.path.join(out_dir, 'periodograms')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fig.savefig(os.path.join(out_dir, f'{title.replace(" ", "_")}_ls_periodogram.png'), facecolor='white', bbox_inches='tight', dpi=300)
    plt.close()

    if verbose:
        print(f"Periodogram plot saved to {out_dir}")

    return fig, ax

def quick_transit_plot(toi):
    '''
    Make a plot of what the MAP transit fit looks like for each planet before moving on to the sampling.
    '''
    
    out_dir = os.path.join(toi.model_dir, 'plotting')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for i,planet in enumerate(toi.transiting_planets.values()):
        fig, ax = plt.subplots()
        x, y = toi.cleaned_time, toi.cleaned_flux
        x_fold = ((x - toi.map_soln["t0"][i] + 0.5 * toi.map_soln["period"][i]) % toi.map_soln[
            "period"
        ][i] - 0.5 * toi.map_soln["period"][i]).values
        ax.scatter(x_fold, y - toi.extras['gp_pred'] - toi.map_soln['mean_flux'], c=x, s=3)
        phase = np.linspace(-0.3, 0.3, len(toi.extras['lc_phase_pred'][:, i]))
        ax.plot(phase, toi.extras['lc_phase_pred'][:, i], 'r', lw=5)
        ax.set_xlim(-6/24, 6/24)
        ax.xaxis.set_major_locator(MultipleLocator(3/24))
        ax.xaxis.set_minor_locator(MultipleLocator(1.5/24))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 24:g}'))
        ax.set_xlabel("time since transit [hours]")
        ax.set_ylabel("relative flux [ppt]")
        ax.set_title(f"{toi.name} {planet.pl_letter}")
        ax.text(0.1, 0.1, f"$P =$ {toi.map_soln['period'][i]:.1f} d", transform=ax.transAxes)
        ax.text(0.1, 0.05, f"$R_\mathrm{{p}}/R_* =$ {toi.map_soln['ror'][i]:.4f}", transform=ax.transAxes)
        
        save_fname = os.path.join(out_dir, f"initial_transit_fit_{toi.name.replace(' ', '_')}_{planet.pl_letter}.png")
        fig.savefig(save_fname, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()

    if toi.verbose:
        print(f"Initial transit fit plots saved to {out_dir}")

def plot_individual_transits(toi, xlim=0.3):
    '''
    Make a plot of each individual transit for each planet.
    '''
    out_dir = os.path.join(toi.model_dir, 'plotting')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    x = toi.cleaned_time.values
    y = toi.cleaned_flux.values

    # Loop over each planet
    for i, planet in enumerate(toi.transiting_planets.values()):
        
        save_dir = os.path.join(out_dir, f'individual_transits/planet_{planet.pl_letter}')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if toi.verbose:
            print(f"Plotting individual transits for {toi.name} {planet.pl_letter}")

        # Get the relevant phase-folded time array, transit inds, noise model, and transit model for this planet
        t0 = toi.map_soln["t0"][i]
        per = toi.map_soln["period"][i]
        x_fold = (x - t0 + 0.5 * per) % per - 0.5 * per
        x_fold = np.ascontiguousarray(x_fold, dtype=float)
        mask = np.abs(x_fold) < xlim
        transit_inds = np.argwhere(np.abs(np.ediff1d(x_fold[mask])) > 0.5)

        noise_model = toi.extras["gp_pred"] + toi.map_soln["mean_flux"]
        lc_model = toi.extras["light_curves"][:, i]

        def plot_transit(start_ind, end_ind, transit_num):
            '''
            Convenience function.
            '''
            if end_ind is not None:
                slice_obj = slice(start_ind, end_ind + 1)
            else:
                slice_obj = slice(start_ind, None)
            fig, ax = plt.subplots()
            ax.plot(x_fold[mask][slice_obj], y[mask][slice_obj], ".k", ms=4, label="Data")
            # Plot the binned flux in bins of 20 minutes
            bin_duration = 0.333 / 24 # 20 minutes in units of days
            bins = (np.max(x_fold[mask]) - np.min(x_fold[mask])) / bin_duration
            binned_flux, binned_edges, _ = binned_statistic(x_fold[mask][slice_obj], y[mask][slice_obj], statistic="mean", bins=bins)
            binned_edge_diff = np.ediff1d(binned_edges) / 2
            binned_locs = binned_edges[:-1] + binned_edge_diff
            ax.scatter(binned_locs, binned_flux, s=20, color='tomato', edgecolor='red')
            ax.plot(x_fold[mask][slice_obj], noise_model[mask][slice_obj], color="C2", alpha=0.5, ls='--', label="GP prediction + mean")
            ax.plot(x_fold[mask][slice_obj], lc_model[mask][slice_obj] +  noise_model[mask][slice_obj], color=planet.color, label='MAP transit model')

            ax.set_xlabel("Time since transit [days]")
            ax.set_ylabel("Relative flux [ppt]")
            ax.set_title(f"{toi.name} {planet.pl_letter} transit {transit_num}")

            ax.tick_params(axis='x', direction='in', which='both', top=False, bottom=True)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))

            ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))

            # Re-plot the data with the non-phase-folded time on the top x-axis so you know what transit this is.
            ax_top = ax.twiny()
            ax_top.plot(x[mask][slice_obj], y[mask][slice_obj], ".k", ms=4)
            ax_top.set_xlabel(f"Time [BJD - {toi.bjd_ref:.1f}]")
            ax_top.ticklabel_format(useOffset=False)
            ax_top.tick_params(axis='x', which='major', labelsize=10)
            
            ax_top.tick_params(axis='x', direction='in', which='both', top=True, bottom=False)
            ax_top.xaxis.set_major_locator(MultipleLocator(0.1))
            ax_top.xaxis.set_minor_locator(MultipleLocator(0.05))
            
            ax.legend(loc="lower left", fontsize=10, framealpha=0.5, fancybox=True)
            
            save_fname = f'{toi.name.replace(" ", "_")}_{planet.pl_letter}_transit_{transit_num}.png'
            fig.savefig(os.path.join(save_dir, save_fname), facecolor='white', bbox_inches='tight', dpi=300)
            plt.close()
        
        start_ind = 0
        transit_num = 1
        for end_ind in tqdm(transit_inds[:, 0]):
            plot_transit(start_ind, end_ind, transit_num)
            start_ind = end_ind + 1
            transit_num += 1
        plot_transit(start_ind, None, transit_num)
    
    if toi.verbose:
        print(f"Individual transit plots saved to {out_dir}/individual_transits/")

def plot_star_noise_corner(toi, df_derived_chains, overwrite=False):
    if toi.phot_gp_kernel == "exp_decay":
    
        # Corner plot for star properties and noise parameters
        star_labels = ['$\mu$ [ppt]', '$u_1$', '$u_2$']
        noise_labels = ['$\sigma_\mathrm{phot}$ [ppt]', '$\eta_\mathrm{GP,\:phot}$ [PPT]', r'$\rho_\mathrm{GP,\:phot}$ [d]', r'$\tau_\mathrm{GP,\:phot}$ [d]']
        star_noise_chains = np.vstack([
            df_derived_chains['mean_flux'], 
            df_derived_chains['u_0'],
            df_derived_chains['u_1'],
            np.exp(df_derived_chains['log_sigma_phot']),
            np.exp(df_derived_chains['log_sigma_phot_gp']), # Make this "eta" to differentiate the GP amplitudes from jitter terms.
            np.exp(df_derived_chains['log_rho_phot_gp']),
            np.exp(df_derived_chains['log_tau_phot_gp'])
        ]).T
        star_noise_corner = TesslaCornerPlot(toi, star_labels + noise_labels, star_noise_chains, toi.name)
        star_noise_corner.plot(overwrite=overwrite)
    else:
        # TODO: Fix? Or just leave it like this and people can make corner plots on their own if they use a different kernel.
        print("NOTE: Right now automated corner plot generation only works if phot_gp_kernel == 'exp_decay'")

def plot_svalue_gp_corner(toi, df_derived_chains, overwrite=False):
    if toi.svalue_gp_kernel == "exp_decay":
    
        # Corner plot for Svalue GP hyperparameters
        # GP amplitudes for each RV instrument
        noise_labels = [f'$\eta_{{\mathrm{{GP,\:RV,\:}}{tel}}}$ [m s$^{{-1}}$]' for tel in toi.rv_inst_names]
        chains = [np.exp(df_derived_chains[f'log_sigma_rv_gp_{tel}']) for tel in toi.rv_inst_names]
        
        # GP amplitudes and jitter for each Svalue instrument
        noise_labels += [f'$\eta_{{\mathrm{{GP,\:S_{{HK}},\:}}{tel}}}$ ' for tel in toi.svalue_inst_names]
        chains += [np.exp(df_derived_chains[f'log_sigma_svalue_gp_{tel}']) for tel in toi.svalue_inst_names]
        noise_labels += [f'$\sigma_{{S_\mathrm{{HK}},\:{tel}}}$ ' for tel in toi.svalue_inst_names]
        chains += [np.exp(df_derived_chains[f'log_jitter_svalue_gp_{tel}']) for tel in toi.svalue_inst_names]
        
        # Global GP hyperparameters
        noise_labels += [
            '$\mu_{S_\mathrm{HK}}$ ',
        ]
        chains += [
            df_derived_chains['gp_svalue_mean'],
        ]
        # Kernel-specific hyperparameters
        noise_labels += [  
            r'$\rho_{S_\mathrm{HK}}$ [d]',
            r'$\tau_{S_\mathrm{HK}}$ [d]'  
        ]
        chains += [
            np.exp(df_derived_chains['log_rho_rv_svalue_gp']),
            np.exp(df_derived_chains['log_tau_rv_svalue_gp'])
        ]

        svalue_gp_chains = np.vstack(chains).T
        svalue_gp_corner = TesslaCornerPlot(toi, noise_labels, svalue_gp_chains, toi.name + " RV-$S_\mathrm{HK}$ GP hyperparameters")
        svalue_gp_corner.plot(save_fname=f"{toi.name.replace(' ', '_')}_rv_svalue_gp_corner_plot", overwrite=overwrite)
    
    elif toi.svalue_gp_kernel == 'rotation':

        # Corner plot for Svalue GP hyperparameters
        # GP amplitudes for each RV instrument
        noise_labels = [f'$\eta_{{\mathrm{{GP,\:RV,\:}}\mathrm{{{tel}}}}}$ [m s$^{{-1}}$]' for tel in toi.rv_inst_names]
        chains = [df_derived_chains[f'sigma_rv_gp_{tel}'] for tel in toi.rv_inst_names]
        
        # GP amplitudes and jitter for each Svalue instrument
        noise_labels += [f'$\eta_{{\mathrm{{GP,\:S_{{HK}},\:}}\mathrm{{{tel}}}}}$ ' for tel in toi.svalue_inst_names]
        chains += [df_derived_chains[f'sigma_svalue_gp_{tel}'] for tel in toi.svalue_inst_names]
        noise_labels += ['$\sigma_{S_\mathrm{HK}}$ ']
        chains += [np.exp(df_derived_chains['log_jitter_svalue_HIRES'])]
        
        # Global GP hyperparameters
        noise_labels += [
            '$\mu_{S_\mathrm{HK}}$ ',
        ]
        chains += [
            df_derived_chains['gp_svalue_mean'],
        ]
        # Kernel-specific hyperparameters
        noise_labels += [  
            r'$P_\mathrm{rot}$ [d]',
            r'$Q_{0,\:S_\mathrm{HK}}$',
            r'$dQ_{S_\mathrm{HK}}$',
            r'$f_{S_\mathrm{HK}}$',
        ]
        chains += [
            df_derived_chains['prot_rv_svalue_gp'],
            np.exp(df_derived_chains['log_Q0_rv_svalue_gp']),
            np.exp(df_derived_chains['log_dQ_rv_svalue_gp']),
            df_derived_chains['f_rv_svalue_gp']
        ]
        
        svalue_gp_chains = np.vstack(chains).T
        svalue_gp_corner = TesslaCornerPlot(toi, noise_labels, svalue_gp_chains, toi.name + " RV-$S_\mathrm{HK}$ GP hyperparameters")
        svalue_gp_corner.plot(save_fname=f"{toi.name.replace(' ', '_')}_rv_svalue_gp_corner_plot", overwrite=overwrite)

    elif toi.svalue_gp_kernel == 'activity':
        
        # GP amplitudes for each RV instrument
        # Rotation term
        noise_labels = [f'$\eta_{{\mathrm{{GP,\:rot,\:RV,\:}}\mathrm{{{tel}}}}}$ [m s$^{{-1}}$]' for tel in toi.rv_inst_names]
        chains = [df_derived_chains[f'sigma_rv_gp_rot_{tel}'] for tel in toi.rv_inst_names]
        # Exp decay term
        noise_labels += [f'$\eta_{{\mathrm{{GP,\:dec,\:RV,\:}}\mathrm{{{tel}}}}}$ [m s$^{{-1}}$]' for tel in toi.rv_inst_names]
        chains += [np.exp(df_derived_chains[f'log_sigma_rv_gp_dec_{tel}']) for tel in toi.rv_inst_names]
        
        # GP amplitudes and jitter for each Svalue instrument
        # Rotation term
        noise_labels += [f'$\eta_{{\mathrm{{GP,\:rot,\:S_{{HK}},\:}}\mathrm{{{tel}}}}}$ ' for tel in toi.svalue_inst_names]
        chains += [df_derived_chains[f'sigma_svalue_gp_rot_{tel}'] for tel in toi.svalue_inst_names]
        # Exp decay term
        noise_labels += [f'$\eta_{{\mathrm{{GP,\:dec,\:S_{{HK}},\:}}\mathrm{{{tel}}}}}$ ' for tel in toi.svalue_inst_names]
        chains += [np.exp(df_derived_chains[f'log_sigma_svalue_gp_dec_{tel}']) for tel in toi.svalue_inst_names]

        # Svalue jitter and mean
        noise_labels += ['$\sigma_{S_\mathrm{HK}}$ ']
        chains += [np.exp(df_derived_chains['log_jitter_svalue_HIRES'])]
        noise_labels += ['$\mu_{S_\mathrm{HK}}$ ']
        chains += [df_derived_chains['gp_svalue_mean']]

        # Kernel-specific hyperparameters
        # Rotation
        noise_labels += [  
            r'$P_\mathrm{rot}$ [d]',
            r'$Q_{0,\:S_\mathrm{HK}}$',
            r'$dQ_{S_\mathrm{HK}}$',
            r'$f_{S_\mathrm{HK}}$',
        ]
        chains += [
            df_derived_chains['prot_rv_svalue_gp'],
            np.exp(df_derived_chains['log_Q0_rv_svalue_gp']),
            np.exp(df_derived_chains['log_dQ_rv_svalue_gp']),
            df_derived_chains['f_rv_svalue_gp']
        ]

        # Kernel-specific hyperparameters
        # Exp decay
        noise_labels += [  
            r'$\rho_{S_\mathrm{HK}}$ [d]',
        ]
        chains += [
            np.exp(df_derived_chains['log_rho_rv_svalue_gp']),
        ]
        
        svalue_gp_chains = np.vstack(chains).T
        svalue_gp_corner = TesslaCornerPlot(toi, noise_labels, svalue_gp_chains, toi.name + " RV-$S_\mathrm{HK}$ GP hyperparameters")
        svalue_gp_corner.plot(save_fname=f"{toi.name.replace(' ', '_')}_rv_svalue_gp_corner_plot", overwrite=overwrite)

    else:
        # TODO: Fix? Or just leave it like this and people can make corner plots on their own if they use a different kernel.
        print("NOTE: Right now automated corner plot generation only works if svalue_gp_kernel == 'exp_decay' or  svalue_gp_kernel == 'rotation' or svalue_gp_kernel == 'activity'.")

def plot_phot_only_corners(toi, df_derived_chains, overwrite=False):
    '''
    Make the corner plots!
    '''
    plot_star_noise_corner(toi, df_derived_chains, overwrite=overwrite)

    # Corner plot for each planet's transit parameters
    for i,letter in enumerate(toi.transiting_planets.keys()):
        planet_labels = [
            '$P$ [d]',
            '$T_\mathrm{c}$ [BTJD]',
            '$R_\mathrm{p}/R_*$ [$\%$]',
            '$b$',
            '$T_\mathrm{dur}$ [hr]'
        ]
        planet_chains = np.vstack([
            df_derived_chains[f'period_{letter}'],
            df_derived_chains[f't0_BTJD_{letter}'],
            df_derived_chains[f'ror_{letter}'] * 100, # Put in units of percent to make the decimals easier to see
            df_derived_chains[f'b_{letter}'],
            df_derived_chains[f'dur_hr_{letter}']
        ]).T
        planet_corner = TesslaCornerPlot(toi, planet_labels, planet_chains, f"{toi.name} {letter} measured parameters", color=toi.transiting_planets[letter].color)
        planet_corner.plot(overwrite=overwrite)

        # Corner plot for derived planet parameters
        derived_labels = [
            '$R_\mathrm{p}$ [$R_\mathrm{\oplus}$]',
            '$e$',
            '$\omega$ [$^{\circ}$]'
        ]
        derived_chains = np.vstack([
            df_derived_chains[f"rp_{letter}"],
            df_derived_chains[f"ecc_{letter}"],
            df_derived_chains[f"omega_folded_deg_{letter}"]
        ]).T
        derived_corner = TesslaCornerPlot(toi, derived_labels, derived_chains, f"{toi.name} {letter} derived parameters", color=toi.transiting_planets[letter].color)
        derived_corner.plot(overwrite=overwrite)

def plot_joint_corners(toi, df_derived_chains, overwrite=False):
    '''
    Make the corner plots!
    '''
    plot_star_noise_corner(toi, df_derived_chains, overwrite=overwrite)
    
    if toi.include_svalue_gp:
        plot_svalue_gp_corner(toi, df_derived_chains, overwrite=overwrite)

    # Corner plot for instrument parameters
    rv_trend_labels_dict = {
        0:"$\dot{\gamma}$ [m s$^{-1}$ d$^{-1}$]",
        1:"$\ddot{\gamma}$ [m s$^{-1}$ d$^{-2}$]",
    }

    rv_inst_labels = []
    rv_inst_chains_list = []
    for tel in toi.rv_inst_names:
        rv_inst_labels.append(f'$\gamma_\mathrm{{{tel}}}$ [m s$^{{-1}}$]')
        rv_inst_chains_list.append(df_derived_chains[f'gamma_rv_{tel}'])
        rv_inst_labels.append(f'$\sigma_\mathrm{{{tel}}}$ [m s$^{{-1}}$]')
        rv_inst_chains_list.append(np.exp(df_derived_chains[f'log_sigma_rv_{tel}']))
    if toi.rv_trend:
        for i in range(0, toi.rv_trend_order):
            try:
                rv_inst_labels.append(rv_trend_labels_dict[i])
                rv_inst_chains_list.append(df_derived_chains[f'trend_rv_{i}'])
            except KeyError:
                print("Trend value not in rv_trend_labels_dict.")
                continue
    rv_inst_chains = np.vstack(rv_inst_chains_list).T
    rv_inst_corner = TesslaCornerPlot(toi, rv_inst_labels, rv_inst_chains, f"{toi.name} RV instrument parameters")
    rv_inst_corner.plot(overwrite=overwrite)

    # Corner plot for the measured parameters
    for letter, planet in toi.planets.items():
        if planet.is_transiting:
            if toi.force_circular_orbits_for_transiting_planets:
                planet_labels = [
                    '$P$ [d]',
                    '$T_\mathrm{c}$ [BTJD]',
                    '$R_\mathrm{p}/R_*$ [$\%$]',
                    '$b$',
                    '$K$ [m s$^{-1}$]'
                ]
                planet_chains = np.vstack([
                    df_derived_chains[f'period_{letter}'],
                    df_derived_chains[f't0_BTJD_{letter}'],
                    df_derived_chains[f'ror_{letter}'] * 100, # Put in units of percent to make the decimals easier to see
                    df_derived_chains[f'b_{letter}'],
                    df_derived_chains[f'K_{letter}']
                ]).T
            else:
                planet_labels = [
                    '$P$ [d]',
                    '$T_\mathrm{c}$ [BTJD]',
                    '$R_\mathrm{p}/R_*$ [$\%$]',
                    '$b$',
                    '$e$',
                    '$\omega$ [Rad]',
                    '$K$ [m s$^{-1}$]'
                ]
                planet_chains = np.vstack([
                    df_derived_chains[f'period_{letter}'],
                    df_derived_chains[f't0_BTJD_{letter}'],
                    df_derived_chains[f'ror_{letter}'] * 100, # Put in units of percent to make the decimals easier to see
                    df_derived_chains[f'b_{letter}'],
                    df_derived_chains[f'ecc_{letter}'],
                    df_derived_chains[f'omega_{letter}'],
                    df_derived_chains[f'K_{letter}']
                ]).T

            planet_corner = TesslaCornerPlot(toi, planet_labels, planet_chains, f"{toi.name} {letter} measured parameters", color=planet.color)
            planet_corner.plot(overwrite=overwrite)

            # Corner plot for derived planet parameters
            derived_labels = [
                '$R_\mathrm{p}$ [$R_\mathrm{\oplus}$]',
                '$M_\mathrm{p}$ [$M_\mathrm{\oplus}$]',
                '$\\rho_\mathrm{p}$ [g cm$^{-1}$]',
                '$a$ [AU]',
                '$T_\mathrm{eq}$ [K]'
            ]
            derived_chains = np.vstack([
                df_derived_chains[f"rp_{letter}"],
                df_derived_chains[f"mp_{letter}"],
                df_derived_chains[f"rho_{letter}"],
                df_derived_chains[f"a_{letter}"],
                df_derived_chains[f"teq_{letter}"],
            ]).T
            derived_corner = TesslaCornerPlot(toi, derived_labels, derived_chains, f"{toi.name} {letter} derived parameters", color=planet.color)
            derived_corner.plot(overwrite=overwrite)

        else:
            prefix = 'nontrans_'
            planet_labels = [
                '$P$ [d]',
                '$T_\mathrm{c}$ [BTJD]',
                '$e$',
                '$\omega$ [Rad]',
                '$K$ [m s$^{-1}$]'
            ]
            planet_chains = np.vstack([
                df_derived_chains[f'{prefix}period_{letter}'],
                df_derived_chains[f'{prefix}t0_BTJD_{letter}'],
                df_derived_chains[f'{prefix}ecc_{letter}'],
                df_derived_chains[f'{prefix}omega_{letter}'],
                df_derived_chains[f'{prefix}K_{letter}']
            ]).T
            planet_corner = TesslaCornerPlot(toi, planet_labels, planet_chains, f"{toi.name} {letter} measured parameters", color=planet.color)
            planet_corner.plot(overwrite=overwrite)

            # Corner plot for derived planet parameters
            derived_labels = [
                '$M_\mathrm{p} \sin i$ [$M_\mathrm{\oplus}$]',
                '$a$ [AU]',
                '$T_\mathrm{eq}$ [K]'
            ]
            derived_chains = np.vstack([
                df_derived_chains[f"{prefix}msini_{letter}"],
                df_derived_chains[f"{prefix}a_{letter}"],
                df_derived_chains[f"{prefix}teq_{letter}"],
            ]).T
            derived_corner = TesslaCornerPlot(toi, derived_labels, derived_chains, f"{toi.name} {letter} derived parameters", color=planet.color)
            derived_corner.plot(overwrite=overwrite)

def phase_plot(out_dir, title, ylabel, x, y, per, t0, **kwargs):
    '''
    Plot the data folded to the period specified. For e.g. folding the light curve at the stellar rotation period.
    '''
    fig, ax = plt.subplots()

    x_phase = ((x - t0) % per) / per
    
    ax.plot(x_phase, y, '.k', alpha=0.3, **kwargs)
    # Plot the binned data as well
    binned_y, binned_edges, _ = binned_statistic(x_phase, y, statistic="mean", bins=12) # Number of bins in phase
    binned_edge_diff = np.ediff1d(binned_edges) / 2
    binned_locs = binned_edges[:-1] + binned_edge_diff
    ax.scatter(binned_locs, binned_y, s=20, color='tomato', edgecolor='red', zorder=1000)
    ax.text(0.7, 0.9, f"$P =$ {per:.2f} d", ha='left', va='center', transform=ax.transAxes)
    ax.set_xlabel('Phase')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    save_fname = f'{title.replace(" ", "_")}_phase_plot.png'
    fig.savefig(os.path.join(out_dir, save_fname), facecolor='white', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Phase plot saved to {out_dir}/{save_fname}")

def add_ymd_label(bjd_ref, fig, ax, xlims, left_or_right):
    '''
    Add a ymd label to the top panel left-most and right-most chunks.
    '''
    fig.canvas.draw() # Needed in order to get back the ticklabels otherwise they'll be empty

    ax_yrs = ax.secondary_xaxis('top')
    ax_yrs.set_xticks(ax.get_xticks())
    ax_yrs_ticks = ax.get_xticks()
    ax_yrs_ticklabels = ax.get_xticklabels()
    tick_mask = (ax_yrs_ticks > xlims[0]) & (ax_yrs_ticks < xlims[-1])
    ax_yrs_ticks = ax_yrs_ticks[tick_mask]
    ax_yrs_ticklabels = [label.get_text() for i,label in enumerate(ax_yrs_ticklabels) if tick_mask[i]]

    bjd_date = None
    ind = None
    if left_or_right == 'left':
        ind = 0
        bjd_date = ax_yrs_ticks[ind] + bjd_ref
    elif left_or_right == 'right':
        ind = -1
        bjd_date = ax_yrs_ticks[ind] + bjd_ref
    else:
        assert False, "Choose left or right chunk."
    ymd_date = Time(bjd_date, format='jd', scale='utc').ymdhms
    month_number = ymd_date['month'] # Next few lines get the abbreviation for the month's name from the month's number
    datetime_obj = datetime.datetime.strptime(f"{month_number}", "%m")
    month_name = datetime_obj.strftime("%b")
    day = ymd_date['day']
    year = ymd_date['year']
    
    ax_yrs.set_xticks(ax_yrs_ticks)
    ax_yrs_ticklabels = [''] * len(ax_yrs_ticklabels)
    ax_yrs_ticklabels[ind] = f"{year}-{month_name}-{day}"
    ax_yrs.set_xticklabels(ax_yrs_ticklabels)

    # Make the ticks themselves invisible 
    ax_yrs.tick_params(axis="x", top=False, bottom=False, pad=1.0)