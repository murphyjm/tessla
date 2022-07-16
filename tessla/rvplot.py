import os
from re import I
import warnings
from tqdm import tqdm

import pandas as pd
import numpy as np
from math import ceil
from scipy.stats import binned_statistic
from astropy import units

# Exoplanet stuff
import exoplanet as xo
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing" # Not exactly sure what this flag change does.

# Plotting stuff
from matplotlib import rcParams
rcParams["font.size"] = 16
rcParams["figure.constrained_layout.use"] = False
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

from tessla.plotting_utils import add_ymd_label

DEFAULT_MARKER_MAPPER = {
    'j':{
        'color':'black',
        'marker':'o',
        'label':'HIRES'
    },
    'hires_j':{
        'color':'black',
        'marker':'o',
        'label':'HIRES'
    },
    'HIRES':{
        'color':'black',
        'marker':'o',
        'label':'HIRES'
    },
    'apf':{
        'color':'green',
        'marker':'d',
        'label':'APF'
    },
    'APF':{
        'color':'green',
        'marker':'d',
        'label':'APF'
    }
}

class RVPlot:
    '''
    Object for making a plot of the RVs and RV model. 
    '''
    def __init__(self, 
                toi,
                figwidth=12,
                ylabelpad=10,
                plot_random_orbit_draws=False, # If true, plot random realizations of the phase-folded RV curve using the posteriors of the model fit.
                num_random_orbit_draws=25, # Number of random draws to plot.
                save_format='.png',
                save_dpi=400,
                df_summary_fname=None,
                tel_marker_mapper=None,
                rms_yscale_phase_folded_panels=True,
                rms_yscale_phase_folded_panels_scale=3,
                param_fontsize=14, # Fontsize for annotating the phase folded plots
                timeseries_phase_hspace=0.04,
                ) -> None:
    
        self.toi = toi
        self.figwidth = figwidth
        self.ylabelpad = ylabelpad
        self.plot_random_orbit_draws = plot_random_orbit_draws
        self.num_random_orbit_draws = num_random_orbit_draws
        self.save_format = save_format
        self.save_dpi = save_dpi
        self.df_summary_fname = df_summary_fname
        self.df_summary = None
        if df_summary_fname is not None:
            self.df_summary = pd.read_csv(df_summary_fname, index_col=0)

        if tel_marker_mapper is None:
            self.tel_marker_mapper = DEFAULT_MARKER_MAPPER
        else:
            for tel in self.toi.rv_inst_names:
                assert tel in tel_marker_mapper.keys(), "RV instrument not found in tel_marker_mapper dictionary."
            self.tel_marker_mapper = tel_marker_mapper
        
        self.rms_yscale_phase_folded_panels = rms_yscale_phase_folded_panels # If true, set the Y axis limits for the phase-folded panels based on the residuals rms.
        self.rms_yscale_phase_folded_panels_scale = rms_yscale_phase_folded_panels_scale
        self.param_fontsize = param_fontsize
        self.timeseries_phase_hspace = timeseries_phase_hspace

    def plot(self, save_fname=None, overwrite=False, save_and_close=True, return_fig_obj=False):
        '''
        Make the plot!
        '''
        if self.toi.verbose:
            print("Creating RV plot...")
        
        out_dir = os.path.join(self.toi.model_dir, 'plotting')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # Save fname housekeeping and overwrite handling.
        if save_fname is None:
            default_save_fname = f"{self.toi.name.replace(' ', '_')}_rv_model"
            save_fname = os.path.join(out_dir, default_save_fname + self.save_format)
        else:
            save_fname = os.path.join(out_dir, save_fname + self.save_format)
        if not overwrite and os.path.isfile(save_fname):
            warnings.warn("Exiting before plotting to avoid overwriting exisiting plot file.")
            return None
        
        fig = self.__rv_plot()
        
        # Option to either save and close the figure right away or return it for further investigation.
        if save_and_close:
            # Save the figure!
            fig.savefig(save_fname, facecolor='white', bbox_inches='tight', dpi=self.save_dpi)
            print(f"RV model plot saved to {save_fname}")
            plt.close()
        elif return_fig_obj:
            return fig
    
    def __get_ytick_spacing(self):
        '''
        Hacky.
        '''
        yspan = np.max(self.toi.rv_df.mnvel) - np.min(self.toi.rv_df.mnvel)
        major = 5
        minor = 2.5
        if yspan >= 50 and yspan < 100:
            major = 10
            minor = 5
        elif yspan >= 100 and yspan < 200:
            major = 20
            minor = 10
        elif yspan >= 200:
            major = 50
            minor = 25
        return major, minor

    def __get_residuals_ytick_spacing(self, residuals):
        '''
        Hacky.
        '''
        yspan = np.max(residuals) - np.min(residuals)
        major = 10
        minor = 5
        if yspan >= 25 and yspan < 35:
            major = 15
            minor = 7.5
        elif yspan >= 35:
            major = 20
            minor = 10
        elif yspan >= 45:
            major = 50
            minor = 25
        return major, minor
    
    def __get_ytick_phase_spacing(self, kamp):
        major = 5
        minor = 2.5
        if kamp >= 10 and kamp < 50:
            major = 10
            minor = 5
        elif kamp >= 50 and kamp < 100:
            major = 30
            minor = 15
        elif kamp >= 100:
            major = 50
            minor = 25
        return major, minor

    def __rv_plot(self):
        '''
        '''
        figheight = 14 # Default fig height
        timeseries_height = 3 * figheight / 5
        num_planet_rows = ceil(self.toi.n_planets / 2)
        planet_row_height = 2 * figheight / 5
        if num_planet_rows > 1:
            figheight = figheight + (num_planet_rows - 1) * planet_row_height

        # Create the figure object
        fig = plt.figure(figsize=(self.figwidth, figheight))

        # Create the GridSpec objects for the timeseries panel and its residuals
        heights = [1, 0.25]
        gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=heights, hspace=0.05)
        timeseries_bottom = 1 - timeseries_height / figheight
        gs0.update(bottom=timeseries_bottom + self.timeseries_phase_hspace*0.5)
        sps1, sps2 = gs0
        
        ################################################################################################
        ############################### TOP PANEL: RVs and full RV model ###############################
        ################################################################################################
        ax1 = fig.add_subplot(sps1)
        ax1.set_title(self.toi.name, pad=0)
        
        # Plot the data
        for tel in self.toi.rv_inst_names:
            mask = self.toi.rv_df.tel.values == tel
            ax1.errorbar(self.toi.rv_df.time[mask], 
                        self.toi.rv_df.mnvel[mask] - self.toi.extras['mean_rv'][mask], 
                        self.toi.extras['err_rv'][mask], fmt='.', **self.tel_marker_mapper[tel])
        
        # Plot the underlying trend, if any
        if self.toi.rv_trend:
            ax1.plot(self.toi.t_rv, self.toi.extras['bkg_rv_pred'], color="#aaaaaa", ls='--', lw=2, zorder=-1)

        # Plot the RV model
        if not self.toi.include_svalue_gp:
            ax1.plot(self.toi.t_rv, self.toi.extras['full_rv_model_pred'], color="blue", lw=3)
        else:
            for tel in self.toi.rv_inst_names:
                full_mod_pred = self.toi.extras['full_rv_model_pred'] + self.toi.extras[f'gp_rv_pred_{tel}']
                # Plot the GP error envelope and the solution
                ax1.fill_between(self.toi.t_rv, full_mod_pred + self.toi.extras[f'gp_rv_pred_stdv_{tel}'], full_mod_pred - self.toi.extras[f'gp_rv_pred_stdv_{tel}'], 
                                    alpha=0.3, 
                                    color=self.tel_marker_mapper[tel]['color'])
                ax1.plot(self.toi.t_rv, full_mod_pred, color='blue', lw=1)

        # Add label for years to the upper axis
        add_ymd_label(self.toi.bjd_ref, fig, ax1, (np.min(self.toi.rv_df.time), np.max(self.toi.rv_df.time)), 'left')
        add_ymd_label(self.toi.bjd_ref, fig, ax1, (np.min(self.toi.rv_df.time), np.max(self.toi.rv_df.time)), 'right')
        
        # Top panel housekeeping
        ax1.set_ylabel("RV [m s$^{-1}$]", fontsize=14, labelpad=self.ylabelpad)
        major, minor = self.__get_ytick_spacing()
        ax1.yaxis.set_major_locator(MultipleLocator(major))
        ax1.yaxis.set_minor_locator(MultipleLocator(minor))
        ax1.legend(fontsize=14, loc='upper right')

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ################################## BOTTOM PANEL: Residuals #####################################
        ################################################################################################
        ax2 = fig.add_subplot(sps2, sharex=ax1)

        # Plot the residuals about the full model
        residuals = pd.Series(self.toi.rv_df.mnvel.values - self.toi.extras['full_rv_model'] - self.toi.extras['mean_rv']).copy()
        if self.toi.include_svalue_gp:
            for tel in self.toi.rv_inst_names:
                mask = self.toi.rv_df.tel.values == tel
                residuals.loc[mask] -= self.toi.extras[f"gp_rv_{tel}"]

        for tel in self.toi.rv_inst_names:
            mask = self.toi.rv_df.tel.values == tel
            ax2.errorbar(self.toi.rv_df.time[mask], 
                        residuals[mask], 
                        self.toi.extras['err_rv'][mask], fmt='.', **self.tel_marker_mapper[tel])
        ax2.axhline(0, color="#aaaaaa", lw=1)

        # Plot housekeeping
        ax2.set_ylabel("Residuals", fontsize=14, labelpad=self.ylabelpad)
        ax2.set_xlabel(f"Time [BJD - {self.toi.bjd_ref:.1f}]", fontsize=14)
        major, minor = self.__get_residuals_ytick_spacing(residuals)
        ax2.yaxis.set_major_locator(MultipleLocator(major))
        ax2.yaxis.set_minor_locator(MultipleLocator(minor))
        bottom = -1 * np.max(ax2.get_ylim())
        ax2.set_ylim(bottom=bottom)

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ############################# HOUSEKEEPING FOR TOP TWO PANELS ##################################
        ################################################################################################

        # Make ticks go inward and set multiple locator for x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(50))
            ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)
            ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ################################################################################################
        ############################ PHASE-FOLDED TRANSIT AND RESIDUALS ################################
        ################################################################################################
        fig = self.__plot_phase_folded_orbits(fig, timeseries_bottom)
        
        fig.align_ylabels()
        return fig

    def __plot_phase_folded_orbits(self, fig, timeseries_bottom):
        '''
        Plot the folded transits for each planet.
        '''
        residuals = pd.Series(self.toi.rv_df.mnvel.values - self.toi.extras['full_rv_model'] - self.toi.extras['mean_rv']).copy()
        if self.toi.include_svalue_gp:
            for tel in self.toi.rv_inst_names:
                mask = self.toi.rv_df.tel.values == tel
                residuals.loc[mask] -= self.toi.extras[f"gp_rv_{tel}"]

        chains = None
        if self.plot_random_orbit_draws:
            chains = pd.read_csv(self.toi.chains_path)
        phase_folded_axes = []
        phase_folded_resid_axes = []
        
        # Set up the outer gridspec
        num_planet_rows = ceil(self.toi.n_planets / 2)
        outer_gs = gridspec.GridSpec(num_planet_rows, 2, figure=fig, hspace=0.2)
        outer_gs.update(top=timeseries_bottom - self.timeseries_phase_hspace*0.5)

        for k in range(num_planet_rows):
            # Nested gridspec objects. One for each row of the phased plots. 
            # See https://stackoverflow.com/questions/31484273/spacing-between-some-subplots-but-not-all
            heights = [1, 0.25] # Heigh ratio between phase plot and residuals
            sps = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[k, :], height_ratios=heights, hspace=0.05)

            # Figure out how many columns in this row
            planets_left = self.toi.n_planets - 2 * k
            if planets_left > 1:
                num_cols = 2
            else:
                num_cols = 1
            planet_start_ind = 2 * k
            planet_end_ind = planet_start_ind + num_cols
            planet_ind = planet_start_ind

            for planet in list(self.toi.planets.values())[planet_start_ind:planet_end_ind]:
                
                # Get the correct column.
                if planet_ind % 2 == 0 and planet_ind < self.toi.n_planets - 1:
                    planet_col_ind = 0
                elif planet_ind % 2 == 0 and planet_ind == self.toi.n_planets - 1:
                    planet_col_ind = slice(None)
                else:
                    planet_col_ind = 1
                
                ax0 = fig.add_subplot(sps[0, planet_col_ind])
                phase_folded_axes.append(ax0)

                # Plot the folded data
                x_fold = (self.toi.rv_df.time - planet.t0 + 0.5 * planet.per) % planet.per - 0.5 * planet.per
                x_fold /= planet.per # Put this in unitless phase
                
                # RV contribution from background trend and other planets and GP
                other_rv = pd.Series(self.toi.extras['bkg_rv']).copy()
                if len(self.toi.extras['planet_rv'].shape) > 1:
                    other_rv += np.sum(np.delete(self.toi.extras['planet_rv'], planet_ind, axis=1), axis=1)
                if self.toi.include_svalue_gp:
                    for tel in self.toi.rv_inst_names:
                        mask = self.toi.rv_df.tel.values == tel
                        other_rv.loc[mask] += self.toi.extras[f'gp_rv_{tel}']
                
                # Plot the data
                for tel in self.toi.rv_inst_names:
                    mask = self.toi.rv_df.tel.values == tel
                    ax0.errorbar(x_fold[mask], 
                                self.toi.rv_df.mnvel[mask] - self.toi.extras['mean_rv'][mask] - other_rv[mask], 
                                self.toi.extras['err_rv'][mask], fmt='.', **self.tel_marker_mapper[tel])
                # Plot the phase-folded MAP solution
                x_fold_pred = (self.toi.t_rv - planet.t0 + 0.5 * planet.per) % planet.per - 0.5 * planet.per
                x_fold_pred /= planet.per # Put this in unitless phase
                inds_pred = np.argsort(x_fold_pred)
                if len(self.toi.extras['planet_rv_pred'].shape) > 1:
                    ax0.plot(x_fold_pred[inds_pred], self.toi.extras['planet_rv_pred'][:, planet_ind][inds_pred], color=planet.color, zorder=1000, lw=3)
                else:
                    ax0.plot(x_fold_pred[inds_pred], self.toi.extras['planet_rv_pred'][inds_pred], color=planet.color, zorder=1000, lw=3)

                # Plot the binned RVs like in RadVel
                bin_duration = 0.125 # 1/8 bins of phase
                bins = (x_fold.max() - x_fold.min()) / bin_duration
                binned_rv, binned_edges, _ = binned_statistic(x_fold, self.toi.rv_df.mnvel - self.toi.extras['mean_rv'] - other_rv, statistic="mean", bins=bins)
                binned_rv_var, __, ___ = binned_statistic(x_fold, (self.toi.extras['err_rv'])**2, statistic="mean", bins=bins)
                binned_rv_err = np.sqrt(binned_rv_var)
                binned_edge_diff = np.ediff1d(binned_edges) / 2
                binned_locs = binned_edges[:-1] + binned_edge_diff
                ax0.errorbar(binned_locs, binned_rv, binned_rv_err, fmt='.', marker='o', color='tomato', mec='red', zorder=1001, label='Binned RV')

                if self.plot_random_orbit_draws:
                    if self.toi.verbose:
                        print(f"Plotting {self.num_random_orbit_draws} random draws of phase-folded RV orbit for planet {planet.pl_letter}...")
                    for j in tqdm(range(self.num_random_orbit_draws)): # Could optionally use the "disable" keyword argument to only use the progress if self.toi.verbose == True.
                        
                        ind = np.random.choice(np.arange(self.num_random_orbit_draws))

                        # Build the model we used before
                        # Orbit
                        if planet.is_transiting:
                            prefix = ''
                        else:
                            prefix = 'nontrans_'
                        K = chains[f"{prefix}K_{planet.pl_letter}"].values[ind]
                        period = chains[f"{prefix}period_{planet.pl_letter}"].values[ind]
                        t0 = chains[f"{prefix}t0_{planet.pl_letter}"].values[ind]
                        rstar = chains["rstar"].values[ind]
                        mstar = chains["mstar"].values[ind]
                        if self.toi.force_circular_orbits_for_transiting_planets:
                            ecc = None
                            omega = None
                        else:
                            ecc = chains[f"{prefix}ecc_{planet.pl_letter}"].values[ind]
                            omega = chains[f"{prefix}omega_{planet.pl_letter}"].values[ind]
                        orbit = xo.orbits.KeplerianOrbit(r_star=rstar, m_star=mstar, period=period, t0=t0, ecc=ecc, omega=omega)

                        # Get RV for planet
                        planet_rv_pred = orbit.get_radial_velocity(self.toi.t_rv, K=K).eval()
                        
                        # Plot the random draw
                        ax0.plot(x_fold_pred[inds_pred], planet_rv_pred[inds_pred], color=planet.color, alpha=0.3, zorder=999)

                # Plot the residuals below
                ax1 = fig.add_subplot(sps[1, planet_col_ind], sharex=ax0)
                phase_folded_resid_axes.append(ax1)
                for tel in self.toi.rv_inst_names:
                    mask = self.toi.rv_df.tel.values == tel
                    ax1.errorbar(x_fold[mask],
                                residuals[mask], 
                                self.toi.extras['err_rv'][mask], fmt='.', **self.tel_marker_mapper[tel])
                ax1.axhline(0, color="#aaaaaa", lw=1)

                # Plot housekeeping
                try:
                    planet_name = planet.alt_name
                except AttributeError:
                    planet_name = planet.pl_letter
                ax0.set_title(f"{self.toi.name} {planet_name}")
                for ax in [ax0, ax1]:
                    ax.xaxis.set_major_locator(MultipleLocator(0.25))
                    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                    ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)
                    ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)
                plt.setp(ax0.get_xticklabels(), visible=False)

                if k == num_planet_rows - 1: # Only add the xlabel to the bottom row
                    ax1.set_xlabel("Phase", fontsize=14)
                major, minor = self.__get_ytick_phase_spacing(planet.kamp)
                ax0.yaxis.set_major_locator(MultipleLocator(major))
                ax0.yaxis.set_minor_locator(MultipleLocator(minor))
                major, minor = self.__get_residuals_ytick_spacing(residuals)
                ax1.yaxis.set_major_locator(MultipleLocator(major))
                ax1.yaxis.set_minor_locator(MultipleLocator(minor))

                if planet_col_ind == 0 or isinstance(planet_col_ind, slice):
                    ax0.set_ylabel("RV [m s$^{-1}$]", fontsize=14)
                    ax1.set_ylabel("Resid.", fontsize=14)
                
                # Label phase plots with orbit parameters
                prefix = ''
                if not planet.is_transiting:
                    prefix = "nontrans_"
                # Include errors
                if self.df_summary is not None:
                    per_med = self.df_summary.loc[f'{prefix}period_{planet.pl_letter}', 'median']
                    per_err = self.df_summary.loc[f'{prefix}period_{planet.pl_letter}', 'std']
                    if per_err < 1:
                        per_str = f"$P =$ {per_med:.2f} $\pm$ {per_err:.2e} d"
                    else:
                        per_str = f"$P =$ {per_med:.1f} $\pm$ {per_err:.1f} d"
                    if self.toi.force_circular_orbits_for_transiting_planets:
                        ecc_str = "$e \equiv 0$"
                    else:
                        ecc_med = self.df_summary.loc[f'{prefix}ecc_{planet.pl_letter}', 'median']
                        ecc_err = self.df_summary.loc[f'{prefix}ecc_{planet.pl_letter}', 'std']
                        ecc_str = f"$e = {ecc_med:.2f} \pm {ecc_err:.2f}$"

                    kamp_med = self.df_summary.loc[f'{prefix}K_{planet.pl_letter}', 'median']
                    kamp_err = self.df_summary.loc[f'{prefix}K_{planet.pl_letter}', 'std']
                    kamp_str = f"$K = {kamp_med:.2f} \pm {kamp_err:.2f}$ m s$^{{-1}}$"

                    if planet.is_transiting:
                        mp_med = self.df_summary.loc[f'mp_{planet.pl_letter}', 'median']
                        mp_err = self.df_summary.loc[f'mp_{planet.pl_letter}', 'std']
                        mp_str = '\n' + f"$M_\mathrm{{p}} = {mp_med:.1f} \pm {mp_err:.1f}$ $M_\oplus$"
                        if mp_med > 200: # If mp > 200 Earth masses, put in Jupiter masses
                            mp_map_med = units.Mearth.to(units.Mjup, mp_med)
                            mp_map_err = units.Mearth.to(units.Mjup, mp_err)
                            mp_str = '\n' + f"$M_\mathrm{{p}} = {mp_map_med:.1f} \pm {mp_map_err:.1f}$ $M_\mathrm{{Jup}}$"
                    else:
                        mp_med = self.df_summary.loc[f'{prefix}msini_{planet.pl_letter}', 'median']
                        mp_err = self.df_summary.loc[f'{prefix}msini_{planet.pl_letter}', 'std']
                        mp_str = '\n' + f"$M_\mathrm{{p}} \sin i = {mp_med:.1f} \pm {mp_err:.1f}$ $M_\oplus$"
                        if mp_med > 200: # If mp sini > 200 Earth masses, put in Jupiter masses
                            mp_map_med = units.Mearth.to(units.Mjup, mp_med)
                            mp_map_err = units.Mearth.to(units.Mjup, mp_err)
                            mp_str = '\n' + f"$M_\mathrm{{p}} \sin i = {mp_map_med:.1f} \pm {mp_map_err:.1f}$ $M_\mathrm{{Jup}}$"
                else:
                    per_str = f"$P =$ {planet.per:.2f} d"
                    if self.toi.force_circular_orbits_for_transiting_planets:
                        ecc_str = "$e \equiv 0$"
                    else:
                        ecc_str = f"$e =$ {planet.ecc:.2f}"
                    kamp_str = f"$K =$ {planet.kamp:.2f} m s$^{{-1}}$"
                    mp_str = ''

                planet_str = per_str + '\n' + ecc_str + '\n' + kamp_str + mp_str
                text = ax0.text(0.05, 0.05, planet_str, ha='left', va='bottom', transform=ax0.transAxes, fontsize=self.param_fontsize, zorder=1002)
                alpha = 0.5
                if self.rms_yscale_phase_folded_panels:
                    alpha=0.8
                text.set_bbox(dict(facecolor='white', alpha=alpha, edgecolor='none'))
            
                # Move on to the next planet!
                planet_ind += 1
        
        # Make the y-axes range the same for all of the phase-folded plots IF the planets have reasonably similar K-amplitudes. Otherwise let them be on their own scales
        kamp_planet_b = self.toi.planets['b'].kamp
        kamps_all_planets = np.array([planet.kamp for planet in self.toi.planets.values()])
        kamps_all_planets_minus_b = np.abs(kamps_all_planets - kamp_planet_b)
        if all(kamps_all_planets_minus_b < 7.5): # If all the planets' k-amplitude are within 7.5 m/s of eachother, share y-axis limits
            for k, axes in enumerate([phase_folded_axes, phase_folded_resid_axes]):
                y_phase_max = np.max([max(ax.get_ylim()) for ax in axes])
                y_phase_min = np.min([min(ax.get_ylim()) for ax in axes])
                y_phase_lim = (y_phase_min, y_phase_max)
                if k == 0 and self.rms_yscale_phase_folded_panels:
                    limit = self.rms_yscale_phase_folded_panels * np.std(residuals)
                    if limit < kamp_planet_b: # HACK
                        limit = kamp_planet_b * 1.5
                    y_phase_lim = (-limit, limit)
                for i in range(len(axes)):
                    axes[i].set_ylim(y_phase_lim)
        elif self.rms_yscale_phase_folded_panels: # HACK
            for k, ax in enumerate(phase_folded_axes):
                kamp_current_planet = kamps_all_planets[k]
                if kamp_current_planet < 10:
                    y_phase_lim = (-self.rms_yscale_phase_folded_panels * np.std(residuals), self.rms_yscale_phase_folded_panels * np.std(residuals))
                    ax.set_ylim(y_phase_lim)
        
        fig.align_ylabels()
        return fig