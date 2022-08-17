# General imports
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from tessla.data_utils import find_breaks
from astropy import units
from math import ceil

# Exoplanet stuff
import exoplanet as xo
import theano

from tessla.plotting_utils import plot_periodogram, add_ymd_label
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing" # Not exactly sure what this flag change does.
import aesara_theano_fallback.tensor as tt

# Progress bar
from tqdm import tqdm

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.size"] = 16
from matplotlib.ticker import MultipleLocator, FuncFormatter
from brokenaxes import brokenaxes
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec

class ThreePanelPhotPlot:
    '''
    Object for making the fancy three panel photometry plot. 

    Should think about this plot in the context of TOIs with many sectors of data, since it might not make sense in that case e.g. TOI-1247. Will worry about that later.
    '''
    def __init__(self, 
                toi,
                use_broken_x_axis=False, # Whether or not to break up the x-axis to avoid large gaps between sectors
                data_gap_thresh=10, # Days. If gap in data is larger than this threshold, consider it another chunk in the broken axis (need to have use_broken_x_axis=True)
                figwidth=12,
                margin=1, # Units of days
                ylabelpad=10,
                wspace=0.025, # space of gap in broken axis
                d=0.0075, # size of diagonal lines for broken axis
                plot_random_transit_draws=False, # If true, plot random realizations of the phase-folded transit using the posteriors of the model fit.
                num_random_transit_draws=25, # Number of random draws to plot.
                save_format='.png',
                save_dpi=400,
                df_summary_fname=None,
                rms_yscale_phase_folded_panels=True,
                rms_yscale_multiplier=5,
                data_uncert_label_rms_yscale_multiplier=-3,
                sector_marker_fontsize=12,
                param_fontsize=14,
                timeseries_phase_hspace=0.05,
                ) -> None:
        
        self.toi = toi
        self.x = toi.cleaned_time.values
        self.y = toi.cleaned_flux.values
        self.yerr = toi.cleaned_flux_err.values
        self.residuals = None
        self.num_sectors = len(np.unique(self.toi.lc.sector))
        
        self.use_broken_x_axis = use_broken_x_axis
        self.data_gap_thresh = data_gap_thresh
        self.plot_random_transit_draws = plot_random_transit_draws
        self.num_random_transit_draws = num_random_transit_draws

        self.df_summary = None
        if df_summary_fname is not None:
            self.df_summary = pd.read_csv(df_summary_fname, index_col=0)

        # Plot hyperparameters
        self.figwidth = figwidth
        self.margin = margin
        self.ylabelpad = ylabelpad
        self.wspace = wspace
        self.d = d
        self.save_format = save_format
        self.save_dpi = save_dpi
        self.rms_yscale_phase_folded_panels = rms_yscale_phase_folded_panels
        self.rms_yscale_multiplier = rms_yscale_multiplier
        self.data_uncert_label_rms_yscale_multiplier = data_uncert_label_rms_yscale_multiplier
        self.sector_marker_fontsize = sector_marker_fontsize
        self.param_fontsize = param_fontsize
        self.timeseries_phase_hspace = timeseries_phase_hspace

    def plot(self, save_fname=None, overwrite=False):
        '''
        Make the plot!
        '''
        if self.toi.verbose:
            print("Creating three panel plot...")
        
        out_dir = os.path.join(self.toi.model_dir, 'plotting')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # Save fname housekeeping and overwrite handling.
        if save_fname is None:
            default_save_fname = f"{self.toi.name.replace(' ', '_')}_phot_model"
            save_fname = os.path.join(out_dir, default_save_fname + self.save_format)
        else:
            save_fname = os.path.join(out_dir, save_fname + self.save_format)
        if not overwrite and os.path.isfile(save_fname):
            warnings.warn("Exiting before plotting to avoid overwriting exisiting plot file.")
            return None
        
        # Use broken axis plotting or regular plotting
        if self.use_broken_x_axis:
            fig = self.__broken_three_panel_plot()
        else:
            fig = self.__three_panel_plot()
        
        # Save the figure!
        fig.savefig(save_fname, facecolor='white', bbox_inches='tight', dpi=self.save_dpi)
        print(f"Photometry model plot saved to {save_fname}")
        plt.close()
    
    def __get_ytick_spacing(self, yspan):
        '''
        Hacky.
        '''
        major = None
        minor = None
        if yspan < 3:
            major = 0.5
            minor = 0.25
        elif yspan >= 3 and yspan < 4:
            major = 1
            minor = 0.5
        elif yspan >= 4 and yspan < 8:
            major = 2
            minor = 1
        elif yspan >= 6:
            major = 4
            minor = 2
        return major, minor

    def __get_residuals_ytick_spacing(self, yspan):
        '''
        Hacky.
        '''
        major = None
        minor = None
        if yspan < 3:
            major = 0.75
            minor = 0.25
        elif yspan >= 3 and yspan < 3.5:
            major = 1
            minor = 0.5
        elif yspan >= 3.5:
            major = 2
            minor = 1
        return major, minor

    def __get_xtick_spacing(self):
        '''
        Hacky.
        '''
        major = None
        minor = None
        if self.num_sectors < 3:
            major = 10
            minor = 1
        elif self.num_sectors >= 3 and self.num_sectors < 6:
            major = 20
            minor = 10
        elif self.num_sectors >= 6 and self.num_sectors < 12:
            major = 25
            minor = 12.5
        elif self.num_sectors >= 12:
            major = 40
            minor = 20
        return major, minor

    def __get_xlim_tuple(self, break_inds):
        '''
        Get the x-limits of the broken axes
        '''
        xlim_list = []
        xlim_list.append( (np.min(self.x) - self.margin, self.x[break_inds[0]] + self.margin) ) # Get left-most chunk
        
        # Middle chunk(s)
        for i in range(len(break_inds) - 1):
            curr_break_ind = break_inds[i]
            next_break_ind = break_inds[i + 1]
            xlim_list.append( (self.x[curr_break_ind + 1] - self.margin, self.x[next_break_ind] + self.margin) ) # TODO: Does the indexing here make sense?

        xlim_list.append( (self.x[break_inds[-1] + 1] - self.margin, np.max(self.x) + self.margin) ) # Get right-most chunk

        return tuple(xlim for xlim in xlim_list)
    
    def __get_t0s_in_range(self, xstart, xstop, per, t0):
        '''
        Return a list of transit times between xstart and xstop for a planet with period per and reference transit epoch t0.
        '''
        first_transit_in_range = t0 + int((xstart - t0)/per) * per
        t0s = []
        t0_curr = first_transit_in_range
        # Definitely a quicker way to do this, but this is fine for now.
        while t0_curr < xstop:
            t0s.append(t0_curr)
            t0_curr += per
        if len(t0s) == 0:
            if self.toi.verbose:
                print("No transits in range...")
        return t0s

    def __plot_transit_markers(self, ax, xstart, xstop):
        '''
        Mark the transits on ax for the chunk of time between xstart and xstop
        '''
        for planet in self.toi.transiting_planets.values():
            t0s = self.__get_t0s_in_range(xstart, xstop, planet.per, planet.t0)
            for t0 in t0s:
                ax.plot(t0, np.min(self.y), '^', ms=10, color=planet.color)

    def __annotate_sector_marker(self, ax, xstart, xstart_ind, xstop_ind):
        '''
        Annotate the chunk(s) with their sector(s) at the top of the upper panel.
        The indexing kung-fu is a bit arcane here, but it should work.
        '''
        ha = 'left'
        original_indices = self.toi.cleaned_time.index.values
        sectors = self.toi.lc.sector[original_indices[xstart_ind:xstop_ind]]
        assert len(np.unique(sectors)) > 0, "No sector labels."
        if len(np.unique(sectors)) == 1:
            sector_str = f"Sector {np.unique(sectors).value[0]}"
        elif len(np.unique(sectors)) > 1:
            sectors = np.unique(sectors)
            sector_str = "Sectors "
            for j in range(len(sectors) -1):
                sector_str += f"{sectors.value[j]}, "
            sector_str += f"{sectors.value[-1]}"
        
        xpos = (self.toi.cleaned_time.values[xstop_ind - 1] + xstart)/2
        ha = 'center'
        text = ax.text(xpos, np.max(self.y), sector_str, horizontalalignment=ha, verticalalignment='top', fontsize=self.sector_marker_fontsize)
        text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='none'))

    def __broken_three_panel_plot(self):
        '''
        Make the three panel plot using a broken x-axis. Used for TOIs with widely time-separated sectors.
        '''
        break_inds = find_breaks(self.x, diff_threshold=self.data_gap_thresh, verbose=self.toi.verbose)
        
        figheight = 14 # Default fig height
        num_planet_rows = ceil(self.toi.n_transiting / 2)
        if num_planet_rows > 1:
            timeseries_height = 2 * figheight / 3
            planet_row_height = 1 * figheight / 3
        else:
            timeseries_height = 3 * figheight / 5
            planet_row_height = 2 * figheight / 5
            self.timeseries_phase_hspace = 0.07
        if num_planet_rows > 1:
            figheight = figheight + (num_planet_rows - 1) * planet_row_height

        # Create the figure object
        fig = plt.figure(figsize=(self.figwidth, figheight))

        # Create the GridSpec objects
        heights = [1, 1, 0.33]
        gs0 = GridSpec(3, 1, figure=fig, height_ratios=heights, hspace=0.1)
        timeseries_bottom = 1 - timeseries_height / figheight
        gs0.update(bottom=timeseries_bottom + self.timeseries_phase_hspace*0.5)
        sps1, sps2, sps3 = gs0

        xlim_tuple = self.__get_xlim_tuple(break_inds)
        
        ################################################################################################
        ################################# TOP PANEL: Data and GP model #################################
        ################################################################################################
        bax1 = brokenaxes(xlims=xlim_tuple, d=self.d, subplot_spec=sps1, despine=False, wspace=self.wspace)
        bax1.set_title(self.toi.name, pad=10)

        # Plot the data
        bax1.plot(self.x, self.y, '.k', alpha=0.3, label="Data")
        
        # Plot the GP model on top chunk-by-chunk to avoid the lines that extend into the gaps
        # Also mark the transits in each chunk for each planet
        gp_mod = self.toi.extras["gp_pred"] + self.toi.map_soln["mean_flux"]
        assert len(self.x) == len(gp_mod), "Different lengths for data being plotted and GP model"

        # --------------- #
        # Left-most chunk #
        # --------------- #
        xstop_ind = break_inds[0] + 1
        bax1.plot(self.x[:xstop_ind], gp_mod[:xstop_ind], color="C2", label="GP model")
        xstart = np.min(self.x)
        xstop  = self.x[break_inds[0]]
        self.__plot_transit_markers(bax1, xstart, xstop)

        # Annotate the top of this chunk with the sector number.
        ax_left = bax1.axs[0]
        self.__annotate_sector_marker(ax_left, xstart, 0, xstop_ind)

        # ------------- #
        # Middle chunks #
        # ------------- #
        for i in range(len(break_inds) - 1):
            # Plot the GP model
            xstart_ind = break_inds[i] + 1
            xstop_ind  = break_inds[i + 1] + 1 # Plus 1 here is for array slicing
            bax1.plot(self.x[xstart_ind:xstop_ind], gp_mod[xstart_ind:xstop_ind], color="C2", label="GP model")
            xstart = self.x[xstart_ind]
            xstop  = self.x[xstop_ind - 1] # Minus 1 here because don't need array slicing.
            self.__plot_transit_markers(bax1, xstart, xstop)
            
            # Annotate the top of this chunk with the sector number
            ax_curr = bax1.axs[i + 1]
            self.__annotate_sector_marker(ax_curr, xstart, xstart_ind, xstop_ind)
            
        # ---------------- #
        # Right-most chunk #
        # ---------------- #
        # Plot the GP model
        xstart_ind = break_inds[-1] + 1
        bax1.plot(self.x[xstart_ind:], gp_mod[xstart_ind:], color="C2", label="GP model")
        xstart = self.x[break_inds[-1] + 1]
        xstop  = np.max(self.x)
        self.__plot_transit_markers(bax1, xstart, xstop)

        # Annotate the top of this chunk with the sector number
        ax_right = bax1.axs[-1]
        # The -1 isn't technically the correct index (leaves our last element) but it shouldn't matter because there wouldn't be a single data point from a different sector at the end.
        self.__annotate_sector_marker(ax_right, xstart, xstart_ind, -1)

        # Top panel housekeeping
        bax1.set_xticklabels([])
        bax1.set_ylabel("Relative flux [ppt]", fontsize=14, labelpad=self.ylabelpad)
        major, minor = self.__get_ytick_spacing(np.max(self.y) - np.min(self.y))
        for ax in [ax_left, ax_right]:
            ax.yaxis.set_major_locator(MultipleLocator(major))
            ax.yaxis.set_minor_locator(MultipleLocator(minor))
        ax_right.tick_params(axis='y', label1On=False)

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ####################### MIDDLE PANEL: Flattened data and orbital model #########################
        ################################################################################################
        bax2 = brokenaxes(xlims=xlim_tuple, d=self.d, subplot_spec=sps2, despine=False, wspace=self.wspace)

        # Plot the flattened data and the orbital model for each planet. Could also plot the orbital models in chunks 
        # like the gp model so plot lines don't extend into the data gaps but just being lazy here. 
        bax2.plot(self.x, self.y - gp_mod, ".k", alpha=0.3, label="Flattened data")
        for k, planet in enumerate(self.toi.transiting_planets.values()):
            bax2.plot(self.x, self.toi.extras["light_curves"][:, k], color=planet.color, label=f"{self.toi.name} {planet.pl_letter}", alpha=1.0, zorder=2000 - k)
        
        # Plot housekeeping
        bax2.set_xticklabels([])
        bax2.set_ylabel("Relative flux [ppt]", fontsize=14, labelpad=self.ylabelpad)
        ax_left, ax_right = bax2.axs[0], bax2.axs[-1]
        major, minor = self.__get_ytick_spacing(np.max(self.y - gp_mod) - np.min(self.y - gp_mod))
        for ax in [ax_left, ax_right]:
            ax.yaxis.set_major_locator(MultipleLocator(major))
        ax_right.set_yticklabels([])

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ################################## BOTTOM PANEL: Residuals #####################################
        ################################################################################################
        bax3 = brokenaxes(xlims=xlim_tuple, d=self.d, subplot_spec=sps3, despine=False, wspace=self.wspace)

        # Plot the residuals about the full model
        residuals = self.y - gp_mod - np.sum(self.toi.extras["light_curves"], axis=-1)
        bax3.plot(self.x, residuals, ".k", alpha=0.3)
        bax3.axhline(0, color="#aaaaaa", lw=1)
        for ax in bax3.axs[1:]:
            ax.tick_params(axis='y', label1On=False) # Avoid y-axis labels popping up.
        try:
            self.residuals = residuals.value # Save these
        except AttributeError:
            self.residuals = residuals

        # Plot housekeeping
        bax3.set_ylabel("Residuals", fontsize=14, labelpad=self.ylabelpad)
        bax3.set_xlabel(f"Time [BJD - {self.toi.bjd_ref:.1f}]", fontsize=14, labelpad=30)

        ax_left, ax_right = bax3.axs[0], bax3.axs[-1]
        major, minor = self.__get_residuals_ytick_spacing(np.max(self.residuals) - np.min(self.residuals))
        for ax in [ax_left, ax_right]:
            ax.yaxis.set_major_locator(MultipleLocator(major))
            bottom = -1 * np.max(ax.get_ylim())
            ax.set_ylim(bottom=bottom)

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ############################ HOUSEKEEPING FOR TOP THREE PANELS #################################
        ################################################################################################
        fig.align_ylabels() # Align ylabels for each panel
        # TODO: Need to the X ticklabel spacing??
        # Make ticks go inward and set multiple locator for x-axis
        for ax_num, bax in enumerate([bax1, bax2, bax3]):
            # Left-most
            ax_left = bax.axs[0]
            
            major, minor = self.__get_xtick_spacing()
            ax_left.xaxis.set_major_locator(MultipleLocator(major))
            ax_left.xaxis.set_minor_locator(MultipleLocator(minor))

            ax_left.tick_params(axis="y", direction="in", which="both", left=True, right=False)
            ax_left.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)

            if ax_num == 0:
                # Add label for years to the upper axis for the left-most chunk
                left_xlims = xlim_tuple[0]
                add_ymd_label(self.toi.bjd_ref, fig, ax_left, left_xlims, 'left')

            # Middle
            for j in range(1, len(bax.axs) - 1):
                ax_mid = bax.axs[j]
                ax_mid.tick_params(axis="y", which="both", left=False, right=False)
                
                ax_mid.xaxis.set_major_locator(MultipleLocator(major))
                ax_mid.xaxis.set_minor_locator(MultipleLocator(minor))

                ax_mid.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
            
            # Right-most
            ax_right = bax.axs[-1]

            ax_right.xaxis.set_major_locator(MultipleLocator(major))
            ax_right.xaxis.set_minor_locator(MultipleLocator(minor))

            ax_right.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
            ax_right.tick_params(axis="y", direction="in", which="both", left=False, right=True)

            if ax_num == 0:
                # Add label for years to the upper axis for the right-most chunk
                right_xlims = xlim_tuple[-1]
                add_ymd_label(self.toi.bjd_ref, fig, ax_right, right_xlims, 'right')

        ################################################################################################
        ############################ PHASE-FOLDED TRANSIT AND RESIDUALS ################################
        ################################################################################################
        fig = self.__plot_phase_folded_transits(fig, timeseries_bottom)
        
        return fig

    def __three_panel_plot(self):
        '''
        Make a three panel plot but don't have to worry about breaks in the axis. E.g. if there's only one sector of photometry or all sectors are consecutive.
        '''
        figheight = 14 # Default fig height
        num_planet_rows = ceil(self.toi.n_transiting / 2)
        if num_planet_rows > 1:
            timeseries_height = 2 * figheight / 3
            planet_row_height = 1 * figheight / 3
        else:
            timeseries_height = 3 * figheight / 5
            planet_row_height = 2 * figheight / 5
            self.timeseries_phase_hspace = 0.07
        if num_planet_rows > 1:
            figheight = figheight + (num_planet_rows - 1) * planet_row_height

        # Create the figure object
        fig = plt.figure(figsize=(self.figwidth, figheight))

        # Create the GridSpec objects
        heights = [1, 1, 0.33]
        gs0 = GridSpec(3, 1, figure=fig, height_ratios=heights, hspace=0.1)
        timeseries_bottom = 1 - timeseries_height / figheight
        gs0.update(bottom=timeseries_bottom + self.timeseries_phase_hspace*0.5)
        sps1, sps2, sps3 = gs0

        ################################################################################################
        ################################# TOP PANEL: Data and GP model #################################
        ################################################################################################
        ax1 = fig.add_subplot(sps1)
        ax1.set_title(self.toi.name, pad=10)
        
        # Plot the data
        ax1.plot(self.x, self.y, '.k', alpha=0.3, label="Data")

        # Plot the GP model and mark the transits
        gp_mod = self.toi.extras["gp_pred"] + self.toi.map_soln["mean_flux"]
        assert len(self.x) == len(gp_mod), "Different lengths for data being plotted and GP model"
        ax1.plot(self.x, gp_mod, color="C2", label="GP model")
        self.__plot_transit_markers(ax1, np.min(self.x), np.max(self.x))

        # Annotate the top of the plot with the sector number
        self.__annotate_sector_marker(ax1, np.min(self.x), 0, -1)

        # Add label for years to the upper axis
        add_ymd_label(self.toi.bjd_ref, fig, ax1, (np.min(self.x), np.max(self.x)), 'left')
        add_ymd_label(self.toi.bjd_ref, fig, ax1, (np.min(self.x), np.max(self.x)), 'right')
        
        # Top panel housekeeping
        ax1.set_xticklabels([])
        ax1.set_ylabel("Relative flux [ppt]", fontsize=14, labelpad=self.ylabelpad)
        major, minor = self.__get_ytick_spacing(np.max(self.y) - np.min(self.y))
        ax1.yaxis.set_major_locator(MultipleLocator(major))
        ax1.yaxis.set_minor_locator(MultipleLocator(minor))

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ####################### MIDDLE PANEL: Flattened data and orbital model #########################
        ################################################################################################
        ax2 = fig.add_subplot(sps2)
        ax2.plot(self.x, self.y - gp_mod, ".k", alpha=0.3, label="Flattened data")
        for k, planet in enumerate(self.toi.transiting_planets.values()):
            ax2.plot(self.x, self.toi.extras["light_curves"][:, k], color=planet.color, label=f"{self.toi.name} {planet.pl_letter}", alpha=1.0, zorder=2000 - k)
        
        # Plot housekeeping
        ax2.set_xticklabels([])
        ax2.set_ylabel("Relative flux [ppt]", fontsize=14, labelpad=self.ylabelpad)
        major, minor = self.__get_ytick_spacing(np.max(self.y - gp_mod) - np.min(self.y - gp_mod))
        ax2.yaxis.set_major_locator(MultipleLocator(major))
        ax2.yaxis.set_minor_locator(MultipleLocator(minor))

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ################################## BOTTOM PANEL: Residuals #####################################
        ################################################################################################
        ax3 = fig.add_subplot(sps3)

        # Plot the residuals about the full model
        residuals = self.y - gp_mod - np.sum(self.toi.extras["light_curves"], axis=-1)
        ax3.plot(self.x, residuals, ".k", alpha=0.3)
        ax3.axhline(0, color="#aaaaaa", lw=1)

        # Plot housekeeping
        ax3.set_ylabel("Residuals", fontsize=14, labelpad=self.ylabelpad)
        ax3.set_xlabel(f"Time [BJD - {self.toi.bjd_ref:.1f}]", fontsize=14)
        major, minor = self.__get_residuals_ytick_spacing(np.max(residuals) - np.min(residuals))
        ax3.yaxis.set_major_locator(MultipleLocator(major))
        bottom = -1 * np.max(ax3.get_ylim())
        ax3.set_ylim(bottom=bottom)

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ############################ HOUSEKEEPING FOR TOP THREE PANELS #################################
        ################################################################################################
        fig.align_ylabels()

        # Make ticks go inward and set multiple locator for x-axis
        major, minor = self.__get_xtick_spacing()
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_locator(MultipleLocator(major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
            ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)
            ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)
        
        ################################################################################################
        ############################ PHASE-FOLDED TRANSIT AND RESIDUALS ################################
        ################################################################################################
        fig = self.__plot_phase_folded_transits(fig, timeseries_bottom)

        return fig
    
    def __plot_phase_folded_transits(self, fig, timeseries_bottom):
        '''
        Plot the folded transits for each planet.
        '''

        gp_mod = self.toi.extras["gp_pred"] + self.toi.map_soln["mean_flux"]
        residuals = self.y - gp_mod - np.sum(self.toi.extras["light_curves"], axis=-1)

        chains = None
        if self.plot_random_transit_draws:
            chains = pd.read_csv(self.toi.chains_path)
        phase_folded_axes = []
        phase_folded_resid_axes = []

        # Set up the outer gridspec
        num_planet_rows = ceil(self.toi.n_transiting / 2)
        outer_gs = GridSpec(num_planet_rows, 2, figure=fig, hspace=0.2)
        outer_gs.update(top=timeseries_bottom - self.timeseries_phase_hspace*0.5)

        for k in range(num_planet_rows):
            # Nested gridspec objects. One for each row of the phased plots. 
            # See https://stackoverflow.com/questions/31484273/spacing-between-some-subplots-but-not-all
            heights = [1, 0.33] # Heigh ratio between phase plot and residuals
            sps = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[k, :], height_ratios=heights, hspace=0.05)

            # Figure out how many columns in this row
            planets_left = self.toi.n_transiting - 2 * k
            if planets_left > 1:
                num_cols = 2
            else:
                num_cols = 1
            planet_start_ind = 2 * k
            planet_end_ind = planet_start_ind + num_cols
            planet_ind = planet_start_ind

            for planet in list(self.toi.transiting_planets.values())[planet_start_ind:planet_end_ind]:
                
                # Get the correct column.
                if planet_ind % 2 == 0 and planet_ind < self.toi.n_transiting - 1:
                    planet_col_ind = 0
                elif planet_ind % 2 == 0 and planet_ind == self.toi.n_transiting - 1:
                    planet_col_ind = slice(None)
                else:
                    planet_col_ind = 1
                
                ax0 = fig.add_subplot(sps[0, planet_col_ind])
                phase_folded_axes.append(ax0)

                # Plot the folded data
                x_fold = (self.x - planet.t0 + 0.5 * planet.per) % planet.per - 0.5 * planet.per
                ax0.plot(x_fold, self.y - gp_mod, ".k", label="Data", zorder=-1000, alpha=0.3)

                # Plot the binned flux in bins of 30 minutes
                bin_duration = 0.5 / 24 # 30 minutes in units of days
                bins = (x_fold.max() - x_fold.min()) / bin_duration
                binned_flux, binned_edges, _ = binned_statistic(x_fold, self.y - gp_mod, statistic="mean", bins=bins)
                binned_edge_diff = np.ediff1d(binned_edges) / 2
                binned_locs = binned_edges[:-1] + binned_edge_diff
                ax0.scatter(binned_locs, binned_flux, s=20, color='tomato', edgecolor='red', zorder=1000, label='Binned flux')

                # Calculate indices for the folded model
                inds = np.argsort(x_fold)
                xlim = 0.3 # Days
                inds = inds[np.abs(x_fold)[inds] < xlim] # Get indices within the xlim of the transit
                map_model = self.toi.extras["light_curves"][:, planet_ind][inds]
                
                # Plot the MAP solution
                ax0.plot(x_fold[inds], map_model, color=planet.color, alpha=1, zorder=1001, label="MAP solution")

                if self.plot_random_transit_draws:
                    if self.toi.verbose:
                        print(f"Plotting {self.num_random_transit_draws} random draws of phase-folded transit for planet {planet.pl_letter}...")
                    for j in tqdm(range(self.num_random_transit_draws)): # Could optionally use the "disable" keyword argument to only use the progress if self.toi.verbose == True.
                        
                        ind = np.random.choice(np.arange(self.num_random_transit_draws))

                        # Build the model we used before
                        # Star
                        u = [chains['u_0'].values[ind], chains['u_1'].values[ind]]
                        xo_star = xo.LimbDarkLightCurve(u)

                        # Orbit
                        period = chains[f"period_{planet.pl_letter}"].values[ind]
                        t0 = chains[f"t0_{planet.pl_letter}"].values[ind]
                        r_pl = chains[f"r_pl_{planet.pl_letter}"].values[ind]
                        b = chains[f"b_{planet.pl_letter}"].values[ind]
                        if not self.toi.is_joint_model:
                            dur = chains[f"dur_{planet.pl_letter}"].values[ind]
                            orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, duration=dur, r_star=self.toi.star.rstar)
                        else:
                            # If a joint model, parameterized in terms of stellar density and ecc and omega directly
                            rstar = chains["rstar"].values[ind]
                            mstar = chains["mstar"].values[ind]
                            if self.toi.force_circular_orbits_for_transiting_planets:
                                ecc = None
                                omega = None
                            else:
                                ecc = chains[f"ecc_{planet.pl_letter}"].values[ind]
                                omega = chains[f"omega_{planet.pl_letter}"].values[ind]
                            orbit = xo.orbits.KeplerianOrbit(r_star=rstar, m_star=mstar, period=period, t0=t0, b=b, ecc=ecc, omega=omega)

                        # Light curves
                        N_EVAL_POINTS = 500
                        phase_lc = np.linspace(-xlim, xlim, N_EVAL_POINTS)
                        lc_phase_pred = 1e3 * xo_star.get_light_curve(orbit=orbit, r=r_pl, t=t0 + phase_lc, texp=self.toi.cadence/60/60/24)
                        lc_phase_pred = lc_phase_pred.eval()

                        # Plot the random draw. Wasteful because it only uses one of the planet light curves and does this again for the next planet
                        ax0.plot(phase_lc, lc_phase_pred, color=planet.color, alpha=0.3, zorder=999, label='Random posterior draw')

                # Plot the residuals below
                ax1 = fig.add_subplot(sps[1, planet_col_ind], sharex=ax0)
                phase_folded_resid_axes.append(ax1)
                ax1.plot(x_fold, residuals, '.k', label='Residuals', alpha=0.3, zorder=0)
                ax1.axhline(0, color="#aaaaaa", lw=1)

                # Plot housekeeping
                try:
                    planet_name = planet.alt_name
                except AttributeError:
                    planet_name = planet.pl_letter
                ax0.set_title(f"{self.toi.name} {planet_name}")

                # Put the x-axis labels and ticks in units of hours instead of days
                for ax in [ax0, ax1]:
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 24:g}"))
                    ax.xaxis.set_major_locator(MultipleLocator(2/24))
                    ax.xaxis.set_minor_locator(MultipleLocator(1/24))
                    ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)
                    ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)
                plt.setp(ax0.get_xticklabels(), visible=False)
                if k == num_planet_rows - 1: # Only add the xlabel to the bottom row
                    ax1.set_xlabel("Time since transit [hours]", fontsize=14)
                major, minor = self.__get_ytick_spacing(np.max(self.y - gp_mod) - np.min(self.y - gp_mod))
                ax0.yaxis.set_major_locator(MultipleLocator(major))
                ax0.yaxis.set_minor_locator(MultipleLocator(minor))

                # Residuals
                major, minor = self.__get_residuals_ytick_spacing(np.max(self.y - gp_mod) - np.min(self.y - gp_mod))
                ax1.yaxis.set_major_locator(MultipleLocator(major))

                if planet_col_ind == 0 or isinstance(planet_col_ind, slice):
                    ax0.set_ylabel("Relative flux [ppt]", fontsize=14)
                    ax1.set_ylabel("Residuals", fontsize=14)
                
                ax0.set_xlim([-xlim, xlim])
                ax1.set_xlim([-xlim, xlim])
                data_uncert = np.sqrt(np.exp(2 * self.toi.map_soln['log_sigma_phot']) + np.median(self.yerr)**2)
                if self.rms_yscale_phase_folded_panels:
                    ax0.set_ylim([-self.rms_yscale_multiplier * data_uncert, self.rms_yscale_multiplier * data_uncert])
                    ax1.set_ylim([-self.rms_yscale_multiplier * data_uncert, self.rms_yscale_multiplier * data_uncert])

                ax0.errorbar(-xlim + xlim*0.1, self.data_uncert_label_rms_yscale_multiplier * data_uncert, yerr=data_uncert, 
                                fmt='none', color='k', elinewidth=2, capsize=4)
                
                if planet_col_ind == 0 or isinstance(planet_col_ind, slice):
                    if planet_ind == 0:
                        text = ax0.text(-xlim + xlim*0.15, self.data_uncert_label_rms_yscale_multiplier * data_uncert, 'Data uncert.', fontsize=12)
                        text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='none'))
                
                # Annotate the phase folded panels with parameter values
                if self.df_summary is not None:
                    per_med = self.df_summary.loc[f'period_{planet.pl_letter}', 'median']
                    per_err = self.df_summary.loc[f'period_{planet.pl_letter}', 'std']
                    if per_err < 1:
                        per_str = f"$P =$ {per_med:.2f} $\pm$ {per_err:.2e} d"
                    else:
                        per_str = f"$P =$ {per_med:.1f} $\pm$ {per_err:.1f} d"

                    ror_med = self.df_summary.loc[f'ror_{planet.pl_letter}', 'median'] * 100
                    ror_err = self.df_summary.loc[f'ror_{planet.pl_letter}', 'std'] * 100
                    ror_str = f"$R_\mathrm{{p}}/R_* = {ror_med:.2f} \pm {ror_err:.2f}$ $\%$"

                    b_med = self.df_summary.loc[f'b_{planet.pl_letter}', 'median']
                    b_err = self.df_summary.loc[f'b_{planet.pl_letter}', 'std']
                    b_str = f"$b = {b_med:.2f} \pm {b_err:.2f}$"

                    rp_med = self.df_summary.loc[f'rp_{planet.pl_letter}', 'median']
                    rp_err = self.df_summary.loc[f'rp_{planet.pl_letter}', 'std']
                    rp_str = f"$R_\mathrm{{p}} = {rp_med:.2f} \pm {rp_err:.2f}$ $R_\oplus$"
                    
                else:
                    # Annotate with MAP solution values for Period and radius and rp/rstar and impact parameter.
                    per_str = f"$P =$ {planet.per:.2f} d"
                    ror_str = f"$R_\mathrm{{p}}/R_* = {np.sqrt(planet.depth * 1e-3) * 100:.2f}$ $\%$"
                    b_str = f"$b =$ {planet.b:.2f}"
                    rp_map = (units.Rsun.to(units.Rearth, np.sqrt(planet.depth * 1e-3) * self.toi.star.rstar))
                    rp_str = f"$R_\mathrm{{p}} = {rp_map:.2f}$ $R_\oplus$"
                
                text_obj = ax0.text(0.05, 0.975, per_str + '\n' + ror_str + '\n' + b_str + '\n' + rp_str, ha='left', va='top', transform=ax0.transAxes, fontsize=self.param_fontsize)
                text_obj.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='none'))

                # Move on to the next planet!
                planet_ind += 1
        
        # Make the y-axes range the same for all of the phase-folded transit plots *IF* the transit depths are similar.
        depth_planet_b = self.toi.planets['b'].depth # ppt
        depths_all_planets = np.array([planet.depth for planet in self.toi.transiting_planets.values()])
        depths_all_planets_minus_b = np.abs(depths_all_planets - depth_planet_b)
        if all(depths_all_planets_minus_b < 1.0):
            for axes in [phase_folded_axes, phase_folded_resid_axes]:
                y_phase_max = np.max([max(ax.get_ylim()) for ax in axes])
                y_phase_min = np.min([min(ax.get_ylim()) for ax in axes])
                y_phase_lim = (y_phase_min, y_phase_max)
                for i in range(len(axes)):
                    axes[i].set_ylim(y_phase_lim)
        else:
            for axes in [phase_folded_axes, phase_folded_resid_axes]:
                for i in range(len(axes)):
                    y_phase_lim = (-1.5 * depths_all_planets[i], 1.5 * depths_all_planets[i]) # HACK
                    axes[i].set_ylim(y_phase_lim)

        
        fig.align_ylabels()

        return fig

    def residuals_periodogram(self, save_fname=None, overwrite=False, min_per=1, max_per=50, samples_per_peak=1000, **kwargs):
        '''
        Make a plot of the residuals of the photometric model.
        '''
        if self.toi.verbose:
            print("Creating periodogram of photometric model residuals...")
        
        out_dir = os.path.join(self.toi.model_dir, 'plotting')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # Save fname housekeeping and overwrite handling.
        if save_fname is None:
            default_save_fname = f"{self.toi.name.replace(' ', '_')}_phot_residuals_periodogram"
            save_fname = os.path.join(out_dir, default_save_fname + self.save_format)
        else:
            save_fname = os.path.join(out_dir, save_fname + self.save_format)
        if not overwrite and os.path.isfile(save_fname):
            warnings.warn("Exiting before plotting to avoid overwriting exisiting plot file.")
            return None
        
        # Make and plot the periodogram
        try:
            xo_ls = xo.estimators.lomb_scargle_estimator(self.x, self.residuals, self.yerr.value, 
                                                        min_period=min_per, max_period=max_per, samples_per_peak=samples_per_peak)
            fig, ax = plot_periodogram(self.toi.output_dir, f"{self.toi.name} Photometric Residuals LS Periodogram", 
                        xo_ls, 
                        self.toi.transiting_planets,
                        verbose=self.toi.verbose,
                        **kwargs) # What to do with figure and ax that is returned?
            return fig, ax
        except:
            print("Error when constructing periodogram of residuals. Continuing.")
            return None, None