# General imports
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from tessla.data_utils import find_breaks

# Exoplanet stuff
import exoplanet as xo
import theano

from tessla.plotting_utils import plot_periodogram
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

import datetime
from astropy.time import Time

class ThreePanelPhotPlot:
    '''
    Object for making the fancy three panel photometry plot. 

    Should think about this plot in the context of TOIs with many sectors of data, since it might not make sense in that case e.g. TOI-1247. Will worry about that later.
    '''
    def __init__(self, 
                toi,
                use_broken_x_axis=False, # Whether or not to break up the x-axis to avoid large gaps between sectors
                data_gap_thresh=10, # Days. If gap in data is larger than this threshold, consider it another chunk in the broken axis (need to have use_broken_x_axis=True)
                figsize=(12,14), 
                margin=1, # Units of days
                ylabelpad=10,
                wspace=0.025, # space of gap in broken axis
                d=0.0075, # size of diagonal lines for broken axis
                plot_random_transit_draws=False, # If true, plot random realizations of the phase-folded transit using the posteriors of the model fit.
                num_random_transit_draws=25, # Number of random draws to plot.
                save_format='.png',
                save_dpi=400,
                ) -> None:
        
        self.toi = toi
        self.x = toi.cleaned_time.values
        self.y = toi.cleaned_flux.values
        self.yerr = toi.cleaned_flux_err.values
        self.residuals = None
        
        self.use_broken_x_axis = use_broken_x_axis
        self.data_gap_thresh = data_gap_thresh
        self.plot_random_transit_draws = plot_random_transit_draws
        self.num_random_transit_draws = num_random_transit_draws

        # Plot hyperparameters
        self.figsize = figsize
        self.margin = margin
        self.ylabelpad = ylabelpad
        self.wspace = wspace
        self.d = d
        self.save_format = save_format
        self.save_dpi = save_dpi
    
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
        text = ax.text(xstart, np.max(self.y), sector_str, horizontalalignment='left', verticalalignment='top', fontsize=12)
        text.set_bbox(dict(facecolor='lightgray', alpha=0.65, edgecolor='lightgray'))

    def __add_ymd_label(self, fig, ax, xlims, left_or_right):
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
            bjd_date = ax_yrs_ticks[ind] + self.toi.bjd_ref
        elif left_or_right == 'right':
            ind = -1
            bjd_date = ax_yrs_ticks[ind] + self.toi.bjd_ref
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

    def __broken_three_panel_plot(self):
        '''
        Make the three panel plot using a broken x-axis. Used for TOIs with widely time-separated sectors.
        '''
        break_inds = find_breaks(self.x, diff_threshold=self.data_gap_thresh, verbose=self.toi.verbose)

        # Create the figure object
        fig = plt.figure(figsize=self.figsize)

        # Create the GridSpec objects
        gs0, gs1 = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.5])
        heights = [1, 1, 0.33]
        sps1, sps2, sps3 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0, height_ratios=heights, hspace=0.1)

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

        # Add label for years to the upper axis for the left-most chunk
        left_xlims = xlim_tuple[0]
        self.__add_ymd_label(fig, ax_left, left_xlims, 'left')

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

        # Add label for years to the upper axis for the left-most chunk
        right_xlims = xlim_tuple[-1]
        self.__add_ymd_label(fig, ax_right, right_xlims, 'right')

        # Top panel housekeeping
        bax1.set_xticklabels([])
        bax1.set_ylabel("Relative flux [ppt]", fontsize=14, labelpad=self.ylabelpad)
        for ax in [ax_left, ax_right]:
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
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
        for ax in [ax_left, ax_right]:
            ax.yaxis.set_major_locator(MultipleLocator(1))
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
        bax3.set_ylim([-2.5, 2.5]) # Can change this probably.
        self.residuals = residuals.value # Save these

        # Plot housekeeping
        bax3.set_ylabel("Residuals", fontsize=14, labelpad=self.ylabelpad)
        bax3.set_xlabel(f"Time [BJD - {self.toi.bjd_ref:.1f}]", fontsize=14, labelpad=30)
        
        # If there are more than 3 chunks, make the x-axis ticklabel font smaller so that the numbers don't overlap
        # if len(break_inds) > 2:
        #     bax3.tick_params(axis='x', which='major', labelsize=12)

        ax_left, ax_right = bax3.axs[0], bax3.axs[-1]
        for ax in [ax_left, ax_right]:
            ax.yaxis.set_major_locator(MultipleLocator(2))
            bottom = -1 * np.max(ax.get_ylim())
            ax.set_ylim(bottom=bottom)

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ############################ HOUSEKEEPING FOR TOP THREE PANELS #################################
        ################################################################################################
        fig.align_ylabels() # Align ylabels for each panel
        
        # Make ticks go inward and set multiple locator for x-axis
        for bax in [bax1, bax2, bax3]:
            # Left-most
            ax_left = bax.axs[0]
            ax_left.xaxis.set_major_locator(MultipleLocator(10))
            # If there are more than 3 chunks, make the x-axis ticklabels spacing larger so that the numbers don't overlap
            if len(break_inds) > 2:
                ax_left.xaxis.set_major_locator(MultipleLocator(20))
            ax_left.xaxis.set_minor_locator(MultipleLocator(5))
            ax_left.tick_params(axis="y", direction="in", which="both", left=True, right=False)
            ax_left.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)

            # Middle
            for j in range(1, len(bax.axs) - 1):
                ax_mid = bax.axs[j]
                ax_mid.tick_params(axis="y", which="both", left=False, right=False)
                ax_mid.xaxis.set_major_locator(MultipleLocator(10))
                # If there are more than 3 chunks, make the x-axis ticklabels spacing larger so that the numbers don't overlap
                if len(break_inds) > 2:
                    ax_mid.xaxis.set_major_locator(MultipleLocator(20))
                ax_mid.xaxis.set_minor_locator(MultipleLocator(5))
                ax_mid.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
            
            # Right-most
            ax_right = bax.axs[-1]
            ax_right.xaxis.set_major_locator(MultipleLocator(10))
            # If there are more than 3 chunks, make the x-axis ticklabels spacing larger so that the numbers don't overlap
            if len(break_inds) > 2:
                ax_right.xaxis.set_major_locator(MultipleLocator(20))
            ax_right.xaxis.set_minor_locator(MultipleLocator(5))
            ax_right.tick_params(axis="x", direction="in", which="both", top=True, bottom=True)
            ax_right.tick_params(axis="y", direction="in", which="both", left=False, right=True)

        ################################################################################################
        ############################ PHASE-FOLDED TRANSIT AND RESIDUALS ################################
        ################################################################################################
        fig = self.__plot_phase_folded_transits(fig, gs1)
        
        return fig


    def __three_panel_plot(self):
        '''
        Make a three panel plot but don't have to worry about breaks in the axis. E.g. if there's only one sector of photometry or all sectors are consecutive.
        '''
        # Create the figure object
        fig = plt.figure(figsize=self.figsize)

        # Create the GridSpec objects
        gs0, gs1 = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.5])
        heights = [1, 1, 0.33]
        sps1, sps2, sps3 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0, height_ratios=heights, hspace=0.1)
        
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
        self.__add_ymd_label(fig, ax1, (np.min(self.x), np.max(self.x)), 'left')
        self.__add_ymd_label(fig, ax1, (np.min(self.x), np.max(self.x)), 'right')
        
        # Top panel housekeeping
        ax1.set_xticklabels([])
        ax1.set_ylabel("Relative flux [ppt]", fontsize=14, labelpad=self.ylabelpad)
        ax1.yaxis.set_major_locator(MultipleLocator(2))
        ax1.yaxis.set_minor_locator(MultipleLocator(1))

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
        ax2.yaxis.set_major_locator(MultipleLocator(1))

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
        ax3.yaxis.set_major_locator(MultipleLocator(2))
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
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(2.5))
            ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)
            ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)
        
        ################################################################################################
        ############################ PHASE-FOLDED TRANSIT AND RESIDUALS ################################
        ################################################################################################
        fig = self.__plot_phase_folded_transits(fig, gs1)

        return fig
    
    def __plot_phase_folded_transits(self, fig, gs1):
        '''
        Plot the folded transits for each planet.
        '''
        heights = [1, 0.33]
        sps = gridspec.GridSpecFromSubplotSpec(2, len(self.toi.transiting_planets), subplot_spec=gs1, height_ratios=heights, hspace=0.05)

        gp_mod = self.toi.extras["gp_pred"] + self.toi.map_soln["mean_flux"]
        residuals = self.y - gp_mod - np.sum(self.toi.extras["light_curves"], axis=-1)

        chains = None
        if self.plot_random_transit_draws:
            chains = pd.read_csv(self.toi.chains_path)
        phase_folded_axes = []
        phase_folded_resid_axes = []
        for i,planet in enumerate(self.toi.transiting_planets.values()):

            ax0 = fig.add_subplot(sps[0, i])
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
            map_model = self.toi.extras["light_curves"][:, i][inds]
            
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
                    period = np.array([chains[f"period_{letter}"].values[ind] for letter in self.toi.transiting_planets.keys()])
                    t0 = np.array([chains[f"t0_{letter}"].values[ind] for letter in self.toi.transiting_planets.keys()])
                    ror = np.array([chains[f"ror_{letter}"].values[ind] for letter in self.toi.transiting_planets.keys()])
                    b = np.array([chains[f"b_{letter}"].values[ind] for letter in self.toi.transiting_planets.keys()])
                    dur = np.array([chains[f"dur_{letter}"].values[ind] for letter in self.toi.transiting_planets.keys()])
                    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, ror=ror, duration=dur)

                    # Light curves
                    N_EVAL_POINTS = 500
                    phase_lc = np.linspace(-xlim, xlim, N_EVAL_POINTS)
                    lc_phase_pred = 1e3 * tt.stack(
                                            [
                                                xo_star.get_light_curve(
                                                    orbit=orbit, r=ror, t=t0[n] + phase_lc, texp=self.toi.cadence/60/60/24)[..., n]
                                                    for n in range(self.toi.n_transiting)
                                            ],
                                            axis=-1,
                    )
                    lc_phase_pred = lc_phase_pred.eval()

                    # Plot the random draw. Wasteful because it only uses one of the planet light curves and does this again for the next planet
                    ax0.plot(phase_lc, lc_phase_pred[:, i], color=planet.color, alpha=0.3, zorder=999, label='Random posterior draw')

            # Plot the residuals below
            ax1 = fig.add_subplot(sps[1, i])
            phase_folded_resid_axes.append(ax1)
            ax1.plot(x_fold, residuals, '.k', label='Residuals', alpha=0.3, zorder=0)
            ax1.axhline(0, color="#aaaaaa", lw=1)

            # Plot housekeeping
            ax0.set_title(f"{self.toi.name} {planet.pl_letter}")

            # Put the x-axis labels and ticks in units of hours instead of days
            for ax in [ax0, ax1]:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x * 24:g}"))
                ax.xaxis.set_major_locator(MultipleLocator(2/24))
                ax.xaxis.set_minor_locator(MultipleLocator(1/24))
                ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)
                ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)

            ax0.set_xticklabels([])
            ax1.set_xlabel("Time since transit [hours]", fontsize=14)
            ax0.yaxis.set_major_locator(MultipleLocator(1))
            ax0.yaxis.set_minor_locator(MultipleLocator(0.5))
            ax1.yaxis.set_major_locator(MultipleLocator(2))

            if i == 0:
                ax0.set_ylabel("Relative flux [ppt]", fontsize=14)
                ax1.set_ylabel("Residuals", fontsize=14)
            
            ax0.set_xlim([-xlim, xlim])
            ax1.set_xlim([-xlim, xlim])
            axis_to_data = ax.transAxes + ax.transData.inverted()
            points_data = axis_to_data.transform((0.035, 0.))
            ax0.errorbar(points_data[0], points_data[1], yerr=np.sqrt(np.exp(self.toi.map_soln['log_sigma_lc'])**2 + np.median(self.yerr)**2), fmt='none', color='k', elinewidth=2, capsize=4)
            if i == 0:
                text = ax0.text(points_data[0] + 0.4/24, points_data[1], 'Data pointwise error', fontsize=12)
                text.set_bbox(dict(facecolor='lightgray', alpha=0.65, edgecolor='lightgray'))
        
        # Make the y-axes range the same for all of the phase-folded transit plots
        for axes in [phase_folded_axes, phase_folded_resid_axes]:
            y_phase_max = np.max([max(ax.get_ylim()) for ax in axes])
            y_phase_min = np.min([min(ax.get_ylim()) for ax in axes])
            y_phase_lim = (y_phase_min, y_phase_max)
            for i in range(len(axes)):
                axes[i].set_ylim(y_phase_lim)

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