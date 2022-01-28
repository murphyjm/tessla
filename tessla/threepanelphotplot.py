import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from brokenaxes import brokenaxes
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec

from tessla.data_utils import find_breaks

class ThreePanelPhotPlot:
    '''
    Object for making the fancy three panel photometry plot. 

    Should think about this plot in the context of TOIs with many sectors of data, since it might not make sense in that case e.g. TOI-1247. Will worry about that later.
    '''
    def __init__(self, 
                toi, 
                map_soln, 
                extras,
                use_broken_x_axis=False, # Whether or not to break up the x-axis to avoid large gaps between sectors
                data_gap_thresh=10, # Days. If gap in data is larger than this threshold, consider it another chunk in the broken axis (need to have use_broken_x_axis=True)
                figsize=(12,14), 
                margin=1, # Units of days
                ylabelpad=10,
                wspace=0.025, # space of gap in broken axis
                d=0.0075, # size of diagonal lines for broken axis

                ) -> None:
        
        self.toi = toi
        self.x = toi.cleaned_time.values
        self.y = toi.cleaned_flux.values
        self.yerr = toi.cleaned_flux_err.values

        self.map_soln = map_soln
        self.extras = extras
        self.use_broken_x_axis = use_broken_x_axis
        self.data_gap_thresh = data_gap_thresh

        # Plot hyperparameters
        self.figsize = figsize
        self.margin = margin
        self.ylabelpad = ylabelpad
        self.wspace = wspace
        self.d = d

    def __plot_top_panel(self):
        '''
        Plot the flux and GP noise model.
        '''
        

    def __plot_middle_panel(self):
        pass
    
    def __plot_bottom_panel(self):
        pass

    def __plot_phase_folded_transit(self):
        pass
    
    def plot(self):
        '''
        Make the plot!
        '''
        if self.toi.verbose:
            print("Creating three panel plot...")

        if self.broken_x_axis:
            self.__broken_three_panel_plot()
        else:
            self.__three_panel_plot()

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
        for planet in self.toi.transiting_planets.keys():
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
        if len(np.unique(sectors)) == 1:
            sector_str = f"Sector {np.unique(sectors)}"
        else:
            sectors = np.unique(sectors)
            sector_str = "Sectors "
            for j in range(len(sectors) -1):
                sector_str += f"{sectors[j]}, "
            sector_str += f"{sectors[-1]}"
        ax.text(xstart, np.max(self.y), sector_str, horizontalalignment='left', verticalalignment='top', fontsize=12)

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
        gp_mod = self.toi.extras["gp_pred"] + self.toi.map_soln["mean"]
        assert len(self.x) == len(gp_mod), "Different lengths for data being plotted and GP model"
        
        # Left-most chunk
        xstop_ind = break_inds[0] + 1
        bax1.plot(self.x[:xstop_ind], gp_mod[:xstop_ind], color="C2", label="GP model")
        xstart = np.min(self.x)
        xstop  = self.x[break_inds[0]]
        self.__plot_transit_markers(bax1, xstart, xstop)

        # Annotate the top of this chunk with the sector number.
        ax_left = bax1.axs[0]
        self.__annotate_sector_marker(ax_left, xstart, 0, xstop_ind)

        # Middle chunks
        for i in range(len(break_inds) - 1):
            xstart_ind = break_inds[i] + 1
            xstop_ind  = break_inds[i + 1] + 1 # Plus 1 here is for array slicing
            bax1.plot(self.x[xstart_ind:xstop_ind], gp_mod[xstart_ind:xstop_ind], color="C2", label="GP model")
            xstart = self.x[xstart_ind]
            xstop  = self.x[xstop_ind - 1] # Minus 1 here because don't need array slicing.
            self.__plot_transit_markers(bax1, xstart, xstop)
            
            # Annotate the top of this chunk with the sector number
            ax_curr = bax1.axs[i + 1]
            self.__annotate_sector_marker(ax_curr, xstart, xstart_ind, xstop_ind)
            
        
        # Right-most chunk
        xstart_ind = break_inds[-1] + 1
        bax1.plot(self.x[xstart_ind:], gp_mod[xstart_ind:], color="C2", label="GP model")
        xstart = self.x[break_inds[-1] + 1]
        xstop  = np.max(self.x)
        self.__plot_transit_markers(bax1, xstart, xstop)
        # Annotate the top of this chunk with the sector number
        ax_right = bax1.axs[-1]
        # The -1 isn't technically the correct index (leaves our last element) but it shouldn't matter because there wouldn't be a single data point from a different sector at the end.
        self.__annotate_sector_marker(ax_right, xstart, xstart_ind, -0)

        # Top panel housekeeping
        bax1.set_xticklabels([])
        bax1.set_ylabel("Relative flux [ppt]", fontsize=14, labelpad=self.ylabelpad)
        ax_left.yaxis.set_major_locator(MultipleLocator(2))
        ax_left.yaxis.set_minor_locator(MultipleLocator(1))
        ax_right.tick_params(axis='y', label1On=False)

        # TODO:
        # Add label for years to the upper axis for the left-most and right-most chunks


    def __three_panel_plot(self):
        '''
        '''
        pass