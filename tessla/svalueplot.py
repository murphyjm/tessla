
import os
import warnings

from tessla.rvplot import DEFAULT_MARKER_MAPPER
from tessla.plotting_utils import add_ymd_label

import numpy as np

# Plotting stuff
from matplotlib import rcParams
rcParams["font.size"] = 16
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

class SvaluePlot():
    '''
    Object for making various plots of the S Values.
    '''
    def __init__(self, 
                toi,
                figsize=(6,7),
                ylabelpad=10,
                save_format='.png',
                save_dpi=400,
                tel_marker_mapper=None
                ) -> None:
        self.toi = toi
        self.figsize = figsize
        self.ylabelpad = ylabelpad
        self.save_format = save_format
        self.save_dpi = save_dpi
        
        self.df_summary = None
        if tel_marker_mapper is None:
            self.tel_marker_mapper = DEFAULT_MARKER_MAPPER
        else:
            for tel in self.toi.rv_inst_names:
                assert tel in tel_marker_mapper.keys(), "RV instrument not found in tel_marker_mapper dictionary."
            self.tel_marker_mapper = tel_marker_mapper
        
    def gp_plot(self, save_fname=None, overwrite=False):
        '''
        Make the GP plot!
        '''
        if self.toi.verbose:
            print("Creating S-Value plot...")
        
        out_dir = os.path.join(self.toi.model_dir, 'plotting')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # Save fname housekeeping and overwrite handling.
        if save_fname is None:
            default_save_fname = f"{self.toi.name.replace(' ', '_')}_svalue_gp"
            save_fname = os.path.join(out_dir, default_save_fname + self.save_format)
        else:
            save_fname = os.path.join(out_dir, save_fname + self.save_format)
        if not overwrite and os.path.isfile(save_fname):
            warnings.warn("Exiting before plotting to avoid overwriting exisiting plot file.")
            return None
        
        fig = self.__gp_plot()
        
        # Save the figure!
        fig.savefig(save_fname, facecolor='white', bbox_inches='tight', dpi=self.save_dpi)
        print(f"S Value GP plot saved to {save_fname}")
        plt.close()

    
    def __gp_plot(self):
        '''
        '''

        # Create the figure object
        fig = plt.figure(figsize=self.figsize)

        # Create the GridSpec object
        heights = [1, 0.25]
        sps1, sps2 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=heights, hspace=0.1)
        
        ################################################################################################
        ########################### TOP PANEL: Svalues and full Svalue model ###########################
        ################################################################################################
        ax1 = fig.add_subplot(sps1)
        ax1.set_title(self.toi.name, pad=0)
        
        # Plot the data
        for tel in self.toi.svalue_inst_names:
            tel_mask = self.toi.svalue_df.tel.values == tel
            ax1.errorbar(self.toi.svalue_df.time[tel_mask],
                        self.toi.svalue_df.svalue[tel_mask],
                        self.toi.extras['err_svalue'][tel_mask], fmt='.', **self.tel_marker_mapper[tel])

        # Plot the SValue model
        for tel in self.toi.svalue_inst_names:
            # Plot the GP error envelope and the solution
            gp_mod = self.toi.extras[f'gp_svalue_pred_{tel}']
            ax1.fill_between(self.toi.t_svalue, gp_mod + self.toi.extras[f'gp_svalue_pred_stdv_{tel}'], 
                                                gp_mod - self.toi.extras[f'gp_svalue_pred_stdv_{tel}'], 
                                                alpha=0.3, 
                                                color=self.tel_marker_mapper[tel]['color'])
            ax1.plot(self.toi.t_svalue, gp_mod, color='blue', lw=1)

        # Add label for years to the upper axis
        add_ymd_label(self.toi.bjd_ref, fig, ax1, (np.min(self.toi.svalue_df.time), np.max(self.toi.svalue_df.time)), 'left')
        add_ymd_label(self.toi.bjd_ref, fig, ax1, (np.min(self.toi.svalue_df.time), np.max(self.toi.svalue_df.time)), 'right')
        
        # Top panel housekeeping
        ax1.set_xticklabels([])
        ax1.set_ylabel("$S_\mathrm{HK}$ [dex]", fontsize=14, labelpad=self.ylabelpad)
        # major, minor = self.__get_ytick_spacing()
        # ax1.yaxis.set_major_locator(MultipleLocator(major))
        # ax1.yaxis.set_minor_locator(MultipleLocator(minor))
        ax1.legend(fontsize=14, loc='upper right')

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ################################## BOTTOM PANEL: Residuals #####################################
        ################################################################################################
        ax2 = fig.add_subplot(sps2)

        # Plot the residuals
        for tel in self.toi.svalue_inst_names:
            tel_mask = self.toi.svalue_df.tel.values == tel
            ax2.errorbar(self.toi.svalue_df.time[tel_mask], 
                        self.toi.svalue_df.svalue[tel_mask] - self.toi.extras[f'gp_svalue_{tel}'], 
                        self.toi.extras['err_svalue'][tel_mask], fmt='.', **self.tel_marker_mapper[tel])
        ax2.axhline(0, color="#aaaaaa", lw=1)

        # Plot housekeeping
        ax2.set_ylabel("Residuals", fontsize=14, labelpad=self.ylabelpad)
        ax2.set_xlabel(f"Time [BJD - {self.toi.bjd_ref:.1f}]", fontsize=14)
        # major, minor = self.__get_residuals_ytick_spacing()
        # ax2.yaxis.set_major_locator(MultipleLocator(major))
        # ax2.yaxis.set_minor_locator(MultipleLocator(minor))
        bottom = -1 * np.max(ax2.get_ylim())
        ax2.set_ylim(bottom=bottom)

        ################################################################################################
        ################################################################################################
        ################################################################################################

        ################################################################################################
        ############################# HOUSEKEEPING FOR TOP TWO PANELS ##################################
        ################################################################################################
        fig.align_ylabels()

        # Make ticks go inward and set multiple locator for x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(50))
            ax.tick_params(axis='y', direction='in', which='both', left=True, right=True)
            ax.tick_params(axis='x', direction='in', which='both', top=True, bottom=True)

        return fig