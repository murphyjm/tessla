import os
import warnings

from astropy.timeseries import LombScargle

import numpy as np
import pandas as pd

# Plotting stuff
from matplotlib import rcParams
rcParams["font.size"] = 16
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter, FormatStrFormatter

class StackedPeriodogram:
    
    def __init__(self,
                toi,
                min_period=1,
                max_period=100, 
                samples_per_peak=1000,
                title='',
                figsize=(14,6),
                save_format='.png',
                save_dpi=400) -> None:

        self.toi = toi
        self.min_period = min_period
        self.max_period = max_period
        self.samples_per_peak = samples_per_peak

        # For a 1 planet system there should 5 periodograms:
        # 1. Photometry, 2. RVs with offsets applied, trend/curvature removed 3. RV residuals, 4. S-Values, 5. RV Window Function
        num_periodograms = 5
        if self.toi.include_svalue_gp:
            num_periodograms += 1 # GP removed
        num_periodograms += len(self.toi.planets) - 1 # Add additional planets so they're removed one by one
        self.num_periodograms = num_periodograms

        self.title = title
        if self.title == '':
            self.title = self.toi.name
        self.figsize = figsize

        self.save_format = save_format
        self.save_dpi = save_dpi
    
    def __get_ls(self, x, y, yerr):
        '''
        Helper function to generate periodograms
        '''
        minimum_frequency = 1 / self.max_period # 1/day
        maximum_frequency = 1 / self.min_period # 1/day
        ls = LombScargle(x, y, yerr)
        freq, power = ls.autopower(minimum_frequency=minimum_frequency, 
                                maximum_frequency=maximum_frequency, 
                                samples_per_peak=self.samples_per_peak) # Set max sampled frequency to 1 day
        
        periods = 1/freq
        peak_per = periods[np.argmax(power)]
        return periods, power, peak_per, ls

    def plot(self, save_fname=None, overwrite=False):
        '''
        Make the plot!
        '''
        assert self.toi.is_joint_model, "Stacked periodogram plot should be used to compare photometry, RVs, residuals, RV window function, etc."

        # Save fname housekeeping and overwrite handling.
        out_dir = os.path.join(self.toi.output_dir, 'periodograms')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if save_fname is None:
            default_save_fname = f"{self.title.replace(' ', '_')}_stacked_periodogram_plot"
            save_fname = os.path.join(out_dir, default_save_fname + self.save_format)
        else:
            save_fname = os.path.join(out_dir, save_fname + self.save_format)
        if not overwrite and os.path.isfile(save_fname):
            warnings.warn("Exiting before plotting to avoid overwriting exisiting plot file.")
            return None
        
        # Plotting!
        fig, ax = plt.subplots(self.num_periodograms, 1, figsize=self.figsize, sharex=True)

        ax = ax.flatten()
        i = 0
        xtext = 0.98
        ytext = 0.6

        ####################
        #### Photometry ####
        ####################
        oot_mask = self.toi.sg_outlier_mask & ~self.toi.all_transits_mask
        periods, power, peak_per, ls = self.__get_ls(self.toi.lc.time.value[oot_mask], 
                                            self.toi.lc.norm_flux.value[oot_mask], 
                                            self.toi.lc.norm_flux_err.value[oot_mask])

        ax[i].plot(periods, power, color='k')
        phot_str = 'TESS OoT '
        if self.toi.flux_origin == 'sap_flux':
            phot_str += "SAP Flux"
        elif self.toi.flux_origin == 'pdcsap_flux':
            phot_str += "PDCSAP Flux"
        text = ax[i].text(xtext, ytext, phot_str, transform=ax[i].transAxes, ha='right')
        text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
        i += 1

        ####################
        ######## RVs #######
        ####################
        rv_residuals = pd.Series(self.toi.rv_df.mnvel.values - self.toi.extras['mean_rv'] - self.toi.extras['bkg_rv']).copy()
        
        # RVs with offsets applied
        periods, power, peak_per, ls = self.__get_ls(self.toi.rv_df.time, 
                                                    rv_residuals,
                                                    self.toi.rv_df.errvel)
        ax[i].plot(periods, power, color='k')
        rv_str = 'RVs - offsets'
        if self.toi.rv_trend:
            rv_str += ' - trend'
        text = ax[i].text(xtext, ytext, rv_str, transform=ax[i].transAxes, ha='right')
        text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
        i += 1
        
        if self.toi.include_svalue_gp:
            # RVs with trend/curvature and GP removed

            for tel in self.toi.rv_inst_names:
                mask = self.toi.rv_df.tel.values == tel
                rv_residuals.loc[mask] -= self.toi.extras[f"gp_rv_{tel}"]
            
            periods, power, peak_per, ls = self.__get_ls(self.toi.rv_df.time, 
                                                        rv_residuals,
                                                        self.toi.rv_df.errvel)
            ax[i].plot(periods, power, color='k')
            rv_str += ' - GP'
            text = ax[i].text(xtext, ytext, rv_str, transform=ax[i].transAxes, ha='right')
            text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
            i += 1

        # Remove planets from RVs
        for planet_ind, planet in enumerate(self.toi.planets.values()):
            if len(self.toi.extras['planet_rv'].shape) > 1:
                rv_residuals -= self.toi.extras['planet_rv'][:, planet_ind]
            else:
                rv_residuals -= self.toi.extras['planet_rv']
            periods, power, peak_per, ls = self.__get_ls(self.toi.rv_df.time,
                                                        rv_residuals,
                                                        self.toi.rv_df.errvel)
            ax[i].plot(periods, power, color='k')
            rv_str += f' - Planet {planet.pl_letter}'
            text = ax[i].text(xtext, ytext, rv_str, transform=ax[i].transAxes, ha='right')
            text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
            i += 1
        
        # S-Values
        periods, power, peak_per, ls = self.__get_ls(self.toi.svalue_df.time,
                                                    self.toi.svalue_df.svalue,
                                                    self.toi.svalue_df.svalue_err)
        ax[i].plot(periods, power, color='k')
        text = ax[i].text(xtext,ytext, 'S-Values', transform=ax[i].transAxes, ha='right')
        text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
        i += 1

        # RV Window function
        periods, power, peak_per, ls = self.__get_ls(self.toi.rv_df.time, 
                                            1, 
                                            1e-6)
        ax[i].plot(periods, power, color='k')
        # ax[i].set_ylim([0, 0.25])
        text = ax[i].text(xtext, ytext, 'RV Window Function', transform=ax[i].transAxes, ha='right')
        text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Plot vertical lines for planets
        for j in range(self.num_periodograms):
            for planet in self.toi.planets.values():
                linestyle = '-'
                if not planet.is_transiting:
                    linestyle = '--' # Give non-transiting planets a different linestyle
                ax[j].axvline(planet.per, color=planet.color, ls=linestyle, zorder=0)
            # Rotation period peak from OoT photometry
            ax[j].axvline(self.toi.rot_per, color='blue', lw=5, alpha=0.5)
            ax[j].axvline(self.toi.rot_per/2, color='cornflowerblue', lw=5, alpha=0.5)

        # Y-axis label
        fig.supylabel('LS Power', fontsize=22)
        # X-axis label
        ax[i].set_xlabel('Period [days]', fontsize=22)

        # Plot housekeeping
        ax0 = ax[-1]
        ax0.set_xlim(left=1)
        ax0.set_xscale('log')
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax0.xaxis.set_minor_formatter(FormatStrFormatter('%d'))
        for i,label in enumerate(ax0.xaxis.get_ticklabels(minor=True)):
            if i%2:
                label.set_visible(False)

        fig.align_ylabels()
        plt.subplots_adjust(hspace=0)
        
        # Save the figure!
        fig.savefig(save_fname, facecolor='white', bbox_inches='tight', dpi=self.save_dpi)
        print(f"{self.title} stacked periodogram plot saved to {save_fname}")
        plt.close()
