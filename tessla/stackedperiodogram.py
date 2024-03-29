import os
import warnings

from astropy.timeseries import LombScargle

import numpy as np
import pandas as pd

from scipy.signal import find_peaks

# Plotting stuff
from matplotlib import rcParams
rcParams["font.size"] = 16
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class StackedPeriodogram:
    
    def __init__(self,
                toi,
                min_period=1,
                max_period=100, 
                samples_per_peak=1000,
                faps=[0.001], # 0.1%
                plot_faps=True,       # If true, plot the lines of FAP listed in the faps argument. Does not do this for the photometry periodogram by default, 
                plot_phot_faps=False, # unless plot_phot_faps is True.
                fap_ls=[':', '-.', '--', '-'],
                plot_phot_vert_line=False,
                label_fontsize=None,
                label_peaks=False,
                label_which_peaks=[],
                label_peak_thresh=1, # In units of percent. Label peaks above this FAP level if label_peaks = True.
                peak_range_low=None,
                peak_range_high=None,
                title='',
                figsize=(14,6),
                save_format='.png',
                save_dpi=400) -> None:

        self.toi = toi
        self.min_period = min_period
        self.max_period = max_period
        self.samples_per_peak = samples_per_peak
        self.faps = faps
        self.plot_faps = plot_faps
        self.plot_phot_faps = plot_phot_faps
        self.fap_ls = fap_ls # Can use at most 4 FAP values
        assert len(faps) <= 4, "Can only specify at most 4 FAP values for now."

        self.plot_phot_vert_line = plot_phot_vert_line # If False, don't plot the vertical lines for the period with max power in the photometry and its first harmonic
        
        if label_fontsize is None and (self.toi.n_planets > 1 or self.toi.include_svalue_gp):
                self.label_fontsize = 12
        else:
            self.label_fontsize = 16

        self.label_peaks = label_peaks
        self.label_which_peaks = label_which_peaks
        if self.label_peaks and len(self.label_which_peaks) == 0:
            print("Warning: label_peaks is True, but did not specify which to label...")
            print("Your options are: ['rvs', 'shk', 'window']")
        self.label_peak_thresh = label_peak_thresh

        if peak_range_low is None:
            self.peak_range_low = self.min_period
        else:
            self.peak_range_low = peak_range_low
        if peak_range_high is None:
            self.peak_range_high = self.max_period
        else:
            self.peak_range_high = peak_range_high
            
        
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
        ls = LombScargle(x, y, yerr) # By default, this is the generalized LS periodogram, since fit_mean=True by default.
        freq, power = ls.autopower(minimum_frequency=minimum_frequency, 
                                maximum_frequency=maximum_frequency, 
                                samples_per_peak=self.samples_per_peak) # Set max sampled frequency to 1 day
        
        periods = 1/freq
        peak_per = periods[np.argmax(power)]
        return periods, power, peak_per, ls
    
    def __label_peaks(self, power, periods, ls, ax):
        '''
        Helper function for labeling periodogram peaks.
        '''
        mask = periods > self.peak_range_low
        mask &= periods < self.peak_range_high
        peak_inds, peak_props = find_peaks(power[mask], height=float(ls.false_alarm_level(self.label_peak_thresh * 1e-2)))
        
        # Sort the peak periods according to highest to lowest power.
        peak_periods = periods[mask][peak_inds]
        sorted_inds = np.argsort(peak_props['peak_heights'])
        peak_heights = peak_props['peak_heights'][sorted_inds]
        peak_periods = peak_periods[sorted_inds]
        for l,per in enumerate(peak_periods):
            ax.text(per, peak_heights[l], f"{per:.1f} d", fontsize=9)

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
        if self.plot_phot_faps:
            phot_faps = ls.false_alarm_level(self.faps)
            for k in range(len(phot_faps)):
                ax[i].axhline(phot_faps[k], ls=self.fap_ls[k])
        phot_str = 'TESS OoT '
        if self.toi.flux_origin == 'sap_flux':
            phot_str += "SAP Flux"
        elif self.toi.flux_origin == 'pdcsap_flux':
            phot_str += "PDCSAP Flux"
        text = ax[i].text(xtext, ytext, phot_str, transform=ax[i].transAxes, ha='right', fontsize=self.label_fontsize)
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
        if self.plot_faps:
            faps = ls.false_alarm_level(self.faps)
            for k in range(len(faps)):
                ax[i].axhline(faps[k], ls=self.fap_ls[k])

        if self.label_peaks and 'rvs' in self.label_which_peaks:
            self.__label_peaks(power, periods, ls, ax[i])

        rv_str = 'RVs - offsets'
        if self.toi.rv_trend:
            rv_str += ' - trend'
        text = ax[i].text(xtext, ytext, rv_str, transform=ax[i].transAxes, ha='right', fontsize=self.label_fontsize)
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
            if self.plot_faps:
                faps = ls.false_alarm_level(self.faps)
                for k in range(len(faps)):
                    ax[i].axhline(faps[k], ls=self.fap_ls[k])

            if self.label_peaks and 'rvs' in self.label_which_peaks:
                self.__label_peaks(power, periods, ls, ax[i])

            rv_str += ' - GP'
            text = ax[i].text(xtext, ytext, rv_str, transform=ax[i].transAxes, ha='right', fontsize=self.label_fontsize)
            text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
            i += 1

        # Remove planets from RVs in order of biggest K-amplitude (using K-amp as a proxy for power)
        planet_values_list = list(self.toi.planets.values())
        kamp_sorted_planet_inds = np.argsort([planet.kamp for planet in planet_values_list])[::-1]
        for planet_ind in kamp_sorted_planet_inds:
            if len(self.toi.extras['planet_rv'].shape) > 1:
                rv_residuals -= self.toi.extras['planet_rv'][:, planet_ind]
            else:
                rv_residuals -= self.toi.extras['planet_rv']
            periods, power, peak_per, ls = self.__get_ls(self.toi.rv_df.time,
                                                        rv_residuals,
                                                        self.toi.rv_df.errvel)
            ax[i].plot(periods, power, color='k')
            if self.plot_faps:
                faps = ls.false_alarm_level(self.faps)
                for k in range(len(faps)):
                    ax[i].axhline(faps[k], ls=self.fap_ls[k])
            
            if self.label_peaks and 'rvs' in self.label_which_peaks:
                self.__label_peaks(power, periods, ls, ax[i])
                
            rv_str += f' - Planet {planet_values_list[planet_ind].pl_letter}'
            text = ax[i].text(xtext, ytext, rv_str, transform=ax[i].transAxes, ha='right', fontsize=self.label_fontsize)
            text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
            i += 1
        
        # S-Values
        periods, power, peak_per, ls = self.__get_ls(self.toi.svalue_df.time,
                                                    self.toi.svalue_df.svalue,
                                                    self.toi.svalue_df.svalue_err)
        ax[i].plot(periods, power, color='k')
        if self.plot_faps:
            faps = ls.false_alarm_level(self.faps)
            for k in range(len(faps)):
                ax[i].axhline(faps[k], ls=self.fap_ls[k])

        if self.label_peaks and 'shk' in self.label_which_peaks:
            self.__label_peaks(power, periods, ls, ax[i])

        text = ax[i].text(xtext,ytext, 'HIRES $S_\mathrm{HK}$', transform=ax[i].transAxes, ha='right', fontsize=self.label_fontsize)
        text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
        i += 1

        # RV Window function
        # This is the last periodogram to plot. Since there is usually a strong peak in the window function at 1 day that messes up the y-axis scale, 
        # make the minimum period slighly larger than 1 day for the RV window function, then set it back to whatever it was before.
        true_min_period = self.min_period
        self.min_period = 1.1
        periods, power, peak_per, ls = self.__get_ls(self.toi.rv_df.time, 
                                            1, 
                                            1e-4)
        ax[i].plot(periods, power, color='k')
        if self.plot_faps:
            faps = ls.false_alarm_level(self.faps)
            for k in range(len(faps)):
                ax[i].axhline(faps[k], ls=self.fap_ls[k])

        if self.label_peaks and 'window' in self.label_which_peaks:
            self.__label_peaks(power, periods, ls, ax[i])

        text = ax[i].text(xtext, ytext, 'RV Window Function', transform=ax[i].transAxes, ha='right', fontsize=self.label_fontsize)
        text.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
        self.min_period = true_min_period

        # Plot vertical lines for planets
        for j in range(self.num_periodograms):
            for planet in self.toi.planets.values():
                linestyle = '-'
                if not planet.is_transiting:
                    linestyle = '--' # Give non-transiting planets a different linestyle
                ax[j].axvline(planet.per, color=planet.color, ls=linestyle, zorder=0)
            if self.toi.include_svalue_gp:
                prot_posterior_med_used_flag = False
                prot_posterior_med_failed_flag = False
                
                # Rotation period from GP used on RVs and Svalues
                if self.toi.svalue_gp_kernel == 'exp_decay':
                    prot_var_name_str = 'log_rho_rv_svalue_gp'
                elif self.toi.svalue_gp_kernel == 'rotation' or self.toi.svalue_gp_kernel == 'activity':
                    prot_var_name_str = 'log_prot_rv_svalue_gp'
                # If sampling has been performed, then use the posterior median. Else, use the MAP value.
                if self.toi.chains_path is not None:
                    try:
                        chains = pd.read_csv(self.toi.chains_path)
                        prot = np.exp(np.median(chains[prot_var_name_str]))
                        prot_posterior_med_used_flag = True
                    except FileNotFoundError:
                        # If reading chains fails, use MAP value
                        prot = np.exp(self.toi.map_soln[prot_var_name_str])
                        prot_posterior_med_used_flag = False
                        prot_posterior_med_failed_flag = True
                else:
                    prot = np.exp(self.toi.map_soln[prot_var_name_str])
                    prot_posterior_med_used_flag = False
                ax[j].axvline(prot, color='red', lw=5, alpha=0.5)
                ax[j].axvline(prot/2, color='tomato', lw=5, alpha=0.5)
            elif self.plot_phot_vert_line:
                # Rotation period peak from OoT photometry
                ax[j].axvline(self.toi.prot, color='blue', lw=5, alpha=0.5)
                ax[j].axvline(self.toi.prot/2, color='cornflowerblue', lw=5, alpha=0.5)

        if self.toi.include_svalue_gp:
            if prot_posterior_med_used_flag:
                print("Chains successfully loaded, GP rotation period line will be posterior median.")
            elif prot_posterior_med_failed_flag:
                print(f"Warning: Problem reading chains using this path: {self.toi.chains_path}. Plotting GP rotation period line using MAP value. This may differ from the posterior median, though.")
            else:
                print("GP rotation period line represents the MAP value.")

        # Y-axis label
        fig.supylabel('GLS Power', fontsize=22, x=0.05)
        # X-axis label
        ax[i].set_xlabel('Period [days]', fontsize=22)
        # Plot title
        ax[0].set_title(self.toi.name, fontsize=22)

        # Plot housekeeping
        ax0 = ax[-1]
        ax0.set_xlim(left=1)
        ax0.set_xscale('log')
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax0.xaxis.set_minor_formatter(FormatStrFormatter('%d'))
        for i,label in enumerate(ax0.xaxis.get_ticklabels(minor=True)):
            if i%2:
                label.set_visible(False)
        
        ax0.tick_params(axis='x', which='major', labelsize=14)
        ax0.tick_params(axis='x', which='minor', labelsize=12)

        fig.align_ylabels()
        plt.subplots_adjust(hspace=0.2)

        # Save the figure!
        fig.savefig(save_fname, facecolor='white', bbox_inches='tight', dpi=self.save_dpi)
        print(f"{self.title} stacked periodogram plot saved to {save_fname}")
        plt.close()
