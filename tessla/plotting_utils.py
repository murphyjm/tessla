import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import savgol_filter

def sg_smoothing_plot(toi):
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
        ax.plot(toi.lc.time[sg_outlier_mask].value, savgol_filter(norm_flux_prime, toi.sg_window_size, polyorder=3), color='C2', label='Smoothed Data')

        # Mark the in-transit data for each planet.
        for pl_letter, planet in toi.transiting_planets.items():
            transit_mask = planet.transit_mask & sector_mask
            ax.plot(toi.lc.time[transit_mask].value, toi.lc.norm_flux[transit_mask], '.', color=planet.color, label=f"Planet {pl_letter} in-transit ($P =$ {planet.per:.1f} d)")
        
        sg_outlier_mask_inv = ~toi.sg_outlier_mask & sector_mask
        ax.plot(toi.lc.time[sg_outlier_mask_inv].value, toi.lc.norm_flux[sg_outlier_mask_inv], 'xr', label="Discarded outliers")

        ax.legend(fontsize=10, bbox_to_anchor=[1,1])
        ax.set_xlabel(f'Time [BJD - {toi.bjd_ref:.1f}]')
        ax.set_ylabel(f'Relative flux [ppt]')
        
        out_dir = os.path.join(toi.phot_dir, 'plotting')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        fig.savefig(os.path.join(out_dir, f'sg_filtering_sector_{sector:02}.png'), bbox_inches='tight', dpi=300)
    
    if toi.verbose:
        print(f"SG smoothing plots saved to {out_dir}")