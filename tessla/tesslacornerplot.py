# I/O stuff
import os
import warnings

# Data stuff
import pandas as pd

# Plotting stuff
import matplotlib.pyplot as plt
import corner

class TesslaCornerPlot:
    '''
    A simple object for creating the corner plots.
    '''

    def __init__(self,
                toi, 
                chain_labels,
                chains=None,
                title='',
                save_format='.png',
                save_dpi=400) -> None:

        self.toi = toi # TODO: Should make the toi object optional in the future, since you might not have access to the TOI object but you have the chains.
        
        # Chain labels for the plot
        self.chain_labels = chain_labels

        # Either pass the chains directly or read them from file.
        # Could be more efficient to pass them directly rather than reading from file.
        if chains is None:
            chains = pd.read_csv(toi.chains_path)
        self.chains = chains

        self.title = title

        # Save/output hyperparameters
        self.save_format = save_format
        self.save_dpi = save_dpi

    def plot(self, save_fname=None, overwrite=False):
        '''
        Handle plotting i/o. Use helper function to actually create the plot.
        '''
        # Save fname housekeeping and overwrite handling.
        out_dir = os.path.join(self.toi.phot_dir, 'plotting')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if save_fname is None:
            default_save_fname = f"{self.title.replace(' ', '_')}_corner_plot"
            save_fname = os.path.join(out_dir, default_save_fname + self.save_format)
        else:
            save_fname = os.path.join(out_dir, save_fname + self.save_format)
        if not overwrite and os.path.isfile(save_fname):
            warnings.warn("Exiting before plotting to avoid overwriting exisiting plot file.")
            return None

        # Make the plot!
        fig = corner.corner(self.chains, 
                            labels=self.chain_labels,
                            label_kwargs={'fontsize':14},
                            quantiles=[0.16, 0.5, 0.84], 
                            labelpad=0.1, 
                            show_titles=True,
                            title_kwargs={'fontsize':12}
                            )
        fig.suptitle(self.plot_title, fontsize=14)

        # Save the figure!
        fig.savefig(save_fname, facecolor='white', bbox_inches='tight', dpi=self.save_dpi)
        print(f"Corner plot saved to {save_fname}")
        plt.close()