import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

class ThreePanelPhotPlot:
    '''
    Object for making the fancy three panel photometry plot. 

    Should think about this plot in the context of TOIs with many sectors of data, since it might not make sense in that case e.g. TOI-1247. Will worry about that later.
    '''
    def __init__(self, 
                toi, 
                map_soln, 
                extras, 
                broken_x_axis=False, # Whether or not to break up the x-axis to avoid large gaps between sectors
                **kwargs) -> None:
        
        self.toi = toi
        self.map_soln = map_soln
        self.extras = extras
        self.broken_x_axis = broken_x_axis

        # What to do about these plot hyperparameters
        self.figsize = kwargs['figsize']


    def __plot_top_panel(self):
        pass

    def __plot_middle_panel(self):
        pass
    
    def __plot_bottom_panel(self):
        pass

    def __plot_phase_folded_transit(self):
        pass
    
    def plot(self):
        if self.broken_x_axis:
            self.__broken_three_panel_plot()
        else:
            self.__three_panel_plot()

    
    def __broken_three_panel_plot(self):
        '''
        '''
        