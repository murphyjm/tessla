class ThreePanelPhotPlot:
    '''
    Object for making the fancy three panel photometry plot. 

    Should think about this plot in the context of TOIs with many sectors of data, since it might not make sense in that case e.g. TOI-1247. Will worry about that later.
    '''
    def __init__(self, 
                toi, 
                map_soln, 
                extras) -> None:
        self.toi = toi
        self.map_soln = map_soln
        self.extras = extras

    def plot_top_panel(self):
        pass

    def plot_middle_panel(self):
        pass
    
    def plot_bottom_panel(self):
        pass

    def plot_phase_folded_transit(self):
        pass