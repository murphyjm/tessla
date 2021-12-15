from tesssystem import TessSystem

class Planet(TessSystem):
    def __init__(self, name, tic=None, toi=None, source=None, sectors=np.array([], dtype=int), ntransiting=1):
        super().__init__(name, tic=tic, toi=toi, source=source, sectors=sectors, ntransiting=ntransiting)