from Data import Data

import pandas as pd

class MyData(Data):
    def __init__(self, datapath):
        super().__init__(datapath)
        self.data = self.readData()

    def getData(self):
        return self.data

    def readData(self):
        return pd.read_csv(self.datapath)