from abc import ABC, abstractmethod

class Data(ABC):
    def __init__(self, datapath):
        super().__init__()
        self.datapath = datapath
    
    @abstractmethod
    def readData():
        pass