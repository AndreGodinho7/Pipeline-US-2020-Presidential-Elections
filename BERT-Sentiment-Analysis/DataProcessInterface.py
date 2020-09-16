from abc import ABC, abstractmethod

class DataProcessInterface(ABC):
    @abstractmethod
    def dataPreprocess(self):
        pass