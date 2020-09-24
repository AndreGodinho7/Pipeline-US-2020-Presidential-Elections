from abc import ABC, abstractmethod

class SentimentClassifierInterface(ABC):
    @abstractmethod
    def get_params(self):
        pass
    
    @abstractmethod
    def train_epoch(self, **kwargs):
        pass

    @abstractmethod
    def eval_epoch(self, **kwargs):
        pass

    @abstractmethod
    def predict(self):
        pass