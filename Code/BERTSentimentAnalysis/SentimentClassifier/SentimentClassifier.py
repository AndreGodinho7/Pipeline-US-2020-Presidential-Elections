from .SentimentClassifierInterface import SentimentClassifierInterface

from abc import abstractmethod
import torch
from torch import nn, optim

class SentimentClassifier(SentimentClassifierInterface):
    def __init__(self, model, device):
        self.model = model.to(device)

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