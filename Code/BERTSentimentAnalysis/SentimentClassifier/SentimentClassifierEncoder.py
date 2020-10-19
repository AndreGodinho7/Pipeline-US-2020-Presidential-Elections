from abc import abstractmethod, ABC

import torch

class SentimentClassifierEncoder(ABC):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def tokenize(self):
        pass

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def move_model_cpu(self):
        self.model = self.model.to(torch.device("cpu"))

    def move_model_gpu(self, device):
        self.model = self.model.to(device)
