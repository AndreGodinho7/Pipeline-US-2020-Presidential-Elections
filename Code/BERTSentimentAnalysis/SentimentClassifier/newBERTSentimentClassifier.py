from .SentimentClassifierEncoder import SentimentClassifierEncoder

from abc import abstractmethod, ABC
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class BERTSentimentClassifier(SentimentClassifierEncoder):
    def __init__(self, model: nn.Module, tokenizer, threads=None):
        super(BERTSentimentClassifier, self).__init__(model, tokenizer)
    
        if threads is not None:
            torch.set_num_threads(threads)


    def tokenize(self, messages):
        return self.tokenizer(messages, 
                              return_token_type_ids=False,
                              padding='longest',
                              truncation=True,
                              return_tensors="pt")

    def predict(self, dataloader: DataLoader):

        predictions = []
        ids = []

        # for batch in dataloader:
        #     encoding = self.tokenize(batch)

        #     input_ids = encoding['input_ids'].to(device=self.model.bert.device)
        #     attention_mask = encoding['attention_mask'].to(device=self.model.bert.device)
        #     outputs = self.model(input_ids, attention_mask)
        #     preds = outputs.argmax(1)

        #     predictions.extend(preds)

        for batch in dataloader:
            encoding = self.tokenize(batch['tweets'])
            input_ids = encoding['input_ids'].to(device=self.model.distillbert.device)
            attention_mask = encoding['attention_mask'].to(device=self.model.distillbert.device)

            outputs = self.model(input_ids, attention_mask)

            preds = outputs.argmax(1) # 0 - negative ; 1 - neutral; 2 - positive
            predictions.extend(preds.int().tolist())
            ids.extend(batch['ids'])

        return ids, predictions  
