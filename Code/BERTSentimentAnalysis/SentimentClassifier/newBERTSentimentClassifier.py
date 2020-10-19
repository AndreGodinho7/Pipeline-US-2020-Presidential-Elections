from .SentimentClassifierEncoder import SentimentClassifierEncoder

from abc import abstractmethod, ABC
import numpy as np
import torch
from torch import nn, optim

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

    def predict(self, dataloader):
        # predictions = []

        # encoding = self.tokenize(messages)
        
        # input_ids = encoding['input_ids'].to(device=self.model.bert.device)
        # attention_mask = encoding['attention_mask'].to(device=self.model.bert.device)

        # outputs = self.model(input_ids, attention_mask)

        # _, predictions = torch.max(outputs, dim=1)

        # predictions = list(predictions.numpy())

        predictions = []
        # messages = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                print("chegou à batch")
                # message = batch["message"]
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                outputs = self.model(input_ids, attention_mask)
                print("calculou outputs")

                _, preds = torch.max(outputs, dim=1)
                print("calculou preds")

                predictions.extend(preds)
                print(len(predictions))
                # messages.extend(message)
            print("saiu da batch")
        print(predictions)

        return predictions # 0 - negative ; 1 - neutral; 2 - positive
        

    