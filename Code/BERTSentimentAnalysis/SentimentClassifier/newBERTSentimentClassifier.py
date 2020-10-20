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

    def predict(self, messages: np.ndarray, batch_size):
        predictions = []
        dataloader = DataLoader(messages, batch_size=batch_size)

        for batch in dataloader:
            encoding = self.tokenize(batch)

            input_ids = encoding['input_ids'].to(device=self.model.bert.device)
            attention_mask = encoding['attention_mask'].to(device=self.model.bert.device)
            # try: 
            outputs = self.model(input_ids, attention_mask)
            preds = outputs.argmax(1)

            predictions.extend(preds)
            # except Exception as e:
                # print(e)

        return predictions # 0 - negative ; 1 - neutral; 2 - positive



        # self.model.eval()
        # with torch.no_grad():
        #     for batch in dataloader:
        #         input_ids = batch['input_ids'].to(self.model.bert.device)
        #         attention_mask = batch['attention_mask'].to(self.model.bert.device)

        #         outputs = self.model(input_ids, attention_mask).numpy()

        #         preds = outputs.argmax(1)
        #         predictions.extend(preds)

        

    