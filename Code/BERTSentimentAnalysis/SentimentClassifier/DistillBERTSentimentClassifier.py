from .SentimentClassifierEncoder import SentimentClassifierEncoder

from abc import abstractmethod
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class DistillBERTSentimentClassifier(SentimentClassifierEncoder):
    def __init__(self, model: nn.Module, tokenizer, threads=None):
        super(DistillBERTSentimentClassifier, self).__init__(model, tokenizer)
        # self.model.distillbert.resize_token_embeddings(len(self.tokenizer))
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

            input_ids = encoding['input_ids'].to(device=self.model.distillbert.device)
            attention_mask = encoding['attention_mask'].to(device=self.model.distillbert.device)

            outputs = self.model(input_ids, attention_mask).detach().numpy()

            preds = outputs.argmax(1)
            predictions.extend(preds)


        # predictions = []

        # self.model.eval()
        # with torch.no_grad():
        #     for batch in dataloader:
        #         message = batch["message"]
        #         input_ids = batch['input_ids'].to(self.model.distillbert.device)
        #         attention_mask = batch['attention_mask'].to(self.model.distillbert.device)

        #         outputs = self.model(input_ids, attention_mask).numpy()

        #         preds = outputs.argmax(1)
        #         predictions.extend(preds)

        return predictions,  # 0 - negative ; 1 - neutral; 2 - positive
        

    def predict_results(self, **kwargs):
        data_loader = kwargs.get('data_loader')
        device = kwargs.get('device')
        correct_predictions = 0
        messages = []
        predictions = []
        prediction_probs = []
        real_values = []

        # notify all layers we are in eval mode
        # e.g., dropout layers will work in eval mode instead of training mode
        self.model.eval()

        # torch.no_grad() impacts the autograd engine and deactivate it 
        # reduce memory usage and speed up
        with torch.no_grad():
            for batch in data_loader:
                text = batch["message"]
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['sentiments'].to(device)
                
                outputs = self.model(input_ids, attention_mask)

                _, preds = torch.max(outputs, dim=1)

                messages.extend(messages)
                predictions.extend(preds)
                prediction_probs.extend(outputs)
                real_values.extend(labels)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()

        return messages, predictions, prediction_probs, real_values 