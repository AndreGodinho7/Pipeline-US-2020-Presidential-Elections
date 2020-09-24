from SentimentClassifier import SentimentClassifier

from abc import abstractmethod
import numpy as np
import torch
from torch import nn, optim


class BERTSentimentClassifier(SentimentClassifier):
    def __init__(self, model, device):
        super(BERTSentimentClassifier, self).__init__(model, device)
    
    def train_epoch(self, **kwargs):
        data_loader = kwargs.get('data_loader')
        loss_fn = kwargs.get('loss_fn')
        optimizer = kwargs.get('optimizer')
        device = kwargs.get('device')
        scheduler = kwargs.get('scheduler')
        n_examples = kwargs.get('n_examples')                            
        
        losses = []
        correct_predictions = 0
        
        # notify all layers we are in train mode
        # e.g., dropout layers will work in train mode instead of training mode
        self.model.train()

        for batch in data_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['sentiments'].to(device)

            outputs = self.model(input_ids, attention_mask)

            # _ is a sequence of hidden states of the last layer of the model
            # preds is obtained after pooling with BertPooler on last_hidden_states
            # we pool the first output token
            _, preds = torch.max(outputs, dim=1) 
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels) # -> Tensor
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # avoid exploding gradients
            optimizer.step()
            scheduler.step()
        return correct_predictions.double() / n_examples, np.mean(losses) # -> Tensor


    def eval_epoch(self, **kwargs):
        data_loader = kwargs.get('data_loader')
        device = kwargs.get('device')
        n_examples = kwargs.get('n_examples')   
        
        correct_predictions = 0

        # notify all layers we are in eval mode
        # e.g., dropout layers will work in eval mode instead of training mode
        self.model.eval()

        # torch.no_grad() impacts the autograd engine and deactivate it 
        # reduce memory usage and speed up
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['sentiments'].to(device)

                outputs = self.model(input_ids, attention_mask)

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)

        return correct_predictions.double() / n_examples

    def predict_results(self, **kwargs):
        data_loader = kwargs.get('data_loader')
        device = kwargs.get('device')
        n_examples = kwargs.get('n_examples')   
        
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
                text = batch["review_text"]
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

    def get_params(self):
        pass