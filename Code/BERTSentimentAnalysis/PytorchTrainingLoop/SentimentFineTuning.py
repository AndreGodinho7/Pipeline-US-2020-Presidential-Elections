from TrainingLoopInterface import TrainingLoopInterface

import numpy as np
import torch
from torch import nn, optim

class SentimentFineTuning(TrainingLoopInterface):
    def __init__(self, **kwargs):
        self.encoder = kwargs.get('encoder')
        self.optimizer = kwargs.get('optimizer')
        self.device = kwargs.get('device')
        self.scheduler = kwargs.get('scheduler')
        self.loss_fn = kwargs.get('loss_fn')
        self.len_train = kwargs.get('len_train')
        self.len_val = kwargs.get('len_val')

    def _train_epoch(self, **kwargs):
        data_loader = kwargs.get('data_loader')
        
        losses = []
        correct_predictions = 0
        
        # notify all layers we are in train mode
        # e.g., dropout layers will work in train mode instead of training mode
        self.encoder.model.train()

        for batch in data_loader:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['sentiments'].to(self.device)

            outputs = self.encoder.model(input_ids, attention_mask)

            # _ is a sequence of hidden states of the last layer of the model
            # preds is obtained after pooling with BertPooler on last_hidden_states
            # we pool the first output token
            _, preds = torch.max(outputs, dim=1) 
            loss = self.loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels) # -> Tensor
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.model.parameters(), max_norm=1.0) # avoid exploding gradients
            self.optimizer.step()
            self.scheduler.step()
        return correct_predictions.double() / self.len_train, np.mean(losses) # -> Tensor


    def _eval_epoch(self, **kwargs):
        data_loader = kwargs.get('data_loader')
        device = kwargs.get('device')
        
        correct_predictions = 0

        # notify all layers we are in eval mode
        # e.g., dropout layers will work in eval mode instead of training mode
        self.encoder.model.eval()

        # torch.no_grad() impacts the autograd engine and deactivate it 
        # reduce memory usage and speed up
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['sentiments'].to(self.device)

                outputs = self.encoder.model(input_ids, attention_mask)

                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)

        return correct_predictions.double() / self.len_val
 

    def trainloop(self, train_dataloader, val_dataloader, epochs, classifier_name, **kwargs):
        results = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': []
        }

        best_accuracy = 0

        for epoch in range(1, epochs+1):
            print(f'Epoch: {epoch}/{epochs}')
            print('-'*10)

            train_acc, train_loss = self._train_epoch(data_loader=train_dataloader)
            print(f'Train accuracy: {train_acc} Train loss: {train_loss} ')

            val_acc = self._eval_epoch(data_loader=val_dataloader)
            print(f'Val   accuracy: {val_acc}\n')

            results['train_acc'].append(train_acc)
            results['train_loss'].append(train_loss)
            results['val_acc'].append(val_acc)

            if val_acc > best_accuracy:
                path = f"/content/drive/My Drive/"+classifier_name+'.bin'
                torch.save(self.encoder.model.state_dict(), path)
                best_accuracy = val_acc

        return results
        
        