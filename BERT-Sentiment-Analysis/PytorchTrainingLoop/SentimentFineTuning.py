from TrainingLoopInterface import TrainingLoopInterface
from BERTSentimentClassifier import BERTSentimentClassifier

import torch


class SentimentFineTuning(TrainingLoopInterface):
    def __init__(self, **kwargs):
        self.optimizer = kwargs.get('optimizer')
        self.device = kwargs.get('device')
        self.scheduler = kwargs.get('scheduler')
        self.loss_fn = kwargs.get('loss_fn')
        self.len_train = kwargs.get('len_train')
        self.len_val = kwargs.get('len_val')

    def trainloop(self, classifier, train_dataloader, val_dataloader, epochs, classifier_name, **kwargs):
        results = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': []
        }

        best_accuracy = 0

        for epoch in range(1, epochs+1):
            print(f'Epoch: {epoch}/{epochs}')
            print('-'*10)

            train_acc, train_loss = classifier.train_epoch(data_loader=train_dataloader, 
                    loss_fn=self.loss_fn, optimizer=self.optimizer, device=self.device, 
                    scheduler=self.scheduler, n_examples=self.len_train)
            print(f'Train accuracy: {train_acc} Train loss: {train_loss} ')

            val_acc = classifier.eval_epoch(data_loader=val_dataloader, device=self.device, 
                                            n_examples=self.len_val)
            print(f'Val   accuracy: {val_acc}\n')

            results['train_acc'].append(train_acc)
            results['train_loss'].append(train_loss)
            results['val_acc'].append(val_acc)

            if val_acc > best_accuracy:
                path = f"/content/drive/My Drive/"+classifier_name+'.bin'
                torch.save(classifier.model.state_dict(), path)
                best_accuracy = val_acc

        return results
        
        