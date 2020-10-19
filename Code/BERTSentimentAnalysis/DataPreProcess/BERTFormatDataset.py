import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

import pandas as pd

class BERTFormatDataset(Dataset):
    """
    An abstract class representing a Dataset.

    All datasets that represent a map from keys to data samples should subclass it. 
    All subclasses should overwrite __getitem__(), supporting fetching a data sample for a given key. 
    Subclasses could also optionally overwrite __len__(), which is expected to 
    return the size of the dataset by many Sampler implementations and the default options of DataLoader.
    """
    def __init__(self, max_len, encoder, messages, labels):
        # self.messages = dataframe.content.to_numpy()
        # self.sentiments = data.sentiment.to_numpy()
        self.messages = messages
        self.sentiments = labels
        self.max_len = max_len
        self.encoder = encoder

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        def dataPreprocess(message):
            encoding = self.encoder.tokenizer(
                        message,
                        return_tensors='pt',
                        pad_to_max_length=True,
                        truncation=True,
                        max_length=self.max_len,
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        add_special_tokens=True
            )
            return encoding

        message = str(self.messages[index]) 
        sentiment = self.sentiments[index]

        encoding = dataPreprocess(message)

        return {
        'message': message,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'sentiments': torch.tensor(sentiment, dtype=torch.long)
        }