from DataProcessInterface import DataProcessInterface

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

import logging
logging.basicConfig(level=logging.ERROR)

PRE_TRAINED_MODEL_NAME = 'bert-base-cased' # cased letters are important for Sentiment analysis

class BERTFormatDataset(Dataset):
    """
    An abstract class representing a Dataset.

    All datasets that represent a map from keys to data samples should subclass it. 
    All subclasses should overwrite __getitem__(), supporting fetching a data sample for a given key. 
    Subclasses could also optionally overwrite __len__(), which is expected to 
    return the size of the dataset by many Sampler implementations and the default options of DataLoader.
    """
    def __init__(self, data, max_len):
        self.messages = data.content.to_numpy()
        self.sentiments = data.sentiment.to_numpy()
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.max_len = max_len

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        def dataPreprocess(message):
            encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
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