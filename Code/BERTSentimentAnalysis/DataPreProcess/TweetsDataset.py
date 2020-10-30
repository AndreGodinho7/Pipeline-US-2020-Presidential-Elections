import torch
from torch.utils.data import Dataset

class TweetsDataset(Dataset):
    """
    An abstract class representing a Dataset.

    All datasets that represent a map from keys to data samples should subclass it. 
    All subclasses should overwrite __getitem__(), supporting fetching a data sample for a given key. 
    Subclasses could also optionally overwrite __len__(), which is expected to 
    return the size of the dataset by many Sampler implementations and the default options of DataLoader.
    """
    def __init__(self, ids, tweets):
        self.ids = ids # list of ids
        self.tweets = tweets # list of strings

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        id = self.ids[index]
        tweet = self.tweets[index]

        return {
            'ids': id,
            'tweets': tweet
        }