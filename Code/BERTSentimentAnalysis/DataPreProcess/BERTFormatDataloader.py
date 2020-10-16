from torch.utils.data import DataLoader

class BERTFormatDataloader:
    """
    Combines a dataset and a sampler, and provides an iterable over the given dataset.
    
    The DataLoader supports both map-style and iterable-style datasets with single- 
    or multi-process loading, customizing loading order and optional automatic batching (collation)
    and memory pinning.
    """
    def __init__(self, dataset, batch_size, num_GPUs):
        if num_GPUs > 0:
            self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4*num_GPUs)
        else: 
            self.dataloader = DataLoader(dataset, batch_size=batch_size)

    def getDataloader(self):
        return self.dataloader
    