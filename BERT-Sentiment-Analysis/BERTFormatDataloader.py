import Data, DataProcessInterface

from torch.utils.data import DataLoader

class BERTFormatDataloader:

    def __init__(self, dataset, batch_size, num_GPUs):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4*num_GPUs)

    def getDataloader(self):
        return self.dataloader
    