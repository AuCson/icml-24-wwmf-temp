from torch.utils.data import Dataset

class InstanceDataset(Dataset):
    def __init__(self, examples):
        super().__init__()

    @classmethod
    def load_data(cls, name):
        return InstanceDataset(name)