import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from tensorflow.keras.utils import Sequence

class Dataset:
    def __len__(self):
        pass
    
    def __getitem__(self, index: int):
        pass
    
class Subdataset(Dataset):
    def __init__(self, upstream: Dataset, indices: np.array):
        self.upstream = upstream
        self.indices = indices
    
    def __len__(self):
        return self.indices.size
    
    def __getitem__(self, index: int):
        return self.upstream[self.indices[index]]

def train_test_split(dataset: Dataset, train_ratio: float = 0.5):
    train_indices, test_indices = sklearn_train_test_split(
        np.arange(len(dataset)),
        train_size = train_ratio,
        random_state = 4,
        shuffle = True,
    )

    train_dataset = Subdataset(dataset, np.array(train_indices))
    test_dataset = Subdataset(dataset, np.array(test_indices))
    
    return train_dataset, test_dataset

class DatasetGenerator(Sequence):
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __getitem__(self, index: int):
        pairs = [
            self.dataset[index]
            for index in range(self.batch_size * index, self.batch_size * (index + 1))
        ]
        
        X = tf.stack([data for (data, _) in pairs])
        y = tf.stack([label for (_, label) in pairs])
        
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)