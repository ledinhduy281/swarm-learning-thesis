import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import load_img, img_to_array
from dataset import Dataset

class NIHChestXRaysDataset(Dataset):
    def __init__(self, directory: str):
        self.directory = directory
        
        entries = pd.read_csv(f"{directory}/labels.csv")
        self.image_names = entries["Image Index"].to_numpy()
        self.labels = tf.convert_to_tensor(
            np.stack(
                entries["Finding Labels"].apply(
                    lambda string: np.array([float(e) for e in string[1:-1].split(", ")])
                )
            )
        )
    
    def __len__(self):
        return self.image_names.size
    
    def __getitem__(self, index: int):
        label = self.labels[index]
        img = img_to_array(load_img(f"{self.directory}/images/{self.image_names[index]}", color_mode = "grayscale"))
        
        return img, label
