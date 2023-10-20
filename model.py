from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Dense
from tensorflow.keras.applications.densenet import DenseNet121

def NIHChestXRaysDense121():
    rescaling = Rescaling(1. / 255)

    backbone = DenseNet121(
        include_top = False,
        weights = None,
        input_shape = (128, 128, 1),
        pooling = "avg"
    )

    head = Dense(14, activation = "sigmoid")

    return Sequential([rescaling, backbone, head])