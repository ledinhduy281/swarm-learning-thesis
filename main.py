from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from nih_dataset import NIHChestXRaysDataset
from dataset import DatasetGenerator, train_test_split
from model import NIHChestXRaysDense121

import os
import logging
from swarmlearning.tf import SwarmCallback

epochs = int(os.getenv("MAX_EPOCHS", str(20)))
minPeers = int(os.getenv("MIN_PEERS", str(2)))

nih_chest_xrays_dataset = NIHChestXRaysDataset("/tmp/dataset")

train_dataset, val_test_dataset = train_test_split(nih_chest_xrays_dataset)
val_dataset, test_dataset = train_test_split(val_test_dataset)

train_generator = DatasetGenerator(train_dataset, 32)
val_generator = DatasetGenerator(val_dataset, 32)
test_generator = DatasetGenerator(test_dataset, 32)

model = NIHChestXRaysDense121()

model.compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = ["binary_accuracy"]
)

early_stopping = EarlyStopping(
    monitor = "val_binary_accuracy",
    patience = 3
)

model_checkpoint = ModelCheckpoint(
    filepath = "/tmp/working/checkpoint/nih-xrays-swarm-checkpoint.h5",
    monitor = "val_binary_accuracy",
    save_best_only = True
)

swarm_callback = SwarmCallback(
    syncFrequency = 128,
    minPeers = minPeers,
    useAdaptiveSync = False,
    adsValData = val_generator
)

swarm_callback.logger.setLevel(logging.DEBUG)

model.fit(
    train_generator,
    epochs = epochs,
    validation_data = val_generator,
    callbacks = [
        early_stopping,
        model_checkpoint,
        swarm_callback
    ]
)

if not os.path.isdir("/tmp/working/model/saved_models"):
    os.makedirs("/tmp/working/model/saved_models")

swarm_callback.logger.info("Saving the final Swarm model ...")
model.save("/tmp/working/model/saved_models/nih-xrays-swarm.h5")
swarm_callback.logger.info("Saved the trained model - /tmp/working/model/saved_models/nih-xrays-swarm.h5")


swarm_callback.logger.info("Starting inference on the test data ...")
test_loss, test_accuracy = model.evaluate(test_generator)
swarm_callback.logger.info("Test loss = %.5f" % (test_loss))
swarm_callback.logger.info("Test accuracy = %.5f" % (test_accuracy))
