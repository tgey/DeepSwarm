# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import context
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

import os, sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from deepswarm.backends import Dataset, TFKerasBackend
from deepswarm.deepswarm import DeepSwarm
from deepswarm.log import Log
import traceback

# Load CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# Create dataset object, which controls all the data
dataset = Dataset(
    training_examples=x_train,
    training_labels=y_train,
    testing_examples=x_test,
    testing_labels=y_test,
    validation_split=0.1,
)
# Create backend responsible for training & validating
backend = TFKerasBackend(dataset=dataset)
# Create DeepSwarm object responsible for optimization
deepswarm = DeepSwarm(backend=backend)
# Find the topology for a given dataset
try:
    topology = deepswarm.find_topology()
except:
    Log.error(f'{sys.exc_info()} occured')
    Log.error(f'{traceback.format_exc()}')
# Evaluate discovered topology
deepswarm.evaluate_topology(topology)
# Train topology on augmented data for additional 50 epochs
trained_topology = deepswarm.train_topology(topology, 50, augment={
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
})
# Evaluate the final topology
deepswarm.evaluate_topology(trained_topology)
