# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import time

from . import cfg
from .resnet import *
from .log import Log

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

class Dataset:
    """Class responsible for encapsulating all the required data."""

    def __init__(self, training_examples, training_labels, testing_examples, testing_labels,
     validation_data=None, validation_split=0.1):
        self.x_train = training_examples
        self.y_train = training_labels
        self.x_test = testing_examples
        self.y_test = testing_labels
        self.validation_data = validation_data
        self.validation_split = validation_split


class BaseBackend(ABC):
    """Abstract class used to define Backend API."""

    def __init__(self, dataset, optimizer=None):
        self.dataset = dataset
        self.optimizer = optimizer

    @abstractmethod
    def generate_model(self, path):
        """Create and return a backend model representation.

        Args:
            path [Node]: list of nodes where each node represents a single
            network layer, the path starts with InputNode and ends with EndNode.
        Returns:
            model which represents neural network structure in the implemented
            backend, this model can be evaluated using evaluate_model method.
        """

    @abstractmethod
    def reuse_model(self, old_model, new_model_path, distance):
        """Create a new model by reusing layers (and their weights) from the old model.

        Args:
            old_model: old model which represents neural network structure.
            new_model_path [Node]: path representing new model.
            distance (int): distance which shows how many layers from old model need
            to be removed in order to create a base for new model i.e. if old model is
            NodeA->NodeB->NodeC->NodeD and new model is NodeA->NodeB->NodeC->NodeE,
            distance = 1.
        Returns:
            model which represents neural network structure.
        """

    @abstractmethod
    def train_model(self, model):
        """Train model which was created using generate_model method.

        Args:
            model: model which represents neural network structure.
        Returns:
            model which represents neural network structure.
        """

    @abstractmethod
    def fully_train_model(self, model, epochs, augment):
        """Fully trains the model without early stopping. At the end of the
        training, the model with the best performing weights on the validation
        set is returned.

        Args:
            model: model which represents neural network structure.
            epochs (int): for how many epoch train the model.
            augment (kwargs): augmentation arguments.
        Returns:
            model which represents neural network structure.
        """

    @abstractmethod
    def evaluate_model(self, model):
        """Evaluate model which was created using generate_model method.

        Args:
            model: model which represents neural network structure.
        Returns:
            loss & accuracy tuple.
        """

    @abstractmethod
    def save_model(self, model, path):
        """Saves model on disk.

        Args:
            model: model which represents neural network structure.
            path: string which represents model location.
        """

    @abstractmethod
    def load_model(self, path):
        """Load model from disk, in case of fail should return None.

        Args:
            path: string which represents model location.
        Returns:
            model: model which represents neural network structure, or in case
            fail None.
        """

    @abstractmethod
    def free_gpu(self):
        """Frees GPU memory."""


class TFKerasBackend(BaseBackend):
    """Backend based on TensorFlow Keras API"""

    def __init__(self, dataset, optimizer=None):
        # If the user passes custom optimizer we serialize it, as reusing the
        # same optimizer instance causes crash in TensorFlow  1.13.1, see issue
        # https://github.com/Pattio/DeepSwarm/issues/3
        if optimizer is not None:
            optimizer = tf.keras.optimizers.serialize(optimizer)

        super().__init__(dataset, optimizer)
        self.data_format = K.image_data_format()

    def generate_model(self, path):
        # print('GENERATE')
        # Create an input layer
        input_layer = self.create_layer(path[0])
        layer = input_layer
        nb_layers = 1
        # Convert each node to layer and then connect it to the previous layer
        for node in path[1:]:
            # Log.warning(node)
            if node.format == 'block':
                x = self.create_block(node)(layer)
                layer = x[0]
                nb_layers += x[1]
            else:
                layer = self.create_layer(node)(layer)
                nb_layers += 1
            # Log.warning(layer.name)

        # Return generated model
        model = tf.keras.Model(inputs=input_layer, outputs=layer)
        self.compile_model(model)
        return model, nb_layers

    def reuse_model(self, old_model, new_model_path, distance):
        # print('REUSE')
        # Find the starting point of the new model
        starting_point = len(new_model_path) - distance
        x = self.generate_model(new_model_path[:starting_point])[1]
        last_layer = old_model.layers[x - 1].output
        Log.debug((starting_point, x, last_layer))
        for x in new_model_path[:starting_point]:
            Log.debug(str(x))
        # Append layers from the new model to the old model
        for node in new_model_path[starting_point:]:
            # Log.warning(node)
            if node.format == 'block':
                last_layer = self.create_block(node)(last_layer)[0]
            else:
                last_layer = self.create_layer(node)(last_layer)
            # Log.warning(last_layer.name)

        # Return new model
        model = tf.keras.Model(inputs=old_model.inputs, outputs=last_layer)
        self.compile_model(model)
        return model

    def compile_model(self, model):
        optimizer_parameters = {
            'optimizer': 'adam',
            'loss': cfg['backend']['loss'],
            'metrics': ['accuracy'],
        }

        # If user specified custom optimizer, use it instead of the default one
        # we also need to deserialize optimizer as it was serialized during init
        if self.optimizer is not None:
            optimizer_parameters['optimizer'] = tf.keras.optimizers.deserialize(self.optimizer)
        model.compile(**optimizer_parameters)

    def create_layer(self, node):
        # Workaround to prevent Keras from throwing an exception ("All layer
        # names should be unique.") It happens when new layers are appended to
        # an existing model, but Keras fails to increment repeating layer names
        # i.e. conv_1 -> conv_2
        parameters = {'name': str(time.time())}

        if node.type == 'Input':
            parameters['shape'] = node.shape
            return tf.keras.Input(**parameters)

        if node.type == 'Conv2D':
            parameters.update({
                'filters': node.filter_count,
                'kernel_size': node.kernel_size,
                'padding': 'same',
                'data_format': self.data_format,
                'activation': self.map_activation(node.activation),
            })
            return tf.keras.layers.Conv2D(**parameters)

        if node.type == 'Pool2D':
            parameters.update({
                'pool_size': node.pool_size,
                'strides': node.stride,
                'padding': 'same',
                'data_format': self.data_format,
            })
            if node.pool_type == 'max':
                return tf.keras.layers.MaxPooling2D(**parameters)
            elif node.pool_type == 'average':
                return tf.keras.layers.AveragePooling2D(**parameters)

        if node.type == 'BatchNormalization':
            return tf.keras.layers.BatchNormalization(**parameters)

        if node.type == 'Flatten':
            return tf.keras.layers.Flatten(**parameters)

        if node.type == 'Dense':
            parameters.update({
                'units': node.output_size,
                'activation': self.map_activation(node.activation),
            })
            return tf.keras.layers.Dense(**parameters)

        if node.type == 'Dropout':
            parameters.update({
                'rate': node.rate,
            })
            return tf.keras.layers.Dropout(**parameters)

        if node.type == 'Output':
            parameters.update({
                'units': node.output_size,
                'activation': self.map_activation(node.activation),
            })
            return tf.keras.layers.Dense(**parameters)

        if node.type == "Skip":
            parameters.update({
                'function': lambda x: x
            })
            return tf.keras.layers.Lambda(**parameters)

        raise Exception(f'Not handled node type: {str(node)}')

    def _shortcut(self, input, residual):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        layers = 0
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = tf.keras.layers.Conv2D(filters=residual_shape[3],
                            kernel_size=(1, 1),
                            strides=(stride_width, stride_height),
                            padding="valid",
                            kernel_initializer="he_normal",
                            name=str(time.time()),
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)
            layers += 1
        return tf.keras.layers.add([shortcut, residual]), (layers + 1)

    def create_block(self, node):
        parameters = {'name': str(time.time())}
        def f(input):
            if node.type == 'resConv2D':
                parameters.update({
                    'filters': node.filter_count,
                    'kernel_size': node.kernel_size,
                    'padding': 'same',
                    'data_format': self.data_format,
                    'kernel_initializer': node.kernel_initializer,
                    'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
                })
                num_layers = node.layers
                # conv1, layers = full_preactivation_resnetBlock(num_layers, **parameters)(input)
                # resconv, shortcut_layers = self._shortcut(input, conv1)
                # return resconv, (layers + shortcut_layers)
                resconv, layers = cspResnetBlock(num_layers, **parameters)(input)
                return resconv, layers

            elif node.type == 'resNeXt':
                parameters.update({
                    'filters': node.filter_count,
                    'kernel_size': node.kernel_size,
                    'padding': 'same',
                    'strides': node.strides,
                    'data_format': self.data_format,
                    'kernel_initializer': node.kernel_initializer,
                    'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
                })
                cardinality = node.cardinality
                block, layers = resneXtBlock(cardinality, **parameters)(input)
                resblock, shortcut_layers = self._shortcut(input, block)
                return resblock, (layers + shortcut_layers)
            
            elif node.type == 'Conv2DBNRelu':
                parameters.update({
                    'filters': node.filter_count,
                    'kernel_size': node.kernel_size,
                    'padding': 'same',
                    'strides': node.strides,
                    'data_format': self.data_format,
                    'kernel_initializer': node.kernel_initializer,
                    'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
                })
                block, layers = Conv2D_BN_ReluBlock(**parameters)(input)
                return block, layers
            
            elif node.type == 'resDense':
                parameters.update({
                    'filters': node.filter_count,
                    'kernel_size': node.kernel_size,
                    'padding': 'same',
                    'data_format': self.data_format,
                    'kernel_initializer': node.kernel_initializer,
                    'kernel_regularizer': tf.keras.regularizers.l2(1e-4),
                    'rate': node.rate,
                })
                num_layers = node.layers
                # block, layers = denseBlock(num_layers, **parameters)(input)
                block, layers =  cspDenseBlock(num_layers, **parameters)(input)
                return block, layers
            else:
                raise Exception(f'Not handled node type: {str(node)}')

        return f

    def map_activation(self, activation):
        if activation == "ReLU":
            return tf.keras.activations.relu
        if activation == "ELU":
            return tf.keras.activations.elu
        if activation == "LeakyReLU":
            return tf.nn.leaky_relu
        if activation == "Sigmoid":
            return tf.keras.activations.sigmoid
        if activation == "Softmax":
            return tf.keras.activations.softmax
        raise Exception(f'Not handled activation: {str(activation)}')

    def train_model(self, model):
        # Create a checkpoint path
        checkpoint_path = 'temp-model'

        # Setup training parameters
        fit_parameters = {
            'x': self.dataset.x_train,
            'y': self.dataset.y_train,
            'epochs': cfg['backend']['epochs'],
            'batch_size': cfg['backend']['batch_size'],
            'callbacks': [
                self.create_early_stop_callback(),
                self.create_checkpoint_callback(checkpoint_path),
            ],
            'validation_split': self.dataset.validation_split,
            'verbose': cfg['backend']['verbose'],
        }

        # If validation data is given then override validation_split
        if self.dataset.validation_data is not None:
            fit_parameters['validation_data'] = self.dataset.validation_data

        # Train model
        model.fit(**fit_parameters)

        # Load model from checkpoint
        checkpoint_model = self.load_model(checkpoint_path)
        # Delete checkpoint
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
        # Return checkpoint model if it exists, otherwise return trained model
        return checkpoint_model if checkpoint_model is not None else model

    def fully_train_model(self, model, epochs, augment):
        # Setup validation data
        if self.dataset.validation_data is not None:
            x_val, y_val = self.dataset.validation_data
            x_train, y_train = self.dataset.x_train, self.dataset.y_train
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                self.dataset.x_train,
                self.dataset.y_train,
                test_size=self.dataset.validation_split,
            )

        # Create checkpoint path
        checkpoint_path = 'temp-model'

        # Create and fit data generator
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(**augment)
        datagen.fit(x_train)

        # Train model
        model.fit_generator(
            generator=datagen.flow(x_train, y_train, batch_size=cfg['backend']['batch_size']),
            steps_per_epoch=len(self.dataset.x_train) / cfg['backend']['batch_size'],
            epochs=epochs,
            callbacks=[self.create_checkpoint_callback(checkpoint_path)],
            validation_data=(x_val, y_val),
            verbose=cfg['backend']['verbose'],
        )

        # Load model from checkpoint
        checkpoint_model = self.load_model(checkpoint_path)
        # Delete checkpoint
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
        # Return checkpoint model if it exists, otherwise return trained model
        return checkpoint_model if checkpoint_model is not None else model

    def create_early_stop_callback(self):
        early_stop_parameters = {
            'patience': cfg['backend']['patience'],
            'verbose': cfg['backend']['verbose'],
            'restore_best_weights': True,
        }
        early_stop_parameters['monitor'] = 'val_loss' if cfg['metrics'] == 'loss' else 'val_acc'
        return tf.keras.callbacks.EarlyStopping(**early_stop_parameters)

    def create_checkpoint_callback(self, checkpoint_path):
        checkpoint_parameters = {
            'filepath': checkpoint_path,
            'verbose': cfg['backend']['verbose'],
            'save_best_only': True,
        }
        checkpoint_parameters['monitor'] = 'val_loss' if cfg['metrics'] == 'loss' else 'val_acc'
        return tf.keras.callbacks.ModelCheckpoint(**checkpoint_parameters)

    def evaluate_model(self, model):
        loss, accuracy = model.evaluate(
            x=self.dataset.x_test,
            y=self.dataset.y_test,
            verbose=cfg['backend']['verbose']
        )
        return (loss, accuracy)

    def save_model(self, model, path):
        model.save(str(path))
        self.free_gpu()

    def load_model(self, path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except:
            return None

    def free_gpu(self):
        K.clear_session()
