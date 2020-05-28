import tensorflow as tf
import time
import copy

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = tf.keras.layers.BatchNormalization(name=str(time.time()), axis=3)(input)
    return tf.keras.layers.Activation("relu")(norm)


def Conv2DLayer(**parameters):
    def f(input):
        return tf.keras.layers.Conv2D(**parameters)(input)
    return f

def DenseLayer(**parameters):
    def f(input):
        return tf.keras.layers.Dense(**parameters)(input)
    return f

def BottleneckBlock(**parameters):
    def f(input):
        dropout_rate = parameters['rate']
        del parameters['rate']
        block = BN_Relu_Conv2DBlock(**parameters)(input)
        params = ({
                    'name': str(time.time()),
                    'rate': dropout_rate,
        })
        return tf.keras.layers.Dropout(**params)(block)
    return f
 
def BN_Relu_Conv2DBlock(**parameters):
    def f(input):
        activation = _bn_relu(input)
        return tf.keras.layers.Conv2D(**parameters)(activation)
    return f

def full_preactivation_resnetBlock(num_layers=2, **parameters):
    def f(input):
        block = BN_Relu_Conv2DBlock(**parameters)(input)
        for _ in range(num_layers - 1):
            parameters.update({
                        'name': str(time.time()),
                    })
            block = BN_Relu_Conv2DBlock(**parameters)(block)
        return block
    return f

def denseBlock(num_layers=2, **parameters):
    """https://stackoverflow.com/a/60225459"""
    def f(input):
        layers_concat = list()
        layers_concat.append(input)
        block = BottleneckBlock(**parameters)(input)
        layers_concat.append(block)

        for _ in range(1, num_layers):
            block = tf.keras.layers.concatenate(copy.copy(layers_concat), axis=3)
            parameters.update({'name': str(time.time())})
            block = BottleneckBlock(**parameters)(block)
            layers_concat.append(block)
        return tf.keras.layers.concatenate(copy.copy(layers_concat), axis=3)
    return f

def Conv2D_BN_ReluBlock(**parameters):
    def f(input):
        conv = tf.keras.layers.Conv2D(**parameters)(input)
        return _bn_relu(conv)
    return f