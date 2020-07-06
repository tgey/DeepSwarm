import time
import copy
import tensorflow as tf

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = tf.keras.layers.BatchNormalization(name=str(time.time()), axis=3)(input)
    parameters= ({
        'name': str(time.time()),
    })
    return tf.keras.layers.Activation("relu",**parameters)(norm)


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
        nb_layers = 3
        block = BN_Relu_Conv2DBlock(**parameters)(input)
        for _ in range(num_layers - 1):
            parameters.update({
                        'name': str(time.time()),
                    })
            block = BN_Relu_Conv2DBlock(**parameters)(block)
            nb_layers += 3
        return block, nb_layers
    return f

def denseBlock(num_layers=2, **parameters):
    """https://stackoverflow.com/a/60225459"""
    def f(input):
        layers_concat = list()
        layers_concat.append(input)
        nb_layers = 4
        block = BottleneckBlock(**parameters)(input)
        layers_concat.append(block)

        for _ in range(1, num_layers):
            block = tf.keras.layers.Concatenate(name=str(time.time()), axis=3)(copy.copy(layers_concat))
            parameters.update({'name': str(time.time())})
            block = BottleneckBlock(**parameters)(block)
            nb_layers += 5
            layers_concat.append(block)
        return tf.keras.layers.Concatenate(name=str(time.time()), axis=3)(copy.copy(layers_concat)), (nb_layers + 1)
    return f

def Conv2D_BN_ReluBlock(**parameters):
    def f(input):
        conv = tf.keras.layers.Conv2D(**parameters)(input)
        return _bn_relu(conv), 3
    return f

def resneXtBlock(cardinality: int, **parameters):
    def f(input): # TODO return nb_layers
        group_list = []
        grouped_channels = int(parameters['filters'] / cardinality)
        parameters['filters'] = grouped_channels
        
        for c in range(cardinality):
            lambda_params = {'name': str(time.time())}
            x = tf.keras.layers.Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels], **lambda_params)(input)
            parameters.update({'name': str(time.time())})
            conv = tf.keras.layers.Conv2D(**parameters)(x)
            group_list.append(conv)
        
        group_merge = tf.keras.layers.Concatenate(name=str(time.time()), axis=-1)(copy.copy(group_list))
        return _bn_relu(group_merge)
    return f

######## CROSS STAGE PARTIAL CONNECTIONS BLOCKS #########

def cspDenseBlock(num_layers=2, **parameters):
    def f(input):
        lambda_params = {'name': str(time.time())}
        inputs = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input)
        nb_layers = 1
        block, layers = denseBlock(num_layers, **parameters)(inputs[1])
        return tf.keras.layers.concatenate(copy.copy([inputs[0], block]), axis=-1), (nb_layers + layers + 1)
    return f

def cspResnetBlock(num_layers=2, **parameters):
    def f(input):
        inputs = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input)
        nb_layers = 1
        block, layers = full_preactivation_resnetBlock(num_layers, **parameters)(inputs[1])
        return tf.keras.layers.concatenate(copy.copy([inputs[0], block]), axis=-1), (nb_layers + layers + 1)
    return f

def cspResneXtBlock(cardinality, **parameters):
    def f(input):
        inputs = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input)
        nb_layers = 1
        block, layers = resneXtBlock(cardinality, **parameters)(inputs[1])
        return tf.keras.layers.concatenate(copy.copy([inputs[0], block]), axis=-1), (nb_layers + layers + 1)
    return f