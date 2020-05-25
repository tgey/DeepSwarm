import tensorflow as tf

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = tf.keras.layers.BatchNormalization(axis=3)(input)
    return tf.keras.layers.Activation("relu")(norm)


def Conv2DBlock(**parameters):
    def f(input):
        return tf.keras.layers.Conv2D(**parameters)(input)
    return f

def DenseBlock(**parameters):
    def f(input):
        return tf.keras.layers.Dense(**parameters)(input)
    return f

def BN_Relu_Conv2DBlock(**parameters):
    def f(input):
        activation = _bn_relu(input)
        return tf.keras.layers.Conv2D(**parameters)(activation)
    return f

def originalResNetBlock(**parameters):
    def f(input):
        activation1 = _bn_relu(input)
        conv1 = tf.keras.layers.Conv2D(**parameters)(activation1)
        activation2 = _bn_relu(conv1)
        return tf.keras.layers.Conv2D(**parameters)(activation2)
    return f

def Conv2D_BN_ReluBlock(**parameters):
    def f(input):
        conv = tf.keras.layers.Conv2D(**parameters)(input)
        return _bn_relu(conv)
    return f