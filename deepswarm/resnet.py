import tensorflow as tf

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = tf.keras.layers.BatchNormalization(axis=3)(input)
    return tf.keras.layers.Activation("relu")(norm)


def resConv2DBlock(**parameters):
    def f(input):
        return tf.keras.layers.Conv2D(**parameters)(input)
    return f