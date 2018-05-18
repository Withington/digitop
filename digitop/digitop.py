import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

def run_tensorflow():
    """Run a simple TensorFlow session.
    Check that the session runs as expected.
    >>> run_tensorflow()
    b'Hello TensorFlow'
    """
    hello = tf.constant("Hello TensorFlow")
    sess = tf.Session()
    return sess.run(hello)

def build_keras_model():
    """Build a simple Keras model and return the number of parameters.
    >>> build_keras_model()
    528
    """
    model = Sequential()
    model.add(Dense(16, input_dim=32))
    return model.count_params()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
