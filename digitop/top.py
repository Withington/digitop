"""Top level calls to TensorFlow and Keras."""
import subprocess

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

from digitop import classifier

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

def version():
    """Return the version number defined in the git repo. Or the last commit 
    date if there is no version number."""
    try:
        version_number = subprocess.check_output(['git', 'describe', '--exact-match'])
    except:
        version_number = subprocess.check_output(['git', 'log', '-1', 
                                                      '--format=%cd', '--date=local'])
    return version_number

def run_and_plot():
    """Run the model and plot its training."""
    score, history = classifier.evaluate_classifier()
    classifier.plot_history(history)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
