import tensorflow as tf

def run_tensorflow():
    """Run a simple TensorFlow session."""
    hello = tf.constant("Hello TensorFlow")
    sess = tf.Session()
    return sess.run(hello)
