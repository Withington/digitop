import tensorflow as tf

def run_tensorflow():
    """Run a simple TensorFlow session.
    Check that the session runs as expected.
    >>> run_tensorflow()
    b'Hello TensorFlow'
    """
    hello = tf.constant("Hello TensorFlow")
    sess = tf.Session()
    return sess.run(hello)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

