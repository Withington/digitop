import tensorflow as tf


def digimodel():
  return "Hello model"


def run_tensorflow():
  hello = tf.constant("Hello TensorFlow")
  sess = tf.Session()
  return sess.run(hello)