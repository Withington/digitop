"""Test the digitop functions."""
import unittest

from .context import digitop

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_version(self):
        """Test version number display."""
        print(digitop.__version__)
        assert digitop.__version__ is not None

    def test_tensorflow(self):
        """Test TensorFlow session."""
        assert digitop.run_tensorflow().decode('utf8') == "Hello TensorFlow"

    def test_keras(self):
        """Test Keras model."""
        assert digitop.build_keras_model() == 16*32+16

    def test_classifier(self):
        """Test the classifier Keras neural network.
        It doesn't give good accuracy but this tests that it runs."""
        score = digitop.classifier()
        print("Score is ")
        print(score)
        accuracy = score[1]
        self.assertGreater(accuracy, 0.04)

if __name__ == '__main__':
    unittest.main()
