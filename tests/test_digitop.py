"""Test the digitop functions."""
import unittest
import pydot

from tests.context import digitop

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
        """Test the classification model."""
        score, __ = digitop.evaluate_classifier()
        print("Score is ")
        print(score)
        accuracy = score[1]
        self.assertGreater(accuracy, 0.60)

    def test_graphviz(self):
        """ Test the pydot/graphviz installation by attempting
        to create an image of a blank graph.
        This test may fail on Windows (Message: [WinError 2] "dot.exe"
        not found in path). To have it work on Windows add it to the path. E.g.
        os.environ["PATH"] += os.pathsep + 'C:/.../Anaconda3/envs/.../Library/bin/graphviz').
        """
        pydot.Dot.create(pydot.Dot())
        assert True

if __name__ == '__main__':
    unittest.main()
