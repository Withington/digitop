import unittest

from .context import digitop

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_tensorflow(self):
        """Test TensorFlow session."""
        assert digitop.run_tensorflow().decode('utf8') == "Hello TensorFlow"


if __name__ == '__main__':
    unittest.main()