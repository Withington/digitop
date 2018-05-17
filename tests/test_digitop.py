from .context import digitop

import unittest


class BasicTestSuite(unittest.TestCase):

    def test_smoke(self):
      assert digitop.digimodel() == "Hello model"

    def test_tensorflow(self):
      assert digitop.run_tensorflow().decode('utf8') == "Hello TensorFlow"


if __name__ == '__main__':
    unittest.main()
