from .context import digitop

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_smoke(self):
      print(digitop.digimodel())
      assert digitop.digimodel() == "Hello model"


if __name__ == '__main__':
    unittest.main()
