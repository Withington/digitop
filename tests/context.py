"""Determine the library path."""
import sys
import os

import digitop

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
