import sys
import os

# Make the prototype package root importable when pytest is run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
