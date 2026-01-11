"""
ChunIVision package entry point.

Allows running the package directly with:
    python -m chunivision
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())
