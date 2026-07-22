# pytest kök conftest — repo kökünü sys.path'e ekler ki tests/ altındaki testler
# `rag` paketini (from rag.core import ...) import edebilsin.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
