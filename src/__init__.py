import inspect
import os
import sys


path_ = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
sys.path.append(path_)
