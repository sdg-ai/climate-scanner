# -*- coding: utf-8 -*-

import os

#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_data(path):
	return os.path.join(_ROOT, 'data', path)

class classifier:
    # class defining the sentiment classifier model
    # methods:
    #     method1:...
    def __init__(self):
        pass

