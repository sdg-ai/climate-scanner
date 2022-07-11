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


class Node:
	# Class defining the Node
	# Methods:
	#     method1:...
	# Attributes:
	def __init__(self):
		pass


class EntityGraph:
	# Class defining the Entity Graph
	# Methods:
	#     method1:...
	# Attributes:
	def __init__(self):
		pass


class GraphConstructor:
	# Class for faccilitating the construction of an entity graph
	# Methods:
	#     method1:...
	# Attributes:
	def __init__(self):
		pass