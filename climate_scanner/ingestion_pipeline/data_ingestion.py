# -*- coding: utf-8 -*-

import os
import psycopg2

#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
	return os.path.join(_ROOT, 'data', path)


class DataIngestion:
	def __init__(self):
		self.coarse_classifier = None

	def process(self):
		pass
