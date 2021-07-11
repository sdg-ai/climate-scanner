# -*- coding: utf-8 -*-

import os
# from climate_scanner.coarse_article_classifier import...
# from climate_scanner.trends_innovation_classifier import...
# from climate_scanner.sentiment import...


#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
	return os.path.join(_ROOT, 'data', path)


class EnrichmentCoordinator:
	def __init__(self):
		self.coarse_classifier = None
		self.innovations_classifer = None
		self.trends_classifier = None
		self.sentiment_classifier = None
		self.entity_recognition = None

	def process(self):
		pass
