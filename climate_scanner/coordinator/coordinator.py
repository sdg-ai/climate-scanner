# -*- coding: utf-8 -*-

import os
import json
from climate_scanner.coarse_article_classifier.coarse_classifier import Classifier
from climate_scanner.trends_innovation_classifier.trends_innovation_classifier import Doc2Trend
from climate_scanner.sentiment.sentiment_classifier import interface


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
		self.coarse_classifier = Classifier()
		self.i_t_classifer = Doc2Trend()
		self.sentiment_classifier = interface()
		self.entity_recognition = None

	def process(self, input_json):
		output_1 = self.coarse_classifier.predict(input_json)

		output_2 = self.i_t_classifer.coordinator_pipeline(output_1, 0.5)

		output_3 = self.sentiment_classifier.text_to_sentiment(output_2)

		return output_3


def run_example():
	ec = EnrichmentCoordinator()
	example_data = json.load(open(get_data('example_input.jsonl'), 'rt', encoding='utf-8', errors='ignore'))

	print('=============	Doc 	=============')
	print('ID:', '\t', example_data['id'])
	print('Title:', '\t', example_data['title'])
	print('Document:', '\t', example_data['Document'])

	output_data = ec.process(example_data)

	print('=============	Enriched Doc 	=============')
	print(json.dumps(output_data))


if __name__ == '__main__':
	run_example()

