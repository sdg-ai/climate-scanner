# -*- coding: utf-8 -*-

import os
import json
from climate_scanner.coarse_article_classifier.coarse_classifier import Doc2Climate
from climate_scanner.trends_innovation_classifier.trends_innovation_classifier import Doc2Trends
from climate_scanner.sentiment_classifier.sentiment_classifier import SentimentInterface


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
		self.d2c = Doc2Climate()
		self.d2t = Doc2Trends()
		self.d2s = SentimentInterface()
		self.entity_recognition = None

	def process(self, input_json):
		output_1 = self.d2c.get_climate_class(input_json)
		print('1: ', output_1)
		output_2 = self.d2t.coordinator_pipeline(output_1, 0.5)
		print('2: ', output_2)
		output_3 = self.d2s.text_to_sentiment(output_2)
		print('3: ', output_3)
		return output_3


def run_example():
	ec = EnrichmentCoordinator()
	example_data = json.load(open(get_data('example_input.jsonl'), 'rt', encoding='utf-8', errors='ignore'))

	print('=============	Doc 	=============')
	print('ID:', '\t', example_data['ID'])
	print('Title:', '\t', example_data['title'])
	print('Document:', '\t', example_data['doc'])

	output_data = ec.process(example_data)

	print('=============	Enriched Doc 	=============')
	print(json.dumps(output_data))


if __name__ == '__main__':
	run_example()

