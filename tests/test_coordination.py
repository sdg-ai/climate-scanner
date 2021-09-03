# -*- coding: utf-8 -*-

###########################################################################
#
# #
# # Basic testing of enrichment pipeline coordination
# #
#
###########################################################################

###########################################################################
#
#   Necessary imports
#
###########################################################################

import pytest
import os
import json
from climate_scanner.coordinator.coordinator import EnrichmentCoordinator
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


###########################################################################
#
#   Test Class
#
###########################################################################

class TestCoordination:

	# class instantiates without exceptions
	def test_init(self):
		assert EnrichmentCoordinator()

	# class to check instantiation of interfaces
	def test_interfaces(self):
		assert Doc2Climate()
		assert Doc2Trends()
		assert SentimentInterface()

	# check entire pipeline runs
	def test_pipeline(self):
		ec = EnrichmentCoordinator()
		example_data = json.load(open(get_data('example_input.json'),
									  'rt', encoding='utf-8', errors='ignore'))
		output_data = ec.process(example_data)
		assert isinstance(output_data, list)
		assert isinstance(output_data[0], dict)
		assert 'ID' in output_data[0]

	# check different sections of enrichment pipeline input/output
	def test_inputs_ouputs(self):
		ec = EnrichmentCoordinator()
		input_json = json.load(open(get_data('example_input.json'),
									'rt', encoding='utf-8', errors='ignore'))

		# Generate climate class and check output format
		output_1 = Doc2Climate().get_climate_class(input_json)
		assert isinstance(output_1, dict)
		assert 'ID' in output_1
		assert 'climate_scanner_prob' in output_1
		assert type(output_1['climate_scanner_prob']) in [float, int]

		# Generate trends and innovations predictions, check format
		output_2 = Doc2Trends().coordinator_pipeline(output_1, 0.5)
		assert 'ID' in output_2[0]
		assert 'string_indices' in output_2[1]
		assert isinstance(output_2[1]['string_indices'][0], int)
		assert 'string_prediction' in output_2[1]
		assert isinstance(output_2[1]['string_prediction'], list)
		assert 'string_prob' in output_2[1]
		assert isinstance(output_2[1]['string_prob'][0], float)

		# Generate sentiment predictions and check format
		output_3 = SentimentInterface().text_to_sentiment(output_2)
		assert 'sentiment_class' in output_3[1]
		assert output_3[1]['sentiment_class'] in ['positive', 'negative', 'neutral']
		assert 'sentiment_proba' in output_3[1]
		assert isinstance(output_3[1]['sentiment_proba'], float)