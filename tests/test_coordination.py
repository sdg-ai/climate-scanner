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
		assert EnrichmentCoordinator

	# check entire pipeline runs
	def test_pipeline(self):
		ec = EnrichmentCoordinator()
		example_data = json.load(open(get_data('example_input.jsonl'),
									  'rt', encoding='utf-8', errors='ignore'))
		output_data = ec.process(example_data)
		assert isinstance(output_data, dict)
		assert 'ID' in output_data

	# check different sections of enrichment pipeline input/output
	def test_inputs_ouputs(self):
		ec = EnrichmentCoordinator()
		input_json = json.load(open(get_data('example_input.jsonl'),
									'rt', encoding='utf-8', errors='ignore'))

		output_1 = ec.coarse_classifier.predict(input_json)

		assert isinstance(output_1, dict)
		assert 'ID' in output_1
		assert 'climate_scanner_prob' in output_1
		assert isinstance(output_1['climate_scanner_prob'], float)
		assert output_1['climate_scanner_prob'] >= 0
		assert output_1['climate_scanner_prob'] <= 1

		output_2 = ec.i_t_classifer.coordinator_pipeline(output_1, 0.5)
		# TODO: add some checks

		output_3 = ec.sentiment_classifier.text_to_sentiment(output_2)
		# TODO: add some checks