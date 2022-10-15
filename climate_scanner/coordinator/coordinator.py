# -*- coding: utf-8 -*-

import os
import json
from climate_scanner.entity_network.graph_demo.entity_extraction import EntityExtractor
from climate_scanner.noise_classifier.body_classifier import BodyNonbodyClassifier
from climate_scanner.sentiment_classifier.sentiment_classifier import SentimentInterface
from climate_scanner.trends_innovation_classifier.trends_innovation_classifier import TrendsInnovationClassifier
import nltk

# Initialize Graph Handler and Entity Extractor Objects
# graph = GraphConstructor()


#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    return os.path.join(_ROOT, 'data', path)


class EnrichmentCoordinator:
    def __init__(self, locs_disambiguation=False):
        # self.d2c = Doc2Climate()
        self.d2t = TrendsInnovationClassifier()
        self.d2s = SentimentInterface()
        self.entity_recognition = EntityExtractor()
        self.cleaner = BodyNonbodyClassifier()
        self.locs_flag = locs_disambiguation
        self.locs = None

        if self.locs_flag:
            from graphlocation.location import Locations
            self.locs = Locations()

    def dummy_process(self, input_json):
        output_1 = self.d2c.get_climate_class(input_json)
        print('1: ', output_1)
        output_2 = self.d2t.coordinator_pipeline(output_1, 0.5)
        print('2: ', output_2)
        output_3 = self.d2s.text_to_sentiment(output_2)
        print('3: ', output_3)
        return output_3

    def clean_text(self, text):
        sentences = nltk.sent_tokenize(text)
        cleaned_sentences = []
        for sentence in sentences:
            prediction = self.cleaner.predict(sentence)
            if prediction == 'BODY':
                cleaned_sentences.append(sentence)
        return ' '.join(cleaned_sentences)


    def process(self, article_text):
        article_text = self.clean_text(article_text)
        trend_results = self.d2t.scan_predict(article_text)
        sentiment = self.d2s.text_to_sentiment(article_text)[0]
        # entities = self.entity_recognition.get_annotations(article_text)
        entities = []
        entities_dedupe_set = set()
        for item in trend_results:
            item['extract_sentiment'] = self.d2s.text_to_sentiment(item['text'])
            entity_list = []
            for entity in self.entity_recognition.get_annotations(item['text']):
                if self.locs_flag:
                    if entity[2]['entityType'] and 'Place' in entity[2]['entityType']:
                        loc_class = self.locs.get_location(entity[0])
                        if loc_class.country:
                            entity[2]['country'] = loc_class.country

                if entity[0] not in entity_list:
                    entity_list.append(entity[0])
                    if entity[0] not in entities_dedupe_set:
                        entities.append(entity)
                        entities_dedupe_set.add(entity[0])

            item['extract_entities'] = entity_list
            # print(item)

        results = {'trend_extractions': trend_results,
                   'article_sentiment': sentiment,
                   'article_entities': entities,
                   # TODO: replace with real model
                   'climate_relevance_score':0.95}

        print(len(trend_results))
        return results


def run_example():
    # Example json
    ec = EnrichmentCoordinator(True)
    '''
    example_data = json.load(open(get_data('example_input.jsonl'), 'rt', encoding='utf-8', errors='ignore'))

    print('=============	Doc 	=============')
    print('ID:', '\t', example_data['ID'])
    print('Title:', '\t', example_data['title'])
    print('Document:', '\t', example_data['doc'])

    output_data = ec.process(example_data['doc'])
    json.dump(output_data, open(get_data('example_output.jsonl'), 'wt', encoding='utf-8', errors='ignore'))
    '''

    # Example txt

    f = open(get_data('525.txt'), 'rt', encoding='utf-8', errors='ignore')

    text = f.read()

    output_data = ec.process(text)
    json.dump(output_data, open(get_data('525_output.jsonl'), 'wt', encoding='utf-8', errors='ignore'))


    # print('=============	Enriched Doc 	=============')
    # print(json.dumps(output_data))


if __name__ == '__main__':
    run_example()

