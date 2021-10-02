import unittest
from climate_scanner.trends_innovation_classifier.data_utils import doc_to_multisentence

class TestDocToMultisentence(unittest.TestCase):
    
    def test_doc_to_multisentence(self):
        text = "What is climate change? What could happen? Get all the facts you need to know. Weather changes day to day—sometimes it rains, other days it’s hot. Climate is the pattern of the weather conditions over a long period of time for a large area. And climate can be affected by Earth’s atmosphere."
        num_sentences = 3
        result = doc_to_multisentence(text,num_sentences)
        output =  [{'string_indices': [0, 24, 43, 79], 
                    'text': 'What is climate change? What could happen? Get all the facts you need to know.'
                   }, {'string_indices': [80, 147, 241, 291], 
                       'text': 'Weather changes day to day—sometimes it rains, other days it’s hot. Climate is the pattern of the weather conditions over a long period of time for a large area. And climate can be affected by Earth’s atmosphere.'
                      }]
        self.assertEqual(result, output, 'Valid Indices')


if __name__ == '__main__':
    unittest.main()