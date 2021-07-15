import unittest
from data_utils import doc_to_sentence


class MyTestCase(unittest.TestCase):
    def test_doc_to_sentence(self):
        text = "Climate change is a long-term shift in global or regional climate patterns. Often climate change refers specifically to the rise in global temperatures from the mid-20th century to present."
        sentences = doc_to_sentence(text)
        output = [{'string_indices': (0, 75),
                   'text': 'Climate change is a long-term shift in global or regional climate patterns.'
                   }, {'string_indices': (76, 189),
                       'text': 'Often climate change refers specifically to the rise in global temperatures from the mid-20th century to present.'
                       }]
        self.assertEqual(sentences, output, 'Valid Indices')


if __name__ == '__main__':
    unittest.main()
