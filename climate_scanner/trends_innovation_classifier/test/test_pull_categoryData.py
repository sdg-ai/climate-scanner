import unittest
# from climate_scanner.trends_innovation_classifier.data_utils import pull_categoryData
from data_utils import pull_categoryData

class TestPullCategoryData(unittest.TestCase):

    def test_pull_categoryData(self):
        inputData = [{'id': 1, 
                      'title': 'What Will 3D Printed Fashion Clothes Look Like',
                      'text': 'The idea is simple: Using a lump of raw materials, you print out your new looks on-demand. When you\'re sick of them, you melt them down to create a new batch of clothes. ... On top of reducing material waste, 3D printing can drastically decrease the number of animals killed for materials like leather.',
                      'category': '3D printed apparel',
                      'climate_scanner': True},
                     {'id': 2,
                      'title': 'Artificial Intelligence Explained in Simple Terms',
                      'text': 'Artificial Intelligence (AI) involves using computers to do things that traditionally require human intelligence . This means creating algorithms to classify, analyze, and draw predictions from data. It also involves acting on data, learning from new data, and improving over time.',
                      'category': 'Artificial Intelligence',
                      'climate_scanner': True},
                     {'id': 3, 
                      'title': '3D Printed Clothes: Myth or Reality?',
                      'text': 'Basically, the garments are made by printing very small pieces that, when assembled together, produce a mesh that adapts to the body shape.',
                      'category': '3D printed apparel',
                      'climate_scanner': True},
                     {'id': 4, 
                      'title': 'What is AI technology and how is it used?',
                      'text': 'Artificial intelligence or AI is a popular buzzword you’ve probably heard or read about. Articles about robots, technology, and the digital age may fill your head when you think about the term AI. But what is it really, and how is it used?',
                      'category': 'Artificial Intelligence',
                      'climate_scanner': True}]
        category = 'Artificial Intelligence'
        result = pull_categoryData(inputData,category)
        output = [{'id': 2,
                      'title': 'Artificial Intelligence Explained in Simple Terms',
                      'text': 'Artificial Intelligence (AI) involves using computers to do things that traditionally require human intelligence . This means creating algorithms to classify, analyze, and draw predictions from data. It also involves acting on data, learning from new data, and improving over time.',
                      'category': 'Artificial Intelligence',
                      'climate_scanner': True},
                  {'id': 4, 
                      'title': 'What is AI technology and how is it used?',
                      'text': 'Artificial intelligence or AI is a popular buzzword you’ve probably heard or read about. Articles about robots, technology, and the digital age may fill your head when you think about the term AI. But what is it really, and how is it used?',
                      'category': 'Artificial Intelligence',
                      'climate_scanner': True}]
        self.assertEqual(result, output, "Valid Category Data")


if __name__ == '__main__':
    unittest.main()