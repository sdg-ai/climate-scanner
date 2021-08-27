import unittest
from climate_scanner.trends_innovation_classifier.trends_innovation_classifier import *

class MyTestCase(unittest.TestCase):
    def test_coordinator_pipeline(self):
        input_dictionary = {"ID": 1545,"title": "The Shattering Truth of 3D-Printed Clothing","doc": "Three-dimensional printing has changed the way we make everything from prosthetic limbs to aircraft parts and even homes. Now it may be poised to upend the apparel industry as well.","climate_scanner":True,"climate_scanner_prob": 0.6}
        cp = Doc2Trends()
        output = cp.coordinator_pipeline(input_dictionary,0.4)
        print(output)

if __name__ == '__main__':
    unittest.main()