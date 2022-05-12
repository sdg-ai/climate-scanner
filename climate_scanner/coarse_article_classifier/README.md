# Guide for running Coarse Article Classifier:

## Necessary dependencies

```
import json
from climate_scanner.coarse_article_classifier.coarse_classifier import Doc2Climate


## Loading function
_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_data(path):
	return os.path.join(_ROOT, 'data', path)
```
## Fetch data and generate climate class
```
## Fetch json file with example data.
input_json = json.load(open(get_data('example_input.json'),'rt', encoding='utf-8', errors='ignore'))

## Generate climate class and check output format
output = Doc2Climate().get_climate_class(input_json)

## The output file is a dictionary containing the estimated climate class for each input.
```