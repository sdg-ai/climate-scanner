# Trends and Innovation Classifier
This is a submodule of climate scanner which focuses on trends and innovation classifier.

## Usage instructions
To fetch all classes and functions, first run this.
```
from climate_scanner.coarse_article_classifier.coarse_classifier import Doc2Climate
from climate_scanner.trends_innovation_classifier import Doc2Trends

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
output_1 = Doc2Climate().get_climate_class(input_json)

## Run Trends and Innvoation Classifier
# Second input is a probability threshold (from 0 to 1) to filter out entries 
output = Doc2Trends().coordinator_pipeline(output_1, 0.5)

## The output file is a dictionary containing the classified trends and innovations.
 
```

## Implementation details
