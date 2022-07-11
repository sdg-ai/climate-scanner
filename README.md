# climate-scanner
An open source library for machine learning based analysis of climate trends and innovations. Authored by the AI for Good Foundation.

The library aims to provide a solution to keeping a pulse on the masses of data which is published daily in the media,
when considering innovations and trends which relate to climate change. The library includes the following modelling:
- Predicting "Climate Change Related" vs "Other" articles
- Identifying and locating mentions of particular "trend-topics" or "climate innovations"
- Analysing sentiment of extractions
- Identifying entities in extractions
- Building networks around entities in media mentions

### Library usage pre-requisites 
1. Python 3.*

## Installation/Configuration

```
git clone https://github.com/sdg-ai/climate-scanner.gt
pip3 install -r requirements.txt
python3 setup.py install

git lfs install
git lfs pull
```

## Usage Instructions
### Enrichment pipeline

```
from climate_scanner.coordinator.coordinator import EnrichmentCoordinator

ec = EnrichmentCoordinator()

input_json = {"ID": <id-string>, "title": <title-string>, "doc": <document string>}

output = ec.process(input_json)
```

For example:


## Testing (for collaborators)
The following runs the library unit tests. Please ensure changes you may have added do not cause issues in
other parts of the code by running:
```
pytest
```
