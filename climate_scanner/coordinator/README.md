# enrichment coordinator
A coordinating module to take a cleaned, parsed document and coordinate over the climate scanner machine learning enrichment models.

## Usage instructions
Run the pipeline with the following syntax

```
from climate_scanner.coordinator.coordinator import EnrichmentCoordinator

document = "Some cleaned string document"

ec = EnrichmentCoordinator()

ec.process(doc)
>> 
```
