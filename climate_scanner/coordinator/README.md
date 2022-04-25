# enrichment coordinator
A coordinating module to take a cleaned, parsed document and coordinate over the climate scanner machine learning enrichment models.

## Usage instructions
Run the pipeline with the following syntax

### Import and define document
```
# Import packages
import json
from climate_scanner.coordinator.coordinator import EnrichmentCoordinator

# Define your own document
document = "Some cleaned string document"
```
### Process data and print
```
# Process data with Enrichement Coordinator
ec = EnrichmentCoordinator()
output_data = ec.process(example_data)

# Print output
print('=========	Enriched Doc 	=========')
print(json.dumps(output_data))
```
