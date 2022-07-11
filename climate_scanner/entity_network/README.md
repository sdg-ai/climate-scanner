# climate-scanner
A module to use extracted entities from the climate scanner ML upstream to construct entity graphs.


## Usage instructions
To run the top level functionality, run the code in the following way
```buildoutcfg
from climate_scanner.entity_network import ...
```

## Implementation details
=======
# Entitiy extraction
This is a submodule of climate scanner which extracts distinct entities from the input document, by using the third party module Wikifier. 
See documentation for more info: https://wikifier.org/info.html

## Import functions
```
from climate_scanner.entity_network.entity_extraction import EntityExtractor
```
## Extract entities
```
x = EntityExtractor()
text = 'Sir Keir Starmer has urged the government to invest urgently in jobs that benefit the environment. ' \
					 'The Labour leader wants £30bn spent to support up to 400,000 “green” jobs in manufacturing' \
					 ' and low-carbon industries.' \
					 'The government says it will create thousands of green jobs as part of its overall climate strategy.' \
					 'But official statistics show no measurable increase in environment-based jobs in recent years.' \
					 'Speaking to the BBC as he begins a two-day visit to Scotland, Sir Keir blamed this on a' \
					 ' "chasm between soundbites and action”.' \
					 'He and PM Boris Johnson are both in Scotland this week, showcasing their green credentials' \
					 ' ahead of Novembers COP 26 climate summit in Glasgow.' \
					 'Criticising the governments green jobs record, Sir Keir points to its decision to' \
					 ' scrap support for solar power and onshore wind energy, and a scheme to help' \
					 ' householders in England insulate their homes.' \
					 'He said: “It’s the government’s failure to match its rhetoric with reality that’s ' \
					 'led to this, They have used soundbites with no substance.' \
					 '“They have quietly been unpicking and dropping critical' \
					 ' commitments when it comes to the climate crisis and the future economy.' \
					 '“It’s particularly concerning when it comes to COP 26.' \
					 '"Leading by example is needed, but just when we need leadership' \
					 ' from the prime minister on the global stage, here in the UK we have a prime' \
					 ' minister who, frankly, is missing in action."'

annotations = x.get_annotations(text)
```
## Printing the annotations
```
	for i, annotation in enumerate(annotations):
		print('============== {} =============='.format(i))
		print(annotation[0],'\n',annotation[1],'\n entityType:',annotation[2]['entityType'], '\n wiki_classes:',annotation[2]['wiki_classes'],'\n url:',annotation[2]['url'], '\n dbPediaIri:',annotation[2]['dbPediaIri'])
```
