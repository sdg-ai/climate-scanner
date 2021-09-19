# -*- coding: utf-8 -*-

#############################################################################
#
# #	A module to handle requests for entity extraction, via the wikifier
#
#############################################################################

#############################################################################
#
# Import necessary modules
#
#############################################################################

import requests
import os



# Set via environment variables
user_key = 'enter user key here'

headers = {

	'Content-Type': 'application/x-www-form-urlencoded'
}

########################################################
#
# 	 Get extractions from the wikifier API
#
# 	params
# 	------------------------
# 	extraction_string (str)	- input string to extract
# 							  DBPedia entities from within
#   page_rank_sq_threshold (float) 
#		"set this to a real number x to calculate a threshold for pruning the annotations on 
#		the basis of their pagerank score. The Wikifier will compute the sum of squares of all 
#		the annotations (e.g. S), sort the annotations by decreasing order of pagerank, and calculate a 
#		threshold such that keeping the annotations whose pagerank exceeds this threshold would bring 
#		the sum of their pagerank squares to S · x. Thus, a lower x results in a higher threshold and 
#		less annotations. (Default value: −1, which disables this mechanism.) The resulting threshold 
#		is reported in the minPageRank field of the JSON result object. If you want the Wikifier to 
#		actually discard the annotations whose pagerank is < minPageRank instead of including them in 
#		the JSON result object, set the applyPageRankSqThreshold parameter to true (its default value is false)."
#  
#   apply_page_rank_sq_threshold (bool) -
#		If you want the Wikifier to actually discard the annotations whose pagerank is < minPageRank 
#		instead of including them in the JSON result object, set the applyPageRankSqThreshold 
#		parameter to true (its default value is false).
#
#	min_link_freq (int) - 
#		if a link with a particular combination of anchor text and
# 		target occurs in very few Wikipedia pages (less than the value of minLinkFrequency),
# 		this link is completely ignored and the target page is not considered as a candidate
# 		annotation for the phrase that matches the anchor text of this link.
# 		(Default value: 1, which effectively disables this heuristic.)
#	
#	max_mention_entropy (float) -
#		set this to a real number x to cause all highly ambiguous mentions to be ignored 
#		(i.e. they will contribute no candidate annotations into the process). 
#		The heuristic used is to ignore mentions where H(link target | anchor text = this mention) > x. 
#		(Default value: −1, which disables this heuristic.)
#
#	max_targets_per_mention (int) -
#		set this to an integer x to use only the most frequent x candidate annotations 
#		for each mention (default value: 20). Note that some mentions appear as the 
#		anchor text of links to many different pages in the Wikipedia, so disabling this 
#		heuristic (by setting x = −1) can increase the number of candidate annotations 
#		significantly and make the annotation process slower.
#
#	wikiDataClasses (bool) - 
#		determines whether to include, for each annotation, a list if wikidata 
#		(concept ID, concept name) pairs for all classes to which concept belongs directly or indirectly.
#
#   
#######################################################




def get_entities(extraction_string, 
				page_rank_sq_threshold,                # a lower x results in a higher threshold and less annotations default(-1)
				apply_page_rank_sq_threshold,          # to actually discard the annotations whose pagerank is < minPageRank (defaultis False)
				min_link_freq,                         # default 1 
				max_mention_entropy,                   # cause all highly ambiguous mentions to be ignored float (default -1)
				max_targets_per_mention,               # to use only the most frequent x candidate annotations for each mention (default x=20)
				wiki_data_classes):                    # wikiDataClasses (True or False)

				
	

	#
	api_url = 'http://www.wikifier.org/annotate-article?'

	payload = {'userKey': user_key, 'text': extraction_string, 'lang': 'auto',
			   'minLinkFrequency': min_link_freq,
			   'applyPageRankSqThreshold': apply_page_rank_sq_threshold,
			   'includeCosines':False,
			   'maxMentionEntropy':max_mention_entropy,
			   'maxTargetsPerMention':max_targets_per_mention,
			   'pageRankSqThreshold':page_rank_sq_threshold,
			   'wikiDataClasses': wiki_data_classes}

	r = requests.post(api_url, headers=headers, data=payload)
	if r.status_code == 200:
		return r.json()

	else:
		return r.status_code, None


#############################################################################
#
# Class to handle entity extraction, through wikifier
#
#############################################################################
				

class EntityExtractor:
	# Wrapper function, extracting the 'annotations' field of the return json
	def get_annotations(self,
						extraction_string, 
						page_rank_sq_threshold       = "-1",       # (float) a lower x results in a higher threshold and less annotations default(-1)
						apply_page_rank_sq_threshold = "false",    # (bool) to actually discard the annotations whose pagerank is < minPageRank (defaultis False)
						min_link_freq                = "1",        # (int) default 1 
						max_mention_entropy          = "-1",       # (float) cause all highly ambiguous mentions to be ignored float (default -1)
						max_targets_per_mention      = "20",       # (int) to use only the most frequent x candidate annotations for each mention (default x=20)
						wiki_data_classes            = "false"):   # (int) to use only the most frequent x candidate annotations for each mention (default x=20)

		annotations = get_entities(extraction_string             = extraction_string,
								   	page_rank_sq_threshold       = page_rank_sq_threshold,
									apply_page_rank_sq_threshold = apply_page_rank_sq_threshold,
									min_link_freq                = min_link_freq,
									max_mention_entropy          = max_mention_entropy,
									max_targets_per_mention      = max_targets_per_mention,
									wiki_data_classes            = wiki_data_classes).get('annotations', None)
		return annotations


def test_code():
	x = EntityExtractor()

	example_string = 'Sir Keir Starmer has urged the government to invest urgently in jobs that benefit the environment. ' \
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

	annotations = x.get_annotations(example_string)

	for i, annotation in enumerate(annotations):
		print('============== {} =============='.format(i))
		for item in annotation:
			print(item, '\t', annotation[item])
		print('\n')


if __name__ == '__main__':
	test_code()