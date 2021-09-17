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
user_key = ""

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
# 	min_page_rank (float) -
# 		"set this to a real number x to calculate a threshold for pruning
# 		the annotations on the basis of their pagerank score. The Wikifier
# 		will compute the sum of squares of all the annotations (e.g. S),
# 		sort the annotations by decreasing order of pagerank, and calculate
# 		a threshold such that keeping the annotations whose pagerank exceeds
# 		this threshold would bring the sum of their pagerank squares to S · x.
# 		Thus, a lower x results in a higher threshold and less annotations.
# 		(Default value: −1, which disables this mechanism.)
# 		The resulting threshold is reported in the minPageRank field of the JSON result object.
# 		If you want the Wikifier to actually discard the annotations whose pagerank is
# 		< minPageRank instead of including them in the JSON result object,
# 		set the applyPageRankSqThreshold parameter to true (its default value is false).
#
# 	min_link_freq (int) - if a link with a particular combination of anchor text and
# 		target occurs in very few Wikipedia pages (less than the value of minLinkFrequency),
# 		this link is completely ignored and the target page is not considered as a candidate
# 		annotation for the phrase that matches the anchor text of this link.
# 		(Default value: 1, which effectively disables this heuristic.)
#
#######################################################


import urllib.parse, urllib.request, json

def CallWikifier(text, lang="en", threshold=0.8):
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "lrccenpxcbtbmpvabyjudwvxehjcpf"),
        ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "false"), ("maxMentionEntropy", "3")
        ])
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout = 60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))

	# Output the annotations.
    for annotation in response["annotations"]:
        print("%s (%s)" % (annotation["title"], annotation["url"]))
	
	




def get_entities(extraction_string, min_page_rank=-1, min_link_freq=1):
	if min_page_rank != -1:
		apply_pagerank_threshold = True

	else:
		apply_pagerank_threshold = False

	#
	api_url = 'http://www.wikifier.org/annotate-article?'

	payload = {'userKey': user_key, 'text': extraction_string, 'lang': 'auto',
			   'minLinkFrequency': min_link_freq, 'min_page_rank': min_page_rank,
			   'applyPageRankSqThreshold': apply_pagerank_threshold}

	r = requests.post(api_url, headers=headers, data=payload)
	print(r.json())
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
	def get_annotations(self, extraction_string, min_page_rank=-1, min_link_freq=1):
		annotations = get_entities(extraction_string, min_page_rank, min_link_freq).get('annotations', None)
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