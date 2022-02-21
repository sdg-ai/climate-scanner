# -*- coding: utf-8 -*-

import requests
import os
import spacy 
nlp = spacy.load('en_core_web_lg')

# Set via environment variables
os.getenv('user_key')
user_key = 'beejatbmicwzpxudjxxslszpidsjep'
headers = {

    'Content-Type': 'application/x-www-form-urlencoded'
}

########################################################
#
#      Get extractions from the wikifier API

#     params
#     ------------------------
#     extraction_string (str)- input string to extract
#                               DBPedia entities from within
#   page_rank_sq_threshold (float) 
#         "set this to a real number x to calculate a threshold for pruning the annotations on 
#         the basis of their pagerank score. The Wikifier will compute the sum of squares of all 
#         the annotations (e.g. S), sort the annotations by decreasing order of pagerank, and calculate a 
#         threshold such that keeping the annotations whose pagerank exceeds this threshold would bring 
#         the sum of their pagerank squares to S · x. Thus, a lower x results in a higher threshold and 
#         less annotations. (Default value: −1, which disables this mechanism.) The resulting threshold 
#         is reported in the minPageRank field of the JSON result object. If you want the Wikifier to 
#         actually discard the annotations whose pagerank is < minPageRank instead of including them in 
#         the JSON result object, set the applyPageRankSqThreshold parameter to true (its default value is false)."

#   apply_page_rank_sq_threshold (bool) -
#         If you want the Wikifier to actually discard the annotations whose pagerank is < minPageRank 
#         instead of including them in the JSON result object, set the applyPageRankSqThreshold 
#         parameter to true (its default value is false).

#     min_link_freq (int) - 
#         if a link with a particular combination of anchor text and
#         target occurs in very few Wikipedia pages (less than the value of minLinkFrequency),
#         this link is completely ignored and the target page is not considered as a candidate
#         annotation for the phrase that matches the anchor text of this link.
#         (Default value: 1, which effectively disables this heuristic.)

#     max_mention_entropy (float) -
#         set this to a real number x to cause all highly ambiguous mentions to be ignored 
#         (i.e. they will contribute no candidate annotations into the process). 
#         The heuristic used is to ignore mentions where H(link target | anchor text = this mention) > x. 
#         (Default value: −1, which disables this heuristic.)

#     max_targets_per_mention (int) -
#         set this to an integer x to use only the most frequent x candidate annotations 
#         for each mention (default value: 20). Note that some mentions appear as the 
#         anchor text of links to many different pages in the Wikipedia, so disabling this 
#         heuristic (by setting x = −1) can increase the number of candidate annotations 
#         significantly and make the annotation process slower.

#     wikiDataClasses (bool) - 
#         determines whether to include, for each annotation, a list if wikidata 
#         (concept ID, concept name) pairs for all classes to which concept belongs directly or indirectly.


# ######################################################




def get_entities(extraction_string, 
                page_rank_sq_threshold,                # a lower x results in a higher threshold and less annotations default(-1)
                apply_page_rank_sq_threshold,          # to actually discard the annotations whose pagerank is < minPageRank (defaultis False)
                min_link_freq,                         # default 1 
                max_mention_entropy,                   # cause all highly ambiguous mentions to be ignored float (default -1)
                max_targets_per_mention,               # to use only the most frequent x candidate annotations for each mention (default x=20)
                wiki_data_classes,                     # wikiDataClasses (true or false)
                wiki_data_class_ids):                  # wikiDataClassIds (true or false)



    api_url = 'http://www.wikifier.org/annotate-article?'

    payload = {'userKey': user_key, 'text': extraction_string, 'lang': 'auto',
               'minLinkFrequency': min_link_freq,
               'applyPageRankSqThreshold': apply_page_rank_sq_threshold,
               'includeCosines':'false',
               'maxMentionEntropy':max_mention_entropy,
               'maxTargetsPerMention':max_targets_per_mention,
               'pageRankSqThreshold':page_rank_sq_threshold,
               'wikiDataClasses': wiki_data_classes,
               'wikiDataClassIds':wiki_data_class_ids}

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
    def get_annotations_wiki(self,
                        extraction_string, 
                        page_rank_sq_threshold       = "-1",       # (float) a lower x results in a higher threshold and less annotations default(-1)
                        apply_page_rank_sq_threshold = "false",    # (bool) to actually discard the annotations whose pagerank is < minPageRank (defaultis False)
                        min_link_freq                = "1",        # (int) default 1 
                        max_mention_entropy          = "-1",       # (float) cause all highly ambiguous mentions to be ignored float (default -1)
                        max_targets_per_mention      = "20",       # (int) to use only the most frequent x candidate annotations for each mention (default x=20)
                        wiki_data_classes            = "true",
                        wiki_data_class_ids          = "false"):   

        annotations = get_entities(extraction_string             = extraction_string,
                                    page_rank_sq_threshold       = page_rank_sq_threshold,
                                    apply_page_rank_sq_threshold = apply_page_rank_sq_threshold,
                                    min_link_freq                = min_link_freq,
                                    max_mention_entropy          = max_mention_entropy,
                                    max_targets_per_mention      = max_targets_per_mention,
                                    wiki_data_classes            = wiki_data_classes,
                                    wiki_data_class_ids          = wiki_data_class_ids).get('annotations', None)
#extracting annotations
        res={}
        if wiki_data_classes=='true':    
            for idx,elem in enumerate(annotations):
                if 'wikiDataClasses' in elem.keys():
                    wikiClassesLength = len(elem['wikiDataClasses']) if len(elem['wikiDataClasses'])<5 else 5
                    classesList       = [elem['wikiDataClasses'][i]['enLabel'] for i in range(wikiClassesLength)] 
                else:
                    classesList       = []                    
                chFrom            = elem['support'][0]['chFrom']
                chTo              = elem['support'][0]['chTo']+1
                word              = extraction_string[chFrom:chTo]
                entities          = elem['dbPediaTypes']
                url               = elem['url']
                dbPediaIri        = elem['dbPediaIri']
                tup               = {'entityType': entities,'wiki_classes':classesList,'url':url,'dbPediaIri':dbPediaIri}
                if len(entities):
                    res[word]=tup
        else:
            for idx,elem in enumerate(annotations):          
                chFrom            = elem['support'][0]['chFrom']
                chTo              = elem['support'][0]['chTo']+1
                word              = extraction_string[chFrom:chTo]
                entities          = elem['dbPediaTypes']
                url               = elem['url']
                dbPediaIri        = elem['dbPediaIri']
                tup               = {'entityType':entities,'wiki_classes':None,'url':url,'dbPediaIri':dbPediaIri} 
                if len(entities):
                    res[word]= tup
        return annotations,res

    def get_annotations_spacy(self,doc):
        """
        Method to extract annotations using spacy
        
		Args:
            doc (spacy)
            A Doc is a sequence of Token objects. Access sentences and named entities,
            export annotations to numpy arrays, losslessly serialize to compressed binary strings. 
        
		Returns:
            List() of lists [entity.text,entity.label_]
        """
        
        res = []
        if doc.ents: 
            for ent in doc.ents: 
                if ent.label_ == "PERSON" or ent.label_=="ORG" or ent.label_== "GPE":
                    res.append([ent.text,ent.label_])
        return res
    
    def get_annotations(self,doc):
        """
        Method to extract annotations using spaCy and wikifier. It uses the data the following methods:
        # get_annotations_spacy()
        # get_annonations_wiki()
        
		Args:
            doc - (str)
            an str object representing given text
        
		Returns:
            entities_list - list([])
                entities_list[0]: (str) representing the annotated word from text
                entities_list[1]: (str) repesenting spaCy entity classification
                entities_list[3]: dict() representing wiki return object featuring: entityType,wiki_classes,url,dbPediaIri
            
        """
        
        txt = nlp(doc) 
        entities_list = self.get_annotations_spacy(txt)
        temp_txt = " ".join([ent[0] for ent in entities_list])
        _,wiki_res = self.get_annotations_wiki(extraction_string=temp_txt)
        for ent in entities_list:
            if ent[0] in wiki_res:
                ent.append(wiki_res[ent[0]])
            else:
                ent.append({'entityType':None,'wiki_classes':None,'url':None,'dbPediaIri':None})
        return entities_list


def test_code():
	x = EntityExtractor()

	example_string_1 = 'Sir Keir Starmer has urged the government to invest urgently in jobs that benefit the environment. ' \
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

	example_string_2 = "By Scott Simone, Contributor On May 2, Chris Ballinger stood on stage of the Future Blockchain Summit in Dubai and made a striking announcement: " \
					   "He’d formed a global consortium of automakers to create an ecosystem that would allow blockchain technology to thrive in the transportation industry." \
					   "The nonprofit organization, mobility open blockchain initiative (mobi), consists of auto giants such as BMW, General Motors, Ford, and Renault, along with " \
					   "companies in the tech, financial, and insurance industries.  It’s all about building trust in a business network between otherwise untrusting parties, "\
					   "Ballinger said.Industry executives have expressed excitement at the prospect of the new technology. "\
					   "We believe blockchain will transform the way people and businesses interact, creating new opportunities in mobility,  Rich Strader, "\
					   "vice president of mobility product solutions at Ford, explained.The announcement, which has been a year in the making, comes on the heels " \
					   "of at least half a dozen auto industry titans announcing forays into the world of blockchain technology. And while most of these projects are in their infancy, "\
					   "Ballinger and his coalition are operating around one ideal: If applied correctly, blockchain can revolutionize the auto industry, "\
					   "and it’s very likely that the industry will look tremendously different a decade from today.Ballinger started toying with blockchain technology when he was head of innovation "\
					   "and new mobility at the Toyota Research Institute.He began seeing movement toward on-demand mobility whereby the infrastructure was pay-per-use, "\
					   "rather than outright purchasing new cars.  I began thinking that in that world, blockchain [and] distributed ledger was the perfect fit because these cars’ "\
					   "infrastructures need to communicate,  Ballinger explained.  They need ways of making machine payments, paying for use, paying for congestion, paying for tolls, " \
					   "paying for energy as they go, and having secure identities for both the car and the riders. Under his tutelage, the Toyota Research Institute presented proofs of "\
					   "concept that utilized blockchain technology to address these concerns. Yet while they were successful in theory, the idea failed to take off. "\
					   "I began noticing that that wasn’t unusual at all,  Ballinger said.  I kept seeing big companies stick an asset on the blockchain, make a public announcement, " \
					   "then have it go nowhere. The reason, according to Ballinger, is that there was no ecosystem that would help those ideas thrive.  If there’s nothing else in the network, "\
					   "the technology alone is useless, Ballinger said.  It’s like email or fax back in the day; if there’s no one to send an email to, or a fax to, your fax machine is just a good doorstop. "\
					   "And so building the ecosystem was the quickest step that was needed. According to Ballinger,  Blockchain and related trust-enhancing technologies are poised to redefine the automotive "\
					   "industry and how consumers purchase, insure, and use vehicles. By bringing together automakers, suppliers, startups, and government agencies, we can accelerate "\
					   "adoption for the benefit of businesses, consumers, and communities.Blockchain and related trust-enhancing technologies are poised to redefine the automotive industry and how "\
					   "consumers purchase, insure, and use vehicles. By bringing together automakers, suppliers, startups, and government agencies, we can accelerate adoption for the benefit of businesses, "\
					   "consumers, and communities. Chris Ballinger, CEO and Founder at MOBI: Mobility Open Blockchain InitiativeIn the first few months of 2018, it seemed like there was a new "\
					   "announcement every day about an auto giant dabbling with blockchain technology.In late March, for example, Ford received a patent to use cryptocurrency and blockchain technology "\
					   "so that cars could communicate with—and, at times, pay—each other while on the road. The revolutionary idea was laid out in the patent application as follows:This system "\
					   "would temporarily allow for … vehicles to drive at higher speeds in less-occupied lanes of traffic and also to merge and pass freely when needed. Other … vehicles voluntarily " \
					   "occupy slower lanes of traffic to facilitate the consumer vehicle to merge into their lanes and pass as needed.But the patent, according to Karen Hampton, a spokesperson " \
					   "for Ford Motor Company, isn’t necessarily indicative of a current product plan. Rather, it represents the company’s willingness to explore new ideas.  "\
					   "Patent applications are intended to protect new ideas, but they aren’t necessarily an indication of new business or product plans,  said Hampton.A few days later, " \
					   "Audi announced that it, too, was testing blockchain technology for its physical and financial distribution processes, aiming to increase security and transparency in its global " \
					   "supply chains. Porsche is exploring the utilization of blockchain apps in its vehicles in cooperation with the Berlin-based startup XAIN. And BMW is reportedly planning " \
					   "to expand its portfolio by partnering with UK startup Circulor to eliminate battery minerals produced by child labor.Perhaps furthest along in developing an automotive-based use " \
					   "for blockchain is Mercedes Benz, which is currently testing its app MobiCoin to reward drivers for environmentally-cautious driving. MobiCoin uses blockchain technology to pair " \
					   "a smartphone app to technology already available in Mercedes Benz vehicles that measures how  green  the car is driving. Depending on how green the driver operates a vehicle, drivers "\
					   "will earn  coins  through the app that can be redeemed for rewards, such as VIP tickets for high-profile events like the MercedesCup final. MobiCoin " \
					   "technology is an experiment to find out how technology can influence behavior,  Jonas von Malottki, senior manager finance and controlling solutions at Daimler, " \
					   "which owns Mercedes, said.  The idea was that we wanted to create something that is car-centric and incentivizes green driving. And while von Malottki had a lot of people "\
					   "explaining to him the different ways the company should use blockchain, it became clear they weren’t the right forms.  For [MobiCoin], we have a coin behind it, " \
					   "so it was good to use blockchain,  he explained.  It’s the type of technology that allows us to make it very people-centric—blockchain makes it a lot easier and a lot more trustworthy. "\
					   "Like the other companies, Mercedes is just in the testing phase; after the two-month test is complete, it will evaluate whether or not it will make MobiCoin available worldwide. " \
					   "Experts Say: They’ll Be Sorry Auto industry insiders and blockchain technology experts agree that if automakers don’t innovate and make use of blockchain now, well, they’ll be sorry. " \
					   "The times are changing to more autonomous, self-driving, electric [cars],  von Malottki described.  We’re adapting to this generation, and we have to change our business model more " \
					   "and more to do so. I think blockchain is one of the technologies that is very much augmented for this type of change. According to Teodoro Lio, managing director and industrial and " \
					   "automotive innovation lead at Accenture, blockchain will drastically affect the auto industry—from the handling of vehicle ID numbers and collision histories to the complex supply chains " \
					   "that lead to assembly lines and dealerships.  Industry leaders are just beginning to understand the unique characteristics of this technology and its diverse forms,  he said. "\
					   "For Derin Cag, founder of Richtopia and co-founder of Marketing Runners and Blockchain Age, companies are jumping onto the blockchain bandwagon both because of the tremendous opportunity " \
					   "and also what will happen if they don’t. If they don’t do it, the competition is going to, and destroy a big segment of their business,  Cag articulated.  The group that does it right the "\
					   "first time is really going to transform the industry as it exists today. In five to 10 years, the industry will not be the same, because of all the transformations. "

	annotations = x.get_annotations(example_string_2)

	for i, annotation in enumerate(annotations):
		print('============== {} =============='.format(i))
		print(annotation[0],'\n',annotation[1],'\n entityType:',annotation[2]['entityType'], '\n wiki_classes:',annotation[2]['wiki_classes'],'\n url:',annotation[2]['url'], '\n dbPediaIri:',annotation[2]['dbPediaIri'])
		

		print('\n')


if __name__ == '__main__':
	test_code()