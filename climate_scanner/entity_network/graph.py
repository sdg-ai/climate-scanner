# -*- coding: utf-8 -*-

import os
import pickle
import networkx as nx
from pyvis.network import Network


#############################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
#############################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
	return os.path.join(_ROOT, 'data', path)


class Node:
	# Class defining the Node
	# Methods:
	#     set             -> set method to update node's attributes
	#	  get             -> get method to get avoid direct access to the attributes
	# Attributes:
	#	named_entity      -> str representing the named entity picked up by NER system (spacy + wikifier)
	#  	entity_type_spacy -> str representing spacy entity classification eg: PERSON, GPE, ORG

	def __init__(self,named_entity,entity_type_spacy):
		self._named_entity = named_entity
		self._entity_type_spacy = entity_type_spacy

	def __hash__(self):
		return hash((self._named_entity,self._entity_type_spacy))

	def __eq__(self,other):
		return (self._named_entity,self._entity_type_spacy) == (other._named_entity,other._entity_type_spacy)

	# getter methods 
	def get_named_entity(self):
		return self._named_entity
	def get_entity_type_spacy(self):
		return self._entity_type_spacy
	def get_entity_type_wiki(self):
		return self._entity_type_wiki

	# setters for the attributes
	def set_named_entity(self,name):
		self._named_entity = name
	def set_entity_type_spacy(self,x):
		self._entity_type_spacy = x




class EntityGraph:
	# Class defining the Entity Graph
	# Methods:
	#     update         -> takes annotation list from entity extraction module and updates the graph
	#     get_neibors    -> takes a Node class as input and returns all the neibohrs, -1 if node is not in graph
	#     get_meta_data  -> takes a Node class as input and returns the meta data of the node. -1 if node not in graph
	#     get_edge_list  -> returns the edge list based on the named_entity
	# 	  delete_node    -> removes a specific node from the graph using a DFS traversal. return True if node deleted False if node not in graph
	#     
	#     query          -> filters the graph based ona queury such as specific GPE locations, etc...

	# Attributes:
	#	metaData        -> is a dictionary containing graph nodes as keys and their respective meta data (described bolw) as values
			#   entity_type_wiki  -> list() of wiki type entities eg: ['PopulatedPlace','Place','Country','AdministrativeRegion','Region','Territory']
			#	wiki_classes      -> list() of wiki_classes eg: ['constituent part of the United Kingdom','cultural area','nation']
			# 	wiki_url          -> str representing wikipedia url to the page with closest similarity to named_entity
			#   dbPediaIri        -> str representing dbpedia url to the page with closest similarity to named_entity
			#	count             -> int representing the number of times a specific named entity was mentioned so far
			#   inDeg             -> int representing the indegree node
			#   outDeg            -> int representing the outDegree node

	#   adjacencyMap   -> is a dictionary of dictionaries connecting each node to its neibors. This adjacencyMap also provide a measure of srength relation between nodes. format: 
	# 					  adjencencyMap[node_a] : {node_a: {node_b:5,node_c:1,node_w:15}}

	def __init__(self):
		self._metaData ={}
		self._adjacencyMap={}
	


	def update(self,ner_list):
		# aupdate :  takes annotation list from entity extraction module and updates the graph
		#	inputs:
		#		ner_list -> list() of detected entities by the NER system(spacy+wikifier) [named_entity_name,entity_type_spacy,metaData]
		# 	outputs:
		#		Bool     -> True when graph is updated False otherwise


		# use exceptions!!!!!
		if len(ner_list)==0:
			return False

		for idx in range(len(ner_list)):
			ner_list[0],ner_list[idx] = ner_list[idx],ner_list[0]

			# extract data from ner_node
			named_entity,entity_type_spacy,metaData =  ner_list[0]

			# create a node:
			node = Node(named_entity=named_entity,entity_type_spacy=entity_type_spacy)

			# check if node is in adjMap if not create an entry:
			if node not in self._adjacencyMap:
				self._adjacencyMap[node] = {}

				# addd count, inDeg and outDeg to the meta Data for later use
				metaData['count']  = 0
				metaData['inDeg']  = 0
				metaData['outDeg'] = 0
				self._metaData[node] = metaData 
			else:
				#update just the count in the meta data
				self._metaData[node]['count']+=1

			# loop throught the neighbors and collect their data
			for idx2 in range(1,len(ner_list)):

				#extract data from ner list entry
				named_entity,entity_type_spacy,metaData = ner_list[idx2]

				# create a nei node
				nei_node = Node(named_entity=named_entity,entity_type_spacy=entity_type_spacy)

				# make sure the neibohr node is not the same as the parent node otherwise just increase count
				if nei_node==node:
					continue

				# check if nei node in adjMap if not add it
				if nei_node not in self._adjacencyMap:
					self._adjacencyMap[nei_node]={}

					# addd count, inDeg and outDeg to the meta Data for later use
					metaData['count']  = 0
					metaData['inDeg']  = 0
					metaData['outDeg'] = 0
					self._metaData[nei_node]=metaData

				# creating the connections in the adjacencyMap
				self._adjacencyMap[node][nei_node]=1
				self._adjacencyMap[nei_node][node]=1

				# update the metaData for each node specifically the indeg and outdeg
				self._metaData[node]['outDeg']+=1
				self._metaData[nei_node]['inDeg']+=1

			ner_list[0],ner_list[idx] = ner_list[idx],ner_list[0]
		return True

	
	def get_edge_list(self):
		# Method to get the edge list based on named_entity_feature
		# inputs:
		#	self
		# outputs:
		#	list representing the edge list of the graph
		edgeList = []
		seen = set()
		for node in self._adjacencyMap:
			source = node.get_named_entity()
			for nei_node in self._adjacencyMap[node]:
				target = nei_node.get_named_entity()
				if (source,target) not in seen or (target,source) not in seen:
					edgeList.append([source,target])
					seen.add((source,target))
		return edgeList
	
	def get_adj_map(self):
		return self._adjacencyMap

	def get_meta_data(self):
		return self._metaData
	





class GraphConstructor:
	# Class for faccilitating the construction of an entity graph
	# Methods:
	#     method1:...
	# Attributes:
	def __init__(self):
		pass




def test_code(): 
	from entity_extraction import EntityExtractor
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

	annotations1 = x.get_annotations(example_string_1)
	annotations2 = x.get_annotations(example_string_2)

	# creating graph object
	graph = EntityGraph()

	# updating graph with annotations1 and annotation2
	graph.update(annotations1)
	graph.update(annotations2)
	adjList = graph.get_adj_map()
	edgeList = graph.get_edge_list()

	net = Network(height='500px', width='1000px', directed=False, notebook=False)



	G = nx.Graph()
	G.add_edges_from(edgeList)
	net.from_nx(G)
	
	net.repulsion(node_distance=500)
	net.show('first_graph.html')

	print('Edge List')

	for elem in edgeList:
		source,target = elem
		print(source,'------->',target)

	
if __name__ == '__main__':
	test_code()


