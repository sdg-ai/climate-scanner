#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:36:48 2022

@author: cirogam
"""

from neomodel import (config, StructuredNode, StringProperty,
                      Relationship, ZeroOrMore,
                      StructuredRel, AliasProperty)  # work with neo4j

config.DATABASE_URL = "bolt://neo4j:test@localhost:7687"

class BasicRel(StructuredRel):
	rel = StringProperty(required = True)
	name = AliasProperty(to='rel')

class Node(StructuredNode):
	name = StringProperty(required=True, unique_index=False)
	entity = StringProperty(required=True, unique_index=False)
	entity_type = name = StringProperty(required=True, unique_index=True)
	wiki_classes = name = StringProperty(required=True, unique_index=True)
	url = name = StringProperty(required=True, unique_index=True)
	dbpedia_uri = name = StringProperty(required=True, unique_index=True)
	related_to = Relationship('Node', 'RELATED_TO', cardinality=ZeroOrMore, model=BasicRel)



class GraphConstructor:

	def __init__(self, entities):
		self.nodes = []
		self.create_nodes(entities)

	def define_rels(self):
		pass

	def create_nodes(self, entities):
		for ann in entities:
			new_node = Node(
				name = ann[0] if ann[0]!= None else "",
				entity = ann[1] if ann[1]!= None else "",
				entity_type = ann[2]['entityType'] if ann[2]['entityType']!= None else [],
				wiki_classes = ann[2]['wiki_classes'] if ann[2]['wiki_classes']!= None else [],
				url = ann[2]['url'] if ann[2]['url']!= None else "",
				dbpedia_uri = ann[2]['dbPediaIri'] if ann[2]['dbPediaIri']!= None else ""
				   ).save()

			for node in self.nodes:
				new_node.related_to.connect(node, {'rel': 'related'})

			self.nodes.append(new_node)


