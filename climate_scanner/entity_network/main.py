#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:04:54 2022

@author: cirogam
"""

from entity_extraction import test_code
#from neo4j_model import GraphConstructor


def upload_sample_data():
	entities = test_code()
	print(entities)
	#graph = GraphConstructor(entities)

def sample_queries():
	pass

if __name__ == '__main__':
	upload_sample_data()
