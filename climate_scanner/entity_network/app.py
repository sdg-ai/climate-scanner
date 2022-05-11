#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from functools import wraps

# Necessary API imports
from flask import Flask, request
from functools import wraps
from flask_restplus import Api, Resource, fields


# Graph modules
from neo4j_model import GraphConstructor
graph = GraphConstructor()

# Initialise application
app = Flask(__name__)

# Defining an authorisation Dict
authorizations = {
	'apikey': {
		'type': 'apiKey',
		'in': 'header',
		'name': 'Authorization'
	},
}

api = Api(app=app, authorizations=authorizations)

# Load in key from environment vars typically
# SWAGGER_KEY = os.environ.get('SWAGGER_KEY')
SWAGGER_KEY = '1234'


##############################################################################
#
#	A function wrapper for the end-point methods to provide some rudimentary
#	security.
#
#	TODO: host key for security, rather than keeping in codebase
#
##############################################################################

def key_required(func):
	@wraps(func)
	def decorated(*args, **kwargs):
		end_key = None

		if 'Authorization' in request.headers:
			end_key = request.headers['Authorization']

		if not end_key:
			return {'status': 401, 'message': 'Api Key required'}

		if end_key != SWAGGER_KEY:
			return {'status': 401, 'message': 'Api Key Incorrect'}

		print('API Key: {}'.format(end_key))
		return func(*args, **kwargs)

	return decorated


# Define a namespace. These are used to divide up different modules of the api
# The api.namespace() function creates a new namespace with a URL prefix.
# The description field will be used in the Swagger UI to describe this set of methods.
name_space = api.namespace('TrendScanner Graph API',
							description='An api for testing the TrendScanner Graph Model.')

##########################################################################################
#
#	Define class for the route and inherit from restplus Resource.
# 	RESTful resources are used to organize different the API endpoints
# 	This example shows only one endpoint but if you add more they are all displayed/
# 	documented on the same UI page
#
#
##########################################################################################

# Model of expected input data for the end point we are about to specify.... here a casing endpoint
casing_data = api.model('Casing Model', {
	'strings': fields.List(required=True, description="List of strings to transform",
						  example=["Java Engineer", "RandomSTRIng"], cls_or_instance=fields.String),
	'mode': fields.String(
		required=False, description="Industry of the company", example='upper')
})

graph_data = api.model('Graph Model', {
	'entities': fields.List(required=True,
					  description="List of incoming entities",
					  example=[
						  ['Chris Ballinger', 'PERSON',
							 {'entityType': None, 'wiki_classes': None,
							 'url': None, 'dbPediaIri': None}],
						  ['the Future Blockchain Summit', 'ORG',
							 {'entityType': None, 'wiki_classes': None,
							 'url': None, 'dbPediaIri': None}],
						  ['Ford', 'ORG', {'entityType': ['Agent', 'Organisation', 'Company'], 'wiki_classes': ['automobile manufacturer', 'sponsor', 'car brand', 'manufacturing company', 'legal person'], 'url': 'http://en.wikipedia.org/wiki/Ford_Motor_Company', 'dbPediaIri': 'http://dbpedia.org/resource/Ford_Motor_Company'}]],
					  cls_or_instance=fields.Arbitrary())
})


# Method to get estimated revenue (in tsd. USD) from a given company headcount and
# optional industry, country, start year and start month.
@name_space.route('/casing/')
class CaseStrings(Resource):
	# Defining a get rest functionality to get disambiguated location object
	@name_space.doc(security='apikey')
	@api.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'},
			  params={'strings': 'Specify some strings to change casing on - list[str]',
				 'mode': 'specify the casing mode, from ["upper", "lower", "title"], - str'})
	@api.expect(casing_data, validate=False)
	@key_required
	def post(self):
		# Triple quoted comments around methods here will display in the swagger UI. You can use markup to
		# Generate nice looking descriptions here
		"""Returns an array of strings with specified casing.
		<strong>Implementation Notes</strong>.
		<p>
		Send a list of strings and a casing mode to this api, the magical  machine will do it's work and return the
		strings cased as specified. Casings include the following:
		<ul> <li>upper (upper cased)</li>
		<li>lower (lower cased)</li>
		<li>title (title cased)</li></ul></p>
		"""
		try:
			data = api.payload

			results = []

			if data['mode'] == 'upper':
				for item in data['strings']:
					results.append(item.upper())


			elif data['mode'] == 'lower':
				for item in data['strings']:
					results.append(item.upper())



			elif data['mode'] == 'title':
				for item in data['strings']:
					results.append(item.title())

			else:
				raise ValueError('Please select mode value from ["upper", "lower", "title"]')

			return {'status': 200, 'results': results}

		except KeyError as e:
			name_space.abort(500,  status="Could not retrieve necessary payload information", statusCode="500")

		except Exception as e:
			name_space.abort(400,  status="Error within app", statusCode="400")

# Method to get estimated revenue (in tsd. USD) from a given company headcount and
# optional industry, country, start year and start month.
@name_space.route('/nodes/')
class ManageNodes(Resource):

	# Defining a get rest functionality to get disambiguated location object
	@name_space.doc(security='apikey')
	@api.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Internal Server Error'},
			  params={'entityType': 'Specify the entity type to filter the resulting nodes. Common entity types are PERSON, ORG, GPE'})
	@key_required
	def get(self):
		"""Get nodes by entity type
		<strong>Implementation Notes</strong>.
		<p>
		Get a list of nodes with all the associated properties. It is possible to filter
		by entity type.
		</p>
		"""
		try:

			entity_type = request.args.get('entityType')
			results = graph.get_nodes(entity_type)

			return {'status': 200, 'nodes': results}

		except KeyError as e:
			name_space.abort(400,  status="Could not retrieve necessary payload information", statusCode="400")

		except Exception as e:
			name_space.abort(500,  status="Error within app: " + str(e), statusCode="500")


	# Defining a get rest functionality to get disambiguated location object
	@name_space.doc(security='apikey')
	@api.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Internal Server Error'})
	@api.expect(graph_data, validate=False)
	#@key_required
	def post(self):
		"""Create nodes and connections
		<strong>Implementation Notes</strong>.
		<p>
		Insert entities as nodes to the graph and generates connections between each pair of nodes,
		generating a fully connected graph.
		</p>
		"""
		try:
			data = api.payload
			success = graph.create_nodes(data['entities'])

			if(success):
				return {'status': 200, 'message': 'Nodes and relationships created successfully'}
			else:
				name_space.abort(400,  status="Could not retrieve necessary payload information", statusCode="400")

		except KeyError as e:
			name_space.abort(400,  status="Could not retrieve necessary payload information", statusCode="400")

		except Exception as e:
			name_space.abort(500,  status="Error within app: " + str(e), statusCode="500")



	def put(self):
		pass

	def delete(self):
		pass


if __name__ == '__main__':
	app.run(debug=True)

