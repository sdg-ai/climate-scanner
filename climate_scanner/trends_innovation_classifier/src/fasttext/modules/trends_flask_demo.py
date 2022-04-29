
# -*- coding: utf-8 -*-

import numpy as np
import os
from functools import wraps

# Necessary API imports
from flask import Flask, request
from flask_restplus import Api, Resource, fields
from flasgger import Swagger
import flasgger
import jsonify

import yaml
from .data_utils import load_params
from trends import predict_against_all_individual_classifiers

# Initialise application
app = Flask(__name__)
Swagger(app)

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
name_space = api.namespace('demo_app',
                           description='An api for testing the Trendscanner library machine learning modules.')

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
            # Estimate revenue
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
trends_data = api.model('Trends Model', {
    'text': fields.String(required=True, description="Text string to classify",
                           example="Autonomous transportation and climate change", 
                          cls_or_instance=fields.String)
})

# Method to get estimated revenue (in tsd. USD) from a given company headcount and
# optional industry, country, start year and start month.
@name_space.route('/trends/',methods = ['GET'])
class TrendsClassifier(Resource):
    # Defining a get rest functionality to get disambiguated location object
    @name_space.doc(security='apikey')
    @api.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'},
             params={'text': 'Specify text string to classify - str'})
    @api.expect(trends_data, validate=False)
    @key_required
    def demo_classifier():
        # Triple quoted comments around methods here will display in the swagger UI. You can use markup to
        # Generate nice looking descriptions here
        """Returns a dictionary containing the most likely trends/innovations.
        <strong>Implementation Notes</strong>.
        <p><p>
        Send a text string to this classifier and it will return the most likely trends/innovations
        that the text string belongs to. 

        The expected input:
        <ul> <li>the text string to classify</li></ul>

        The provided output:
        <ul> <li>the most likely classes</li>
        <li>their corresponding probabilities</ul></p>
        """
        try:
            text = "sample text"
            
            threshold = 0.5
            count = 10
            
            most_likely_trends = predict_against_all_individual_classifiers(text, threshold, count)

            return {'status': 200, 
                    'output': jsonify({"string_prediction":[most_likely_trends[i][0] for i in range(len(most_likely_trends))],
                                       "string_prob":[most_likely_trends[i][1] for i in range(len(most_likely_trends))]})}

        except KeyError as e:
            name_space.abort(500,  status="Could not retrieve necessary payload information", statusCode="500")

        except Exception as e:
            name_space.abort(400,  status="Error within app", statusCode="400")

            

if __name__ == '__main__':
    app.run(debug=True)
