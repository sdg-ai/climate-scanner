# -*- coding: utf-8 -*-

###########################################################################
#
# # 	An module to handle DB connection / article source data retrieval
#
###########################################################################

###########################################################################
#
# 	Import the necessary modules.
#
###########################################################################

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor


########################################################################
#
# 	A necessary utility for accessing the data local to the installation.
#
########################################################################

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
	return os.path.join(_ROOT, 'data', path)

# Path to db configuration file
DB_CONFIG = get_data('db_config.conf')

def get_credentials(requested_parameter, filepath=None):
	"""
	Returns value for the requested parameter
	:param requested_parameter: string
	:param config_file: string
	:return: string
		value for the requested parameter, else None
	"""
	with open(filepath) as f:
		content = f.readlines()
		for line in content:
			parameter, value = line.split("=")
			if parameter == requested_parameter:
				return value.rstrip()

		return None


class SourcesDB(object):
	"""
	Person Db

	"""

	def __init__(self):
		# Retrieve credentials
		db_host = get_credentials('HOST', DB_CONFIG)
		db_name = get_credentials('DBNAME', DB_CONFIG)
		db_user = get_credentials('USERNAME', DB_CONFIG)
		db_pass = get_credentials('PASSWORD', DB_CONFIG)

		# connection to read from the database
		self.read_db_con = psycopg2.connect(database=db_name,
											host=db_host, user=db_user,
											cursor_factory=RealDictCursor, port=5432, password=db_pass)

	def get_sources(self):
		pass

	def ingest_sources(self):
		pass

	def update_sources(self):
		pass