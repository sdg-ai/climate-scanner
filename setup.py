# -*- coding: utf-8 -*-

import setuptools
from setuptools.command.install import install as _install
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="climate_scanner",
    version="1.0",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=setuptools.find_packages(),
    scripts=[],

    # Project uses several external libs.
    setup_requires=["numpy", "pandas", "scikit-learn",
                    "keras", "tensorflow", "pytest"],


    package_data={
        # If any package contains these extensions, include them:
        '': ['*.txt', '*.ods', '*.xlsx', '*.tsv', '*.csv', '*.pkl', 'data/*.*',
             'data/*/*.*', 'data/*/*/*.*', 'data/*/*/*', 'data/*/*/*/*'],
    },

    # metadata
    author="The AI for Good Foundation",
    author_email="eliot@ai4good.org",
    description="Climate Scanner - The climate news parsing and ML insight extraction engine",
    license="Open Source",
    keywords="ai4good, AI for Good, Climate Change, machine learning, global news, innovations,"
             "trends, sentiment, scoring, NLP, network modelling",
    url="https://www.github.com/sdg-ai/climate-scanner.git",


    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Proprietary",
        "Operating System :: OS Independent",
    ],
)