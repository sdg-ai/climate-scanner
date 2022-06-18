# -*- coding: utf-8 -*-

import setuptools
from setuptools.command.install import install as _install
import os


# overide default install
class CustomInstall(_install):
    def run(self):
        # install required modules
        _install.do_egg_install(self)

        # Download nltk models
        import nltk
        nltk.download("punkt")
        nltk.download("vader_lexicon")
        nltk.download("wordnet")

        # download spacy language model
        import spacy.cli
        spacy.cli.download("en_core_web_lg")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    cmdclass={'install': CustomInstall},
    name="climate_scanner",
    version="1.0",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=setuptools.find_packages(),
    scripts=[],

    # Project uses several external libs.
    install_requires=["numpy==1.19.2", "pandas", "scikit-learn",
                    "keras", "tensorflow", "pytest", "pyyaml", "torch",
                      "transformers"],

    setup_requires=['spacy', 'boto3', 'nltk'],


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