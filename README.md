# Project: Disaster Response Pipeline

This project is part of Udacity Nanodegree 'Data Scientist'.

The goal of this project is to process and classify a large amount of messages, as may occur in a 
natural disaster or other catastrophes to improve organisation of emergency forces.

## Overview

Within this project messages are classified by first cleaning and transforming the data, building up a 
machine learning pipeline and displaying through a web application. 

Features:

- Automated data cleaning and transformation
- ML pipeline integration
- Web application for result visualization

## Requirements

In this project, Python 3 is used as programming language with external packages. Please refer to 
requirements.txt for detailed information about package and version.

## Instructions
Please ensure to set up an enviroment with the required Python libaries

- To run **ETL Pipeline** : Switch to directory `DisasterResponsePipeline\2_data` and run command `python process_data.py` -> this creates file **Messages.db** in the same folder
- To run **ML Pipeline** :  Switch to directory `DisasterResponsePipeline\3_models` and run command `python train_classifier.py` -> this creates file **model.pkl** in the same folder
- To run **Web App** : Switch to directory `DisasterResponsePipeline\1_app` and run command `python run.py` -> copy the address to start the web app in your browser

## ToDo's /Future Improvements

- [ ] Improve efficency and accuracy by enlarging grid search parameter.
- [ ] try more / different pipeline components
