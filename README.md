# 2021 Mutual Data Science Capstone Project - Appropriate-Checker

## Files and descriptions:
### README.md
  - A file containting file descriptions and instructions for executing code

### data_prep_and_train.ipynb
- An interactive python notebook that walks through the process of:
  - Pulling in message data from a csv file
  - Cleaning message data necessary for training a classification model
  - Vectorizing data necessary for classification model
  - Creating a classification model using cleaned data
  - Testing model accuracy
  - Saving trained model, vectorizer, and dictionary as pickle files for future use

### data_clean_and_train.py
- A repeatable python script that takes in command line arguments, and performs the process for cleaning data and training a model automatically

### lambda_prep.ipynb
- An interactive python notebook that walks through the process of:
  - Pulling in pickled files created in data_clean_and_train files
  - Uploading previously pickled files to cloud storage (S3)
  - Pulling pickled files from cloud storage (S3)
  - Making predicitions using pickled files pulled from cloud storage (S3)


