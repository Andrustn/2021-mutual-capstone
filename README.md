# 2021 Mutual Data Science Capstone Project - Appropriate-Checker

## Files and descriptions:
### README.md
  - A file containting file descriptions and instructions for executing code


### appropriate-check
- A folder containing all necessary files to:
  - Create a docker image of all necessary files and packages
  - Push docker image to lambda using the "Serverless" framework
    #### Dockerfile
    - Specifications for docker to allow script to build correctly







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

### model.pkl
- Pickled classification model (Random forest) created by data_clean_and_train files

### cv.pkl
- Pickled count vectorizer created by data_clean_and_train files

### bad_words.pkl
- Pickled dictionary of flag words created by data_clean_and_train files

### mutual_messages.csv
- CSV file of labeled message data - message content is accompanied by an "inappropriate" label for model to train on. 

### dictionary_bad_words.csv
- CSV file of "flag words" - a combination of Mutual's own provided words with a publicly available list of words from Facebook's "Blacklist". If any of these words are found in a message, the message is immediately marked inappropriate. 

### training.csv
- A subset of mutal_messages.csv created by data_prep_and_train files that is used to train the classification model

### testing.csv
- A subset of mutal_messages.csv created by data_prep_and_train files that is used to test the classification model

