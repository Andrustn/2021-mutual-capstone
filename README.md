# 2021 Mutual Data Science Capstone Project - Appropriate-Checker

## Final Product:
  - A post request with a JSON body with the format:
    - {"message": "message you want to classify"} 
  - Can be sent to the API Gateway endpoint https://orqp9raml1.execute-api.us-east-1.amazonaws.com/appropriate-check and recieve a classification response for the appropriateness of the message. 

## Files and descriptions:
### README.md
  - A file containting file descriptions and instructions for executing code


### appropriate-check
- A folder containing all necessary files to:
  - Create a docker image of all necessary files and packages
  - Push docker image to lambda using the "Serverless" framework
    #### Dockerfile
    - Specifications for docker to allow script to build correctly
    #### requirements.txt
    - A list of all required packages that are installed by building the docker image
    #### serverless.yml
    - A list of specifications to configure Serverless framework - allows integration with AWS Lambda
    #### test.py
    - A python script that constitutes the appropriate-check script. This is the lambda function that is called by API gateway, and does the following:
      - Imports all necessary packages
      - Pulls in model, count vectorizer, and blacklist dictionary from S3
      - Takes individual message input from API Gateway
      - Checks message against blackglist dictionary, if flag word is found, returns Inappropariate
      - If no flag word is found, runs message through previously trained classification model
      - If classified inappropriate, returns Potentially Inappropriate
      - If no inappropriate flags are thrown, returns appropriate

### data_clean_and_train.ipynb
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




# Instructions for Deployment:
- All retraining and deployment can be done with edits to the following two files:
  - mutual_messages.csv
  - dictionary_bad_words.csv
    - By editing mutal_messages.csv, one can change the data with which the model is trained, and update model predictions. 
    - By editing dictionary_bad_words.csv, one can change the flag words which the model will deem blatanlty inappropriate.

- After making edits to either of these files, both data_clean_and_train and lambda_prep must be run in their entirety. The first will retrain and resave the updated model, and the latter will push these changes to S3, where the lambda function can access the updated values automatically. 

- To edit the lambda function itself (change return value formats etc.), edits will need to be made to the test.py file within the appropriate-check folder. After edits on this file are complete, changes will need to be pushed to AWS using the serverless framework. Documentation on this can be found here: https://www.serverless.com.

- As a part of using serverless, AWS CLI credentials with proper permissions will be needed. 


## Example steps for redeployment:

### 1)

1) I want to update and retrain the model with more data, and I want all future messages featuring the word "Applebee's" to be flagged as blatantly inappropriate. 
2) To do so, I first update mutual_messages.csv (or create a new csv file) with the new data I have acquired, making sure to keep the same format as existing data. 
3) Then, I add the word "Applebee's" to the dictionary_bad_words.csv file
4) To create a new model with these changes, I either run data_clean_and_train.py with appropriate command line arguments, or run data_clean_and_train.ipynb in its entirety. This trains the new model and exports the pickle files necessary for updating.
5) To push these changes to lambda, I then run lambda_prep.ipynb - this pushes my newly created pickle files to S3, where the lambda function will pull the new updates.
6) Calls to API Gateway should now reflect the changes I have made. 

### 2)

1) I want to change how the lambda function returns (or otherwise change the structure of how it runs)
2) To do so, I will need to make edits to the test.py file within the appropriate-check folder. 
3) After doing so, I will need to redeploy using the serverless framework. 
4) I first need to run AWS Configure in the command line and input credentials with appropriate permissions
5) I also need to initialize an instance of serverless in the appropriate-check folder. 
6) After everything is configured, running 'sls deploy' while in the appropriate-check directory will push my changes to AWS. 








