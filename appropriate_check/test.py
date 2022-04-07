import sys
import pickle
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import emoji
from nltk import wordpunct_tokenize, word_tokenize, sent_tokenize
import io
import boto3
import sklearn
import json

sys.path.insert(0, '/tmp/')
nltk.data.path.append("/tmp")
nltk.download("stopwords", download_dir = "/tmp")
nltk.download("wordnet", download_dir = "/tmp")
nltk.download('omw-1.4', download_dir = "/tmp")

from nltk.corpus import stopwords
    
bucket = "byu-capstone-appropriate-checker"
key = "model"  
s3 = boto3.client('s3')
response = s3.get_object(Bucket = bucket, Key = key)
model = pickle.loads(response['Body'].read())

bucket = "byu-capstone-appropriate-checker"
key = "count_vectorizer" 
s3 = boto3.client('s3')
response = s3.get_object(Bucket = bucket, Key = key)
cv = pickle.loads(response['Body'].read())

bucket = "byu-capstone-appropriate-checker"
key = "blacklist_dictionary"
s3 = boto3.client('s3')
response = s3.get_object(Bucket = bucket, Key = key)
my_dict = pickle.loads(response['Body'].read())

punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    
    

wn = WordNetLemmatizer()
sw = stopwords.words('english')
sw_special = ["rt"]

def clean_data(text):
    text = emoji.demojize(text)
    text = text.lower() # coerce data to lower case
    tokens = wordpunct_tokenize(text) # tokenize individual words
    tokens = [tok for tok in tokens if tok.isalnum()] # removing punctuation
    tokens = [tok for tok in tokens if tok not in sw] # removing stop words
    tokens = [wn.lemmatize(tok) for tok in tokens] # lematizing lyrics - reducing to base words
    return " ".join(tokens)

def predict(message, model, cv):
    # check if any of the words automatically imply inappropriate
    for word in message.split():
        for char in word:
            if char in punc:
                word = word.replace(char, "")
        if word.lower() in my_dict:
            return word
    sample_text = clean_data(message)
    
    sample_text = [sample_text]
    sample_cv = cv.transform(sample_text)
    
    sample_df = pd.DataFrame(sample_cv.toarray(), columns = cv.get_feature_names())
    
    # predict on sample message
    val = model.predict(sample_df)[0]
    return val




def handler(event=None, context=None):
   
    raw = json.loads(event['body'])
    
    message = raw['message']
    
    code = predict(message, model, cv)
    
    if code == 0:
            return json.dumps({'Code': 400, 'Message': message, 'Value': "Appropriate"})
   
    elif code == 1:
        return json.dumps({'Code': 401, 'Message': message, 'Value': "Potentially Inappropriate"})

    else:
        return json.dumps({'Code': 402, 'Message': message, 'Value': "Inappropriate", "Flag_word":code})
    
    
    

if __name__ == '__main__':
    handler()
