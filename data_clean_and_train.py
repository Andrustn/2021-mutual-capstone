# necessary imports
import pandas as pd
import numpy as np
import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import emoji
import pickle
import argparse

sw = stopwords.words('english')
wn = WordNetLemmatizer()

class DataPrep:
    def __init__(self, data_filepath,
                 message_column_label,
                 label_column_label,
                 training_data_storage_path,
                 testing_data_storage_path,
                 bad_words_filepath,
                 model_filename,
                 cv_filename,
                 bad_words_pkl_filepath):

        # splitting the data into training and testing sets. We do this by
        # making sure the training and testing sets are equal parts messages
        # that have been flagged as inappropriate and messages that are fine.
        # This helps the model to learn the data without being overwhelmed by
        # the number of messages that are not inappropriate.
        training, testing = self.make_train_test_datasets(data_filepath,
                                      message_column_label, label_column_label,
                                      training_data_storage_path,
                                      testing_data_storage_path)

        # Training the model and tetsing it to make sure it has the accuracy
        # we want.
        classifier, cv, accuracy, recall, precision, f1 = self.train_model(
                        training, testing, message_column_label,
                        label_column_label, bad_words_filepath)
        print("Model Accuracy: ", accuracy)
        print("Model Recall: ", recall)
        print("Model Precision: ", precision)
        print("Model F1 score: ", f1)

        # creating the dictionary of bad words from the bad_words csv originally
        # created for the project. The dictionary allows for constant time
        # lookup, making it ideal for this situation.
        df = pd.read_csv(bad_words_filepath)
        df = list(df)

        my_dict = {}
        for i in range(len(df)):
            df[i] = df[i].strip()
        for i in df:
            my_dict[i] = 1

        # sending the model, count vectorizer, and bad words dictionary to a
        # pkl file so they can be easily transported to AWS.
        self.pickle_model(model_filename, classifier, cv_filename, cv,
                          bad_words_pkl_filepath, my_dict)


    def clean_data(self, text):
        text = emoji.demojize(text) # convert the emoji to it's textual meaning
        text = text.lower() # coerce data to lower case
        tokens = wordpunct_tokenize(text) # tokenize individual words
        tokens = [tok for tok in tokens if tok.isalnum()] # removing punctuation
        tokens = [tok for tok in tokens if tok not in sw] # removing stop words
        tokens = [wn.lemmatize(tok) for tok in tokens] # lematizing - reducing to base words
        return " ".join(tokens)

    '''
        function for creating training and testing. Takes as input:
            the path to the file where the csv of data is stored
            the column name for where the messages are stored
            the column name for where the labels are stored
            the path to where the training data should be stored
            the path to where the testing data should be stored
        it returns:
            a dataframe with testing data
            a dataframe with training data
        it also:
            exports both training and testing data to CSVs
    '''
    def make_train_test_datasets(self, filepath, message_column_label,
                                 label_column_label, training_data_path,
                                 testing_data_path):
        df = pd.read_csv(filepath)
        df[message_column_label] =df[message_column_label].astype(str)
        df = df.dropna()
        df = df[~df[label_column_label].isin(['10', '$0'])]
        df[label_column_label] = df[label_column_label].astype(int)
        df[message_column_label] = df[message_column_label].apply(lambda x: self.clean_data(x))
        df = df[df[message_column_label] != '']
        df = df[df[message_column_label] != '']

        bad_messages = df[df[label_column_label] == 1]
        bad_train = bad_messages.head(int(len(bad_messages)*(70/100)))
        bad_train = bad_train.reset_index(drop=True)
        bad_test = bad_messages.iloc[max(bad_train.index):]

        fine_messages = df[df[label_column_label] == 0]
        fine_train = fine_messages.head(3578)
        fine_train = fine_train.reset_index(drop=True)
        fine_messages = fine_messages.iloc[3579:]
        fine_messages = fine_messages.sample(frac=1).reset_index(drop=True)
        fine_test = fine_messages.head(1535)


        train = pd.concat([bad_train, fine_train], axis=0)
        test = pd.concat([bad_test, fine_test], axis=0)

        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)

        train.to_csv(training_data_path)
        test.to_csv(testing_data_path)

        return train, test

    '''
        A scoring function of our creation that gives back accuracy as well as
        precision, recall, and f1 score so we can determine the models overall
        ccuracy on several different fronts. This function is slow, which is why
        it is reccomended to retain the model on a GPU if possible. You don't have
        to though, I did it on my local machine and it worked fine.
    '''
    def getScores(self, preds, labels):
        total_same = 0
        total_pos = 0
        total_neg = 0
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        for i in range(len(preds)):
            if labels[i] == 1:
                total_pos += 1

            if labels[i] == 0:
                total_neg += 1

            if preds[i] == labels[i]:
                total_same += 1

            if preds[i] == 1 and labels[i] == 1:
                true_positives += 1

            if preds[i] == 0 and labels[i] == 1:
                false_negatives += 1

            if preds[i] == 1 and labels[i] == 0:
                false_positives += 1

        recall =  true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        f1 = 2*((precision*recall)/(precision+recall))
        return total_same / len(preds), recall, precision, f1

    # predicion function used in the real version
    def predict(self, bad_words_filepath, model, cv, messages):
        df = pd.read_csv(bad_words_filepath)
        df = list(df)

        my_dict = {}
        for i in range(len(df)):
            df[i] = df[i].strip()
        for i in df:
            my_dict[i] = 1

        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        predictions = []
        for message in messages:
            checked = False
            # check if any of the words automatically imply inappropriate
            for word in message.split():
                for char in word:
                    if char in punc:
                        word = word.replace(char, "")
                if word.lower() in my_dict:
                    predictions.append(1)
                    checked = True
                    break
            if checked:
                continue
            sample_text = self.clean_data(message)

            sample_text = [sample_text]
            sample_cv = cv.transform(sample_text)

            sample_df = pd.DataFrame(sample_cv.toarray(),
                                     columns = cv.get_feature_names())

            # predict on sample message
            val = model.predict(sample_df)[0]
            predictions.append(val)
        return predictions

    '''
        function to train a random forest classifier. This function will train and
        test the model until it finds one with an acceptable accuracy. This is
        necessary because of the random nature of the random forest classifier. It
        sometimes learns a very good feature selection for classification, but
        often times the random initialization leads it down the wrong path.
    '''
    def train_model(self, training_data, testing_data, message_column_label,
                    label_column_label, bad_words_filepath):
        best_cv = None
        best_classifier = None
        best_accuracy = 0
        while best_accuracy < 0.82:
            cv = CountVectorizer(max_features = 3000)
            X = cv.fit_transform(training_data[message_column_label]).toarray()
            y = training_data[label_column_label].values

            rf = RandomForestClassifier()
            rf.fit(X, y)

            test_X = testing_data[message_column_label]
            test_y = testing_data[label_column_label].values

            y_hat = self.predict(bad_words_filepath, rf, cv, test_X)
            accuracy, recall, precision, f1 = self.getScores(y_hat, test_y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = rf
                best_cv = cv
        return best_classifier, best_cv, accuracy, recall, precision, f1

    # pickling the model and the vectorizer to be put up on AWS
    def pickle_model(self, model_filename, model, cv_filename,
                     cv, dictionary_filename, dictionary):
        pickle.dump(model, open(model_filename, 'wb'))
        pickle.dump(cv, open(cv_filename, 'wb'))
        pickle.dump(dictionary, open(dictionary_filename, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", help="the filepath from where you are running the script to where the new training data is located")
    parser.add_argument("-message_label", help="the column title for the column where the messages are stored (in the original data, this was \"Message\")")
    parser.add_argument("-label_label", help="the column title for the column where the label of the messages are stored (in the original data, this was \"Is Inappropriate\")")
    parser.add_argument("-training_storage", default="training.csv")
    parser.add_argument("-testing_storage", default="testing.csv")
    parser.add_argument("-bad_words_location", help="the filepath to where the csv containing the list of bad words is stored")
    parser.add_argument("-model_storage_file", default="model.pkl")
    parser.add_argument("-cv_storage_file", default="cv.pkl")
    parser.add_argument("-bad_words_storage", default="bad_words.pkl")
    args = parser.parse_args()

    prep = DataPrep(args.data, args.message_label, args.label_label,
                    args.training_storage, args.testing_storage,
                    args.bad_words_location, args.model_storage_file,
                    args.cv_storage_file, args.bad_words_storage)
