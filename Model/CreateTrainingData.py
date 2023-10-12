import pickle 
import pandas as pd 
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Loads data 
classes = pickle.load(open('MainModel_classes.pkl' , 'rb'))
words = pickle.load(open('MainModel_words.pkl' , 'rb'))
corpus = pickle.load(open('MainModel_corpus.pkl' , 'rb'))

# Create empty list for training data 
training = []

# Create an empty list for output that matchs the size of classes tags 
empty_array = [0] * len(classes)

for cor in corpus:
    bag = []
    pattren_words  = cor[0]
    
    pattren_words = [lemmatizer.lemmatize(word) for word in pattren_words]

    for word in words:
        bag.append(1) if word in pattren_words else bag.append(0)

    output_row = list(empty_array)
    output_row[classes.index(cor[1])] = 1

    training.append((bag , output_row))

pickle.dump(training  , open('MainModel_training.pkl' , 'wb'))