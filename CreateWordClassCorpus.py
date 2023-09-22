from nltk import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle 
import json

data_file = open('MainModel.json').read()
dataset = json.loads(data_file)


lemmatizer = WordNetLemmatizer()

## Clean Text 

# Create a list of all words in the file 
words = []

# Create a list of classes for our tags
classes = []

corpus = []

ignore_chracter = [',' , '.' ,'?']
for data in dataset['intents']:
    for pattren in data['patterns']:
        word = word_tokenize(pattren)
        words.extend(word)

        corpus.append((word , data['tag']))
        if data['tag'] not in classes:
            classes.append(data['tag'])
words = [lemmatizer.lemmatize(word.lower()) for word in word if word not in ignore_chracter]

english_stopwords = list(stopwords.words('english'))

[words.remove(word) for word in words if word in english_stopwords]

## Sort data
# Sort Words 
words = sorted(list(set(words)))

# Sort Classes 
classes = sorted(list(set(classes)))

# Save Words , Classes , Corpus
pickle.dump(words , open('MainModel_words' , 'wb'))
pickle.dump(classes , open('MainModel_classes' , 'wb'))
pickle.dump(corpus , open('MainModel_corpus' , 'wb'))

