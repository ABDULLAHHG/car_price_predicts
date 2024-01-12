import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import json 
import pickle 

class StopTrainingAtAccuracyOne(tf.keras.callbacks.Callback):
    def on_epoch_end(self , epoch , logs = None ):
        if logs['accuracy'] >= 1.0:
            self.model.stop_training = True



def BuildModel(file_name : str  = 'MainModel' , SplitData : bool = 0):
    # Read Json file 
    data = json.load(open(f'Data/{file_name}.json'))

    # Get pattrens and tags from data 
    patterns = []
    tags = []

    for intent in data['intents']:
        patterns.extend(intent['patterns'])
        tags.extend([intent['tag']] * len(intent['patterns']))
    
    # print(data)
    # Tokenize the patterns
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(patterns)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(patterns)

    # Pad sequences
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences , maxlen = max_sequence_length)

    # Convert tags to numerical labels 
    label_dict = {tag : i for i ,tag in enumerate(set(tags))}
    labels = [label_dict[tag] for tag in tags]
    labels = np.array(labels)

    # Show result 
    print(labels)
    print()
    print(sequences[0])
    
    # Define LSTM Model 
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.Embedding(len(word_index) + 1 ,100 , input_length = max_sequence_length),
        tf.keras.layers.LSTM(128 , return_sequences = True),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_dict) , activation = 'softmax')

    ]) 

    # Compile the model 
    model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

    # Train the model 
    hist = model.fit(padded_sequences ,  labels , epochs = 100 , batch_size = 32 , callbacks = [StopTrainingAtAccuracyOne()])

    # Save the model 
    # I think its no needed cous this should be only for MainModel so SplitData is no needed 
    if SplitData:
        model.save(f'Data/Models/SplitData/{file_name} {round(model.history.history["accuracy"][-1] , 2)}.h5' , hist)
        pickle.dump( tokenizer, open(f'Data/Pickle/SplitData/{file_name}_tokenizer.pkl' , 'wb'))
        pickle.dump( max_sequence_length, open(f'Data/Pickle/SplitData/{file_name}_max_sequence_length.pkl' , 'wb'))
        pickle.dump( label_dict, open(f'Data/Pickle/SplitData/{file_name}_label_dict.pkl' , 'wb'))
    
    else:
        model.save(f'Data/Models/{file_name} {round(model.history.history["accuracy"][-1] , 2)}.h5' , hist)
        pickle.dump(tokenizer , open(f'Data/Pickle/{file_name}_tokenizer.pkl' , 'wb'))
        pickle.dump(max_sequence_length , open(f'Data/Pickle/{file_name}_max_sequence_length.pkl' , 'wb'))
        pickle.dump(label_dict , open(f'Data/Pickle/{file_name}_label_dict.pkl' , 'wb'))



BuildModel()







