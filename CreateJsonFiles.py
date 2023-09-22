import pandas as pd 
import json 


df = pd.read_csv('data.csv')

# Cleaned dataframe 
df = df[df.mileage.str.contains(r'km')]
df = df[df.engine_capacity.str.match(r'[0-9]')]

Brand_model = df.groupby('brand')['model'].apply(list)
json_data = {'intents' : []}
for i in Brand_model.index:
    json_data['intents'].append(
    {'tag' : i,
    'patterns' : Brand_model[i],
    'responses' : [f'{i}']})

with open('MainModel.json' , 'w') as json_file:
    json.dump(json_data , json_file , indent=4)