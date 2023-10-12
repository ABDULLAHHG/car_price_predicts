import pandas as pd 
import json 

# And that only for a MainModel cous the ManyModel should be analytics models not nlp models 
def CreateJsonData(df , file_name : str = 'MainModel'):

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