import pandas as pd 

df = pd.read_csv('Data/CSV/data.csv')
    

def CreateCsvFile(brand, SplitData : bool = 1):
    Brand_Data = df[df.brand.str.contains(brand)]
    
    Brand_Data.to_csv(f'Data/CSV/SplitData/{brand}')

def CreateManyCsv():
    brands = df.brand.unique()
    for brand in brands:
        CreateCsvFile(brand)

CreateManyCsv()