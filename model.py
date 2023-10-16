import pandas as pd
# from sentence_transformers import SentenceTransformer
data = pd.read_csv('books.csv')
data = data.dropna()


data['Input_data'] = data['title'] + ' ' +  data['description'] + ' ' + data['categories'] 

