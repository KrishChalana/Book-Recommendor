import streamlit as st
from model import data
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

model = SentenceTransformer('bert-base-nli-mean-tokens')


embeddings = np.load('embeddings.npy')

# Create a search bar
search_term = st.text_input("Search for a book")
search_term_em = model.encode(search_term)
sim_all = []
for i in range(len(embeddings)):
    sim  = cos_sim(search_term_em,embeddings[i])
    sim_all.append(sim)



# Use enumerate to get (index, value) pairs
indexed_list = list(enumerate(sim_all))

# Sort the list based on the values (second element of each tuple)
sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

# Get the indices of the top 10 maximum values
top_10_indices = [index for index, _ in sorted_list[:10]]



# Display book thumbnails and titles based on the search term
for i in top_10_indices:
        st.image(data.iloc[i]['thumbnail'], caption=data.iloc[i]['title'], use_column_width=True)

# Optionally, show a message if no results match the search term
# if not any(search_term.lower() in title.lower() for title in books):
#     st.write("No books found.")

