import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load your data and embeddings
from model import data  # Make sure to import your 'data' and 'embeddings' variables
embeddings = np.load('embeddings.npy')
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Page title
st.title("Book Search App")

# Create a search bar with a placeholder
search_term = st.text_input("Search for a book", "Enter book title or keywords...")

if search_term:
    # Encode the search term
    search_term_em = model.encode(search_term)
    sim_all = [util.pytorch_cos_sim(search_term_em, embeddings[i]) for i in range(len(embeddings))]

    # Sort and get the top 10 matches
    top_10_indices = np.argsort(sim_all, axis=None)[-10:][::-1]

    if len(top_10_indices) > 0:
        st.subheader("Top 10 Matching Books")

        # Display book thumbnails and titles with a smaller image size
        for i in top_10_indices:
            st.image(data.iloc[i]['thumbnail'], caption=data.iloc[i]['title'], use_column_width=False, width=100)
    else:
        st.warning("No books found matching your search term.")
else:
    st.info("Enter a search term above to find books.")
