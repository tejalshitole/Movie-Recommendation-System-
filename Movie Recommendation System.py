#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import difflib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[5]:


movies_data=pd.read_csv(r"C:\Users\Dhanashree\Desktop\AI\Movie Recommendation System - movies (2).csv")


# In[6]:


movies_data


# In[7]:


movies_data.shape


# In[8]:


selected_features = ["genres","keywords","tagline","cast","director"]


# In[9]:


selected_features


# In[10]:


movies_data.isna().sum()


# In[11]:


for Feature in selected_features:
    movies_data[Feature] = movies_data[Feature].fillna("")


# In[12]:


movies_data.isna().sum()


# In[13]:


combined_features = movies_data["genres"] + " "+movies_data["keywords"] + " "+movies_data["tagline"] + " "+movies_data["cast"] + " "+movies_data["director"]


# In[14]:


combined_features = movies_data["genres"] + " "+movies_data["keywords"] + " "+movies_data["tagline"] + " "+movies_data["cast"] + " "+movies_data["director"]


# In[15]:


combined_features


# In[16]:


combined_features.shape


# In[17]:


vectorizer = TfidfVectorizer()


# In[18]:


features_vectors = vectorizer.fit_transform(combined_features)


# In[19]:


features_vectors


# In[20]:


print(features_vectors)


# In[21]:


similarity = cosine_similarity(features_vectors)


# In[22]:


print(similarity.shape)


# In[23]:


movie_name = input("enter your favourite movie name : ")


# In[24]:


list_of_all_titles = movies_data["title"].tolist()
print(list_of_all_titles)


# In[30]:


find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)


# In[31]:


find_close_match


# In[32]:


close_match = find_close_match[0]


# In[33]:


close_match


# In[37]:


index_of_the_movie = movies_data[movies_data.title == close_match]["index"].values[0]


# In[38]:


index_of_the_movie


# In[41]:


similarity_score = list(enumerate(similarity[index_of_the_movie]))


# In[42]:


similarity_score


# In[44]:


sorted_similar_movies = sorted(similarity_score, key=lambda x:x[1], reverse = True)


# In[45]:


sorted_similar_movies


# In[46]:


print("movie suggested for you :")
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index = movies_data[movies_data.index==index]["title"].values[0]
    if(i<30):
        print(i,".",title_from_index)
        i=i+1


# In[ ]:




