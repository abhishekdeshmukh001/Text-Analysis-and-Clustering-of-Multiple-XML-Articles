# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:14:38 2023

@author: Shahu
"""

import os
import pandas as pd
import numpy as np
import bs4 as bs
import urllib.request
import spacy
import re, string, unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()

os.chdir(r'D:\7.30 PM DATA SCIENCE AND AI\RESUME PROJECTS\XML - Web Scrapping\xml_many articles')

from glob import glob

path = r'D:\7.30 PM DATA SCIENCE AND AI\RESUME PROJECTS\XML - Web Scrapping\xml_many articles'
all_files = glob(os.path.join(path, '*.xml'))

import xml.etree.ElementTree as ET

dfs = []

# Parse a collection of XML files
for filename in all_files:
    tree = ET.parse(filename)
    root = tree.getroot()
    root = ET.tostring(root, encoding='utf8').decode('utf8')
    dfs.append(root)
    
# Access the first document in the list    
dfs[0]


def data_preprocessing(each_file):
    parsed_article = bs.BeautifulSoup(each_file, 'xml')
    
    # Find and extract text from paragraphs
    paragraphs = parsed_article.find_all('para')
    
    article_text_full = ''
    for p in paragraphs:
        article_text_full += p.text
        print(p.text)
    
    return article_text_full

# Concatenate the text to obtain the full text of each document
data = [data_preprocessing(each_file) for each_file in dfs]


# =============== TEXT CLEANING ========================
def remove_stop_word(file):
    nlp = spacy.load('en_core_web_sm')
    
    punctuations = string.punctuation
    stopwordss = stopwords.words('english')
    SYMBOLS = ' '.join(string.punctuation).split(' ') + ['-','...','â€','â€']
    
    stopwordss = nltk.corpus.stopwords.words('english')+SYMBOLS
    
    doc = nlp(file, disable=['parser','ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwordss and tok not in punctuations]
    
    s = [lem.lemmatize(word) for word in tokens]
    tokens = ' '.join(s)
    
    article_text = re.sub(r'\[[0-9]*\]', ' ', tokens)
    article_text = re.sub(r'\s+', ' ', article_text)
    
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    formatted_article_text = re.sub(r'\W*\b\w{1,3}\b', '', formatted_article_text)
    
    return formatted_article_text


clean_data = [remove_stop_word(file) for file in data]

all_words = ' '.join(clean_data)


# ===================== Visualization ===========================
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, max_font_size=110).generate(all_words)

plt.figure(figsize=(8,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

fredi = word_tokenize(all_words)
freqDist = FreqDist(fredi)
freqDist.plot(100)


# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

vectorizer = CountVectorizer(stop_words=stopwords.words('english')).fit(clean_data)
vectorizer.get_feature_names_out()

X = vectorizer.transform(clean_data).toarray()

data_final = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())


# Tf-IdfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
tran = TfidfTransformer().fit(data_final.values)

X = tran.transform(X).toarray()
X = normalize(X)



# K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2).fit(X)
kmeans.predict(X)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11), wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib.pyplot as plt


distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(1,15) 
  
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(X) 
    kmeanModel.fit(X)     
      
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
    mapping2[k] = kmeanModel.inertia_ 




for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val))

plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show()


'''
for key,val in mapping2.items(): 
	print(str(key)+' : '+str(val)) 


plt.plot(K, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()
'''



