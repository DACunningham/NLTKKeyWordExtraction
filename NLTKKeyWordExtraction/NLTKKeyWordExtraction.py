#import re
from FileIO import FileIO
from TextProcessor import TextProcessor
import nltk
import pandas as pd
#from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize.stanford import StanfordTokenizer
import numpy as np 

###########################
# Logic starts here
filePaths = ["resources/doc1.txt", "resources/doc2.txt", "resources/doc3.txt", 
             "resources/doc4.txt", "resources/doc5.txt", "resources/doc6.txt" ]
#documentLibrary = []
processedDocLibrary = []

files = FileIO()
files.load_all_files(filePaths)

text_processor = TextProcessor()

for document in files.documentLibrary:
    files.processedDocLibrary.append(text_processor.preprocessor(document))

##################################################

from sklearn.feature_extraction.text import CountVectorizer

docs = []

for document in files.processedDocLibrary:
    docs.append(text_processor.stringifyTokenArray(document))

#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
cv = CountVectorizer(max_df=0.85,min_df=0.10,max_features=10000)
word_count_vector = cv.fit_transform(docs)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


def get_highest_word_count(documentNumber):
    sorted_items = sort_coo(word_count_vector[documentNumber].tocoo())
    #Get feature names (words/n-grams). It is sorted by position in sparse matrix
    feature_names = cv.get_feature_names()
    n_grams = extract_topn_from_vector(feature_names,sorted_items,10)
    print(n_grams)

get_highest_word_count(0)
get_highest_word_count(1)
get_highest_word_count(2)
get_highest_word_count(3)
get_highest_word_count(4)
get_highest_word_count(5)


print("#########################")
#import nltk.data

#def convert_to_sentence():
#    fp = open("resources\doc1.txt")
#    data = fp.read()
#    sentences = tokenizer.tokenize(data)
#    return sentences
#    #print('\n-----\n'.join(sentences))

#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = text_processor.convert_to_sentences(files.documentLibrary[0])

for sents in sentences:
    temp = sents.split()
    term = "generation"
    if term in temp:
        print(sents)

## you only needs to do this once
#feature_names=cv.get_feature_names()

## get the document that we want to extract keywords from
#doc=docs[5]

##generate tf-idf for the given document
#tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

##sort the tf-idf vectors by descending order of scores
#sorted_items=sort_coo(tf_idf_vector.tocoo())

##extract only the top n; n here is 10
#keywords=extract_topn_from_vector(feature_names,sorted_items,20)

## now print the results
##print("\n=====Title=====")
##print(docs_title[0])
##print("\n=====Body=====")
##print(docs_body[0])
#print("\n===Keywords===")
#for k in keywords:
#    print(k,keywords[k])








