import re

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize.stanford import StanfordTokenizer
import numpy as np 

def scrub_words(text):
    """Basic cleaning of texts."""

    #remove non-ascii and digits
    #text=re.sub("(\\W|\\d)","",text)
    #print(text)
    #split into words
    #from nltk.tokenize import word_tokenize
    #words = word_tokenize(text)
    words = re.split(r'\W+|\d', text)
    # Lower case
    words = [w.lower() for w in words]
    # remove punctuation from each word
    #import string
    #table = str.maketrans('', '', string.punctuation)
    #words = [w.translate(table) for w in words]

    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    file = open("resources/stopwords.txt", 'rt')
    fileText = file.read()
    file.close()
    #fileText = fileText.split()
    #stop_words.union(fileText)
    stop_words = fileText
    words = [w for w in words if not w in stop_words]
    #print(len(words))
    #remove whitespace
    #words = words.strip()
    return words

def lemmatizeWords(tokens):
    # init lemmatizer
    lemmatizer = WordNetLemmatizer()
    #lemmatize trouble variations
    lemmatized_words=[lemmatizer.lemmatize(word=word,pos='v') for word in tokens]
    #cleaned_stemmed_words=[porter_stemmer.stem(word=word) for word in cleaned_words]
    #print(lemmatized_words)
    #print(len(lemmatized_words))
    return lemmatized_words

def loadTextFile(fileLocation):
    file = open(fileLocation, 'rt', encoding = "utf8")
    fileText = file.read()
    file.close()
    return fileText

###########################
# Logic starts here
filePaths = ["resources/doc1.txt", "resources/doc2.txt", "resources/doc3.txt", 
             "resources/doc4.txt", "resources/doc5.txt", "resources/doc6.txt" ]
documentLibrary = []
processedDocLibrary = []

for filePath in filePaths:
    documentLibrary.append(loadTextFile(filePath))

for document in documentLibrary:
    temp = scrub_words(document)
    processedDocLibrary.append(lemmatizeWords(temp))

#print(processedDocLibrary)

#cleaned_words1 = scrub_words(article1)
#lemmatized_words1 = lemmatizeWords(cleaned_words1)

#cleaned_words2 = scrub_words(article2)
#lemmatized_words2 = lemmatizeWords(cleaned_words2)


from sklearn.feature_extraction.text import CountVectorizer

def stringifyTokenArray(tokenArray):
    return " ".join(tokenArray)

docs = []

for document in processedDocLibrary:
    docs.append(stringifyTokenArray(document))

##get the text column 
#docs=[processedArticle1, processedArticle2]
#lemmatized_words1 = " ".join(lemmatized_words1)
#lemmatized_words2 = " ".join(lemmatized_words2)
#docs = [lemmatized_words1, lemmatized_words2]

#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
#eliminate stop words
cv=CountVectorizer(max_df=0.85,min_df=0.10,max_features=10000)
word_count_vector=cv.fit_transform(docs)
#test = cv.fit(docs)
#print(test)

print(word_count_vector.shape)
#for word in word_count_vector:
#    print(word)
#print(list(cv.vocabulary_.keys())[:10])
#print(cv.vocabulary_)
#print("-------------------------")
#for i in word_count_vector:
#    for j in i:
#        print()



from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

#print(tfidf_transformer.idf_)



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


###############################
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("resources\doc1.txt")
data = fp.read()
sentences = tokenizer.tokenize(data)
#print('\n-----\n'.join(sentences))

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








