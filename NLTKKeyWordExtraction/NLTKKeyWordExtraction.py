from FileIO import FileIO
from TextProcessor import TextProcessor
from KeyWordExtractor import KeyWordExtractor
#from NLTKKeyWordExtraction import Term
import nltk

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

#from sklearn.feature_extraction.text import CountVectorizer

processedTextDocuments = []

for document in files.processedDocLibrary:
    processedTextDocuments.append(text_processor.stringifyTokenArray(document))

##create a vocabulary of words, 
##ignore words that appear in 85% of documents, 
#cv = CountVectorizer(max_df=0.85,min_df=0.10,max_features=10000)
keyword_extractor = KeyWordExtractor()
#word_count_vector = keyword_extractor.count_vectorizer.fit_transform(docs)

from sklearn.feature_extraction.text import TfidfTransformer

#tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
#tfidf_transformer.fit(word_count_vector)

#def sort_coo(coo_matrix):
#    tuples = zip(coo_matrix.col, coo_matrix.data)
#    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

#def extract_topn_from_vector(feature_names, sorted_items, topn=10):
#    """get the feature names and tf-idf score of top n items"""
    
#    #use only topn items from vector
#    sorted_items = sorted_items[:topn]

#    score_vals = []
#    feature_vals = []

#    for idx, score in sorted_items:
#        fname = feature_names[idx]
        
#        #keep track of feature name and its corresponding score
#        score_vals.append(round(score, 3))
#        feature_vals.append(feature_names[idx])

#    #create a tuples of feature,score
#    #results = zip(feature_vals,score_vals)
#    results= {}
#    for idx in range(len(feature_vals)):
#        results[feature_vals[idx]]=score_vals[idx]
    
#    return results


#def get_highest_word_count(documentNumber):
#    sorted_items = sort_coo(word_count_vector[documentNumber].tocoo())
#    #Get feature names (words/n-grams). It is sorted by position in sparse matrix
#    feature_names = cv.get_feature_names()
#    n_grams = extract_topn_from_vector(feature_names,sorted_items,10)
#    print(n_grams)

temp_doc_results = []
sorted_doc_wordcounts = {}

temp_doc_results.append(keyword_extractor.get_highest_word_count(0, processedTextDocuments))
temp_doc_results.append(keyword_extractor.get_highest_word_count(1, processedTextDocuments))
temp_doc_results.append(keyword_extractor.get_highest_word_count(2, processedTextDocuments))
temp_doc_results.append(keyword_extractor.get_highest_word_count(3, processedTextDocuments))
temp_doc_results.append(keyword_extractor.get_highest_word_count(4, processedTextDocuments))
temp_doc_results.append(keyword_extractor.get_highest_word_count(5, processedTextDocuments))

def combine_documents_results(document_results):
    for key in document_results:
        sorted_doc_wordcounts[key] = document_results[key]

for item in temp_doc_results:
    combine_documents_results(item)

sorted_weight = sorted(sorted_doc_wordcounts.items(), key=lambda x:x[1], reverse = True)
 
print(sorted_weight)


print("#########################")
document_sentences = {}
document_sentences["doc1"] = text_processor.convert_to_sentences(files.documentLibrary[0])
document_sentences["doc2"] = text_processor.convert_to_sentences(files.documentLibrary[1])
document_sentences["doc3"] = text_processor.convert_to_sentences(files.documentLibrary[2])
document_sentences["doc4"] = text_processor.convert_to_sentences(files.documentLibrary[3])
document_sentences["doc5"] = text_processor.convert_to_sentences(files.documentLibrary[4])
document_sentences["doc6"] = text_processor.convert_to_sentences(files.documentLibrary[5])


class Term(object):
    """Holds information relating to terms found in documents"""

    def __init__(self, _term, term_count, *args, **kwargs):
        self.term = _term
        self.total_term_count = term_count
        self.term_in_documents = []
        self.term_in_sentences = []
        return super().__init__(*args, **kwargs)






terms = []

for term in sorted_weight:
    #print(term)
    temp_term = Term(term[0], term[1])
    for key in document_sentences:
        #print(key)
        term_count_in_document = 0
        for sentence in document_sentences[key]:
            if term[0] in sentence.casefold():
                temp_term.term_in_sentences.append(sentence)
                term_count_in_document += 1
                if term_count_in_document == 1:
                    temp_term.term_in_documents.append(key)
                #print(sentence)
    terms.append(temp_term)


print("----------------------------------------")

print(terms[0].term)
print(terms[0].total_term_count)
print(terms[0].term_in_documents)
print(terms[0].term_in_sentences)





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








