from sklearn.feature_extraction.text import CountVectorizer

class KeyWordExtractor(object):
    """description of class"""

    def __init__(self, *args, **kwargs):
        #create a vocabulary of words, 
        #ignore words that appear in 85% of documents, 
        self.count_vectorizer = CountVectorizer(max_df = 0.85, min_df = 0.10, max_features = 10000)
        return super().__init__(*args, **kwargs)

    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key = lambda x: (x[1], x[0]), reverse = True)

    def extract_topn_from_vector(self, feature_names, sorted_items, topn = 10):
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
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]
    
        return results

    def get_highest_word_count(self, documentNumber, processedTextDocuments):
        word_count_vector = self.count_vectorizer.fit_transform(processedTextDocuments)
        sorted_items = self.sort_coo(word_count_vector[documentNumber].tocoo())
        #Get feature names (words/n-grams). It is sorted by position in sparse matrix
        feature_names = self.count_vectorizer.get_feature_names()
        n_grams = self.extract_topn_from_vector(feature_names, sorted_items, 3)
        print(n_grams)
        return n_grams