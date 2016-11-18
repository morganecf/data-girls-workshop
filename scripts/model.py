from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Stop words
stop_words = set(get_stop_words('en'))

# Mention/url regexes
mention = r'^(.*\s)?@\w+'
url = r'^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$'

def preprocess(tweet):
    tweet = tweet.replace(mention, '').replace(url, '')
    return tweet

class Model:
    '''
    Create wrapper class for model so that we can save and reuse it
    '''
    def __init__(self, model, train_data, target):
        self.model = model
        self.train_data = train_data
        self.target = target

    def extract_features(self):
      stop_words = set(get_stop_words('en'))

      # Get counts
      self.word_vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer='word', preprocessor=preprocess, stop_words=stop_words)
      word_doc_matrix = self.word_vectorizer.fit_transform(self.train_data)

      # Get tf-idf
      self.tfidf_transformer = TfidfTransformer(use_idf=True)
      self.features = self.tfidf_transformer.fit_transform(word_doc_matrix)

      return self.word_vectorizer, self.tfidf_transformer

    def train(self):
      if self.features is not None:
        self.classifier = self.model().fit(self.features.toarray(), self.target)
      else:
        raise Exception('Model requires features before training')

    def predict(self, tweets):
      if self.classifier:
        counts = self.word_vectorizer.transform(tweets)
        tfidfs = self.tfidf_transformer.transform(counts)
        predictions = self.classifier.predict(tfidfs.toarray())
        return predictions
      raise Exception('Trained classifier is required before predictions can be made')



