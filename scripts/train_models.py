import pickle
import pandas as pd
from model import Model
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier

# Training data
df = pd.read_csv('../sample_tweets.csv', encoding='latin-1')

# Isolate target and tweets
target = df.sentiment
tweets = df.text

def train_and_save(model_type, out):
  model = Model(model_type, tweets, target)
  model.extract_features()
  model.train()

  # Python 3
  # pickle.dump(model, open(out, 'wb'))
  # Python 2
  # pickle.dump(model, open(out.replace('.', '_v2.'), 'wb'), protocol=2)
  pickle.dump(model, open(out, 'wb'))

# Gaussian Naive Bayes
print('Training gaussian NB')
train_and_save(GaussianNB(), '../models/gaussian_naive_bayes.pkl')

# Bernoulli Naive Bayes
print('Training bernoulli NB')
train_and_save(BernoulliNB(), '../models/bernoulli_naive_bayes.pkl')

# SVM
print('Training SVM')
train_and_save(svm.SVC(), '../models/svm.pkl')

# SGD with Elastic Net regularization
print('Training SGD with elastic net regularization')
train_and_save(SGDClassifier(penalty='elasticnet'), '../models/elastic_net.pkl')

# Decision Tree
print('Training decision tree')
train_and_save(tree.DecisionTreeClassifier(), '../models/decision_tree.pkl')
