#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install -U scikit-learn


# In[12]:


from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[13]:


categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x',
              'rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey',
              'sci.crypt','sci.electronics','sci.med','sci.space',
              'misc.forsale','talk.politics.misc','talk.politics.guns','talk.politics.mideast',
              'talk.religion.misc','alt.atheism','soc.religion.christian']

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


#########################
# TRAINING A CLASSIFIER #
#########################
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(twenty_train.data, twenty_train.target)

import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

##########################
# Support vector machine #
##########################
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))

print(metrics.confusion_matrix(twenty_test.target, predicted))

from sklearn.model_selection import GridSearchCV
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

twenty_train.target_names[gs_clf.predict(['God is love'])[0]]

print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

print(gs_clf.cv_results_)


# In[14]:


def printResults(classifier_name, predicted_data, test_data):
    print(f"Metrics Results for {classifier_name}")
    print(metrics.classification_report(test_data.target, predicted_data,target_names=test_data.target_names))
    print(f"The {classifier_name} classifier was able to recognize the test set with this accuracy: "+ str(np.mean(predicted_data == test_data.target)))
def comparePlainClassifiers(train_data, test_data):
    clfByaes = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
        ])
    clfByaes.fit(train_data.data,train_data.target)
    printResults("MultinomialNB",clfByaes.predict(test_data.data), test_data)

    clfSGD = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
        ])
    clfSGD.fit(train_data.data,train_data.target)
    printResults("SGDClassifier",clfSGD.predict(test_data.data), test_data)

    clfRidgeClassifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RidgeClassifier()),
        ])
    clfRidgeClassifier.fit(train_data.data,train_data.target)
    printResults("RidgeClassifier", clfRidgeClassifier.predict(test_data.data), test_data)
comparePlainClassifiers(twenty_train,twenty_test)


# In[15]:


def compareFeatureClassifiers(train_data, test_data):
    classifiers = [
        ('MultinomialNB', MultinomialNB()),
        ('SGDClassifier', SGDClassifier()),
        ('RidgeClassifier', RidgeClassifier())
    ]
    vectorizers = [
        ('CountVectorizer', CountVectorizer()),
        ('TfidfVectorizer', TfidfVectorizer()),
        ('TF', TfidfVectorizer(use_idf=False))
    ]
    for vec_name, vectorizer in vectorizers:
        print(f"Feature Representation: {vec_name}")
        for clf_name, clf in classifiers:
            print(f"Classifier: {clf_name}")
            allpipeline = Pipeline([
                ('vect', vectorizer),
                ('tfidf', TfidfTransformer() if vec_name != 'TF' else None),
                ('clf', clf),
            ])
            allpipeline.fit(train_data.data, train_data.target)
            allpipeline.predict(test_data.data)
            target_names = test_data.target_names
            accuracy = accuracy_score(test_data.target, allpipeline.predict(test_data.data))
            classification_rep = classification_report(test_data.target, allpipeline.predict(test_data.data), target_names=target_names)
            print(f"Accuracy: {accuracy}","Classification Report:\n", classification_rep)


# In[16]:


compareFeatureClassifiers(twenty_train,twenty_test)

