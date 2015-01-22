# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Classification: Building a Language Detector
# 
# 
# - inspired by http://bugra.github.io/work/notes/2014-12-26/language-detector-via-scikit-learn/

# <markdowncell>

# #Overfitting
# 
# ### = Big difference between social science stats and machine learning
# 
# <img src=http://pingax.com/wp-content/uploads/2014/05/underfitting-overfitting.png>
# 
# ###Solution: Split data into training part and testing part
# 
# - "testing" set also called "validation set," "held-out set"
# 
# ###Result: 2 sets of accuracies, 2 sets of errors
# - One for training set <--- no one cares about
# - One for test set <--- everyone cares about, also called "generalization error"
# 
# <img src=https://raw.githubusercontent.com/tijptjik/DS_assets/master/overfitting.png>

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# <markdowncell>

# ##Data Description

# <markdowncell>

# European Parliament Proceedings corpus
# - https://language-detection.googlecode.com/git-history/packages/packages/europarl-test.zip
# - 21 languages, 1000 sentences each 

# <markdowncell>

# ##Import data and put it in pandas dataframe

# <codecell>

import codecs
lines = codecs.open('europarl.txt', 'r', 'utf-8').readlines()
lines = [l.split('\t') for l in lines]

# <codecell>

df = pd.DataFrame(lines, columns=['language', 'text'])
df.head()

# <codecell>

# how many of each language
df.language.value_counts()

# <codecell>

# let's consider just two: english (en) and french (fr)
df[df.language=='en'].head()

# <codecell>

df_sub = df[df.language.isin(('lt', 'lv'))]

# <markdowncell>

# ##Build classifier

# <codecell>

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# <markdowncell>

# ###1. Engineer features we will use to predict

# <codecell>

# vectorizer = TfidfVectorizer(ngram_range=(1,3),
# #                              analyzer='char',
#                              max_features=50)
# #                             use_idf=False)

vectorizer = CountVectorizer(max_features=50,
                             analyzer='char',
                                )


X_features = vectorizer.fit_transform(df_sub.text)  # fit_transform() is like calling fit() and then predict()
print X_features.shape, type(X_features)

# <markdowncell>

# ###2. Split into train and test sets

# <codecell>

y = df_sub.language.values
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2)  
#setting random_state=0 to make #sure we all get the same answer

# <codecell>

#composition of train and test sets
print 'Composition of train set:', np.unique(y_train, return_counts=True)
print 'Composition of test set:', np.unique(y_test, return_counts=True)

# <markdowncell>

# ###3. Train model

# <codecell>

clf = LogisticRegression()
clf.fit(X_train, y_train)

# <markdowncell>

# ###4. Evaluate model
# 
# *Test it on the held-out test set*
# 
# * **accuracy**: percent correct
# 
# 
# * When especially interested in a particular class, say "positive,"
#     - **precision**: of the things you called "positive," what percent were correct?
#     - **recall**: of all positive cases, what percent did you find?

# <codecell>

y_predicted = clf.predict(X_test)

# <codecell>

from sklearn import metrics
print 'Accuracy:', metrics.accuracy_score(y_test, y_predicted)
print
print metrics.classification_report(y_test, y_predicted)
print
print 'confusion matrix'
print
print pd.DataFrame(metrics.confusion_matrix(y_test, y_predicted))

# <markdowncell>

# ###Out of curiousity, how well did we do on the training set?

# <codecell>

print 'Accuracy:', metrics.accuracy_score(y_train, clf.predict(X_train))

# <markdowncell>

# ##ROC curve
# 
# x-axis: What percent of negative things did you falsely call positive?
# 
# y-axis: Of the positive examples, what percent did you find?

# <codecell>

from sklearn.metrics import roc_curve, roc_auc_score

y_label_test = np.asarray(y_test == 'lv', dtype=int)
proba = clf.predict_proba(X_test)
proba_label = proba[:,1]
fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)

plt.plot(fpr, tpr, '-', linewidth=5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate ("Cost")')
plt.ylabel('True Positive Rate ("Benefit")')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")   
    

# <markdowncell>

# #Examine the coefficients

# <codecell>

pd.DataFrame(zip(vectorizer.get_feature_names(), np.exp(clf.coef_[0]))).sort(1)

# <markdowncell>

# #Exercise
# 
# ##Create a classifier for *all* 21 languages
# i.e. Given a sentence, output its most probable language
# 
# **hint**: Create 21 classifiers which classify *langauge x* vs. *all other languages* and choose langauge with highest probability

