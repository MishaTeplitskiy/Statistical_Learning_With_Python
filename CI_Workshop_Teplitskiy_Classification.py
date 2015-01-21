# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Building a Language Detector
# 
# 
# - inspired by http://bugra.github.io/work/notes/2014-12-26/language-detector-via-scikit-learn/

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# <markdowncell>

# ### Example: Language of Tweets

# <codecell>

from IPython.display import Image
Image(url='http://i.imgur.com/Kr5sfJ8.png')

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
#                              analyzer='char'
                                )


X_features = vectorizer.fit_transform(df_sub.text)  # fit_transform() is like calling fit() and then predict()
print X_features.shape, type(X_features)

# <markdowncell>

# ###2. Split into train and test sets

# <codecell>

y = df_sub.language.values
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=1)  
#setting random_state=0 to make #sure we all get the same answer

# <codecell>

#composition of train and test sets
print 'Composition of train set:', np.unique(y_train, return_counts=True)
print 'Composition of test set:', np.unique(y_test, return_counts=True)

# <markdowncell>

# ###3. Train model

# <codecell>

est = LogisticRegression()
est.fit(X_train, y_train)
y_predicted = est.predict(X_test)

# <markdowncell>

# ###4. Evaluate model
# 
# * **accuracy**: percent correct
# 
# 
# * When especially interested in a particular class, say "positive,"
#     - **precision**: of the things you called "positive," what percent were correct?
#     - **recall**: of all positive cases, what percent did you find?

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

# ##ROC curve

# <codecell>

from sklearn.metrics import roc_curve, roc_auc_score

y_label_test = np.asarray(y_test == 'lv', dtype=int)
proba = est.predict_proba(X_test)
proba_label = proba[:,1]
fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)

plt.plot(fpr, tpr, '-', linewidth=5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate ("Cost")')
plt.ylabel('True Positive Rate ("Hit rate")')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")   
    

# <markdowncell>

# #Examine the coefficients

# <codecell>

pd.DataFrame(zip(vectorizer.get_feature_names(), np.exp(est.coef_[0]))).sort(1)

