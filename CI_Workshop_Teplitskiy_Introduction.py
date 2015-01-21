# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Statistical Learning with Python
# 
# Agenda:
# 
# **Intro and Plugs**
# - Sociology / computational social science
# - <a href=http://www.knowledgelab.org>www.knowledgelab.org</a>
# - <a href=http://www.dssg.io>www.dssg.io</a>
# 
# **If you couldn't get your environment set up!**
# - Use: https://wakari.io/ 
# 
# **IPython Notebook** as your IDE
#     - Advantages/disadvantages
#         - notebook: markdown, code, inline images
#         - server
#         - "--script"
#     - Sharing is caring: http://nbviewer.ipython.org/ 
#     - Keyboard shortcuts

# <markdowncell>

# <img src=http://i.minus.com/iEdBFdHPKBG8Q.gif>

# <markdowncell>

# **Pandas**

# <markdowncell>

#     - Creating Series and DataFrames
#         - Setting column names, index, datatypes
#     - Indexing
#         - By index, by label
#     - Subsetting
#     - Missing values

# <markdowncell>

# <img src=http://img3.wikia.nocookie.net/__cb20131231081108/degrassi/images/9/93/Panda-gif.gif>

# <markdowncell>

# **Matplotlib**
#     - scatter, plot, hist
#     - useful plot customazation
#     - plots inside of pandas
#     
# **Regression Example**
# 
# **Classification Example**

# <markdowncell>

# <img src=http://www.totalprosports.com/wp-content/uploads/2012/11/14-nolan-ryan-high-fives-george-w-bush-gif.gif>

# <codecell>

%matplotlib inline

# <markdowncell>

# #Pandas

# <codecell>

import pandas as pd

# <markdowncell>

# Provides a crucial 2-d data structure: the ``pandas.DataFrame``
#     - pandas.Series is 1-d analogue
#     - Like the ``R`` data frames 
#     
# ``numpy`` does too, BUT ``pandas``
# 
#   1. can hold *heterogenous data*; each column can have its own data type,
#   2. the axes of a DataFrame are *labeled* with column names and row indices, 
# 
# Perfect for data-wrangling: can take subsets, apply functions, join with other DataFrames, etc.

# <codecell>

# Load car dataset
df = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Auto.csv')
df.head()  # print the first lines

# <codecell>

print 'Shape of DataFrame:', df.shape
print '\nColumns:', df.columns
print '\nIndex:', df.index[:10]

# <markdowncell>

# ###Get the ``df`` nice and cozy

# <codecell>

df.index = df.name
del df['name']
df.head()

# <markdowncell>

# ###Accessing, adding data
# You can use the dot ``.`` or bracket ``[]`` notation to access columns of the dataset. To add new columns you have to use the bracket ``[]`` notation

# <codecell>

mpg = df.mpg  # get mpg column using '.'
weight = df['weight']  # get weight column using brackets
df['mpg_per_weight'] = mpg / weight # note the element-wise division

print df[['mpg', 'weight', 'mpg_per_weight']].head()  # get a bunch of columns at the same time

# <markdowncell>

# ##Looking at data

# <markdowncell>

# ###Pandas indexing is really smart!

# <codecell>

# To look at all the Fords, create array of length = #rows of True and False, where True if 'ford' in string
arr_for_indexing = ['ford' in name for name in df.index]
df[arr_for_indexing].head() 

# <markdowncell>

# ###But it can get confused: Indexing by "label" and by "location", ``.loc`` vs ``.iloc`` vs ``.ix``

# <markdowncell>

# ``.loc`` -- by label
# 
# ``.iloc`` -- by location
# 
# ``.ix`` -- by a mix

# <codecell>

df.ix[0:5, ['weight', 'mpg']]  # select the first 5 rows and two columns weight and mpg

# <codecell>

# useful function!: value_counts()
df.year.value_counts()

# <markdowncell>

# ###Let's change year from "70" to "1970" 

# <codecell>

df.year.apply(lambda x: '19' + str(x)) # this spits out the Series we like
# df.year = df.year.apply(lambda x: '19' + str(x))

# <codecell>

#Uh oh, let's change it back!
df.year = df.year.str[-2:]

# <markdowncell>

# #Visualizing data

# <markdowncell>

# Most popular library: ``matplotlib``
# 
# Others: 
# - ``seaborn``
# - ``ggplot``
# - ``prettyplotlib``
# - ``bokeh``

# <markdowncell>

# ### common matplotlib plots
# - plt.hist <-- histograms
# - plt.scatter <-- scatter plot
# - plt.plot <-- most others

# <codecell>

import matplotlib.pyplot as plt
plt.hist(df.weight)

# <markdowncell>

# ### common plot features to tweak
# - plt.title('Sk00l Rox', fontsize=20)
# - plt.xlabel('')
# - plt.ylabel('')
# - plt.xlim(min, max)
# - plt.legend()

# <markdowncell>

# ###We can also used pandas' ``plot`` and other plotting function!!!

# <codecell>

df.weight.hist()
plt.title('OMG THERES A TITLE!!!11', fontsize=20)

# let's add decoration
plt.xlabel('weight')
plt.ylabel('frequency')
plt.xlim(0, df.weight.max())
plt.legend()

# <codecell>

plt.scatter(df.year, df.weight)

# <codecell>

df.boxplot('weight')
# df.boxplot('weight', 'year')

# <codecell>

from pandas.tools.plotting import scatter_matrix
_ = scatter_matrix(df[['mpg', 'cylinders', 'displacement']], figsize=(14, 10))

# <markdowncell>

# #Regression next. But first...

# <codecell>

plt.xkcd()

# <markdowncell>

# <img src=http://replygif.net/i/209.gif>

# <codecell>

df.weight.hist()
plt.title('WOT, THERES AN XKCD STYLE???', fontsize=18)
plt.xlabel('weight')
plt.ylabel('freq.')

