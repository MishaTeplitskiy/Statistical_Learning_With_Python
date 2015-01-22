# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Regression Example
# 
# - Regression = continuous outcome variable (e.g. "price")
# - Classification = discrete outcome variable (e.g. "class")
# 
# ###Goal: Find "best" line
# 
# Example with only one predictor, "Year"
# 
# <img src=http://www.sigmazone.com/images/Pit91_Scatter_Error.gif>
# 
# ###Solution:
# - Minimize squared distances ("errors")
# 
# ###Evaluation
# - model fit: R^2 
# <img src=http://www.rapidinsightinc.com/wordpress/wp-content/uploads/r-squared-710x400.png>
# 
# ###Coefficients
# 
# size? statistical significance?
# - Usually calculate: **If** the actual coefficient was 0, how often would we see a coefficient *estimate* this large or larger?
#     - *"How consistent is the evidence with the null hypothesis?"*
# - Not exactly what we want
# - In practice, good enough if p < 0.05

# <codecell>

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# <markdowncell>

# ##Data: Boston Housing Prices

# <markdowncell>

# <img src=http://25.media.tumblr.com/d51ffb3725335088ab12fa34bf4c134c/tumblr_mle1snD31n1qe5ugfo6_250.gif>

# <markdowncell>

# ###How much do people care about pollution?

# <codecell>

from sklearn.datasets import load_boston
boston = load_boston()

# <codecell>

print boston.DESCR

# <markdowncell>

# ###Set up our df

# <codecell>

column_names = [
        'CRIM',     #per capita crime rate by town
        'ZN',       #proportion of residential land zoned for lots over 25,000 sq.ft.
        'INDUS',    #proportion of non-retail business acres per town
        'CHAS',     #Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        'NOX',      #nitric oxides concentration (parts per 10 million)
        'RM',       #average number of rooms per dwelling
        'AGE',      #proportion of owner-occupied units built prior to 1940
        'DIS',      #weighted distances to five Boston employment centres
        'RAD',      #index of accessibility to radial highways
        'TAX',      #full-value property-tax rate per $10,000
        'PTRATIO',  #pupil-teacher ratio by town
        'B',       # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        'LSTAT']  #  % lower status of the population
       # 'MEDV'] #     Median value of owner-occupied homes in $1000's

df = pd.DataFrame(boston.data, columns=column_names)
df['MEDV'] = boston.target

# <codecell>

df.columns

# <markdowncell>

# ### Let's look at value and crime

# <codecell>

plt.scatter(df.CRIM, df.MEDV)

# <codecell>

results = smf.ols(formula='standardize(MEDV) ~ C(CRIM)', data=df).fit()
results.summary()

# <codecell>

results.params

# <codecell>

plt.scatter(df.CRIM, df.MEDV)
xs = np.linspace(0, df.CRIM.max(), 100)
plt.plot(xs, xs*results.params[0], 'r--')

# <markdowncell>

# <img src=http://img1.wikia.nocookie.net/__cb20121212025209/glee/images/4/4c/PaulaWut.gif>

# <markdowncell>

# ###Let's add an intercept

# <codecell>

results = smf.ols(formula='MEDV ~ CRIM', data=df).fit()
results.summary()

# <codecell>

plt.scatter(df.CRIM, df.MEDV)
xs = np.linspace(0, df.CRIM.max(), 100)
plt.plot(xs, results.params[0] + xs*results.params[1], 'r--')

# <markdowncell>

# ##Handling categorical variables, standardizing

# <markdowncell>

# ###by using the R-style formula
# - C(var_name)
# - standardize(var_name)

# <markdowncell>

# #Exercise:
# Predict house prices as well as you can!

