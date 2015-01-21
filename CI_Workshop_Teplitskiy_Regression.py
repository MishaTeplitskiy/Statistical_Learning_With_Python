# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# <markdowncell>

# ##Data: Boston Housing Prices

# <codecell>

from sklearn.datasets import load_boston
boston = load_boston()

# <codecell>

print boston.DESCR

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

# <codecell>

results = smf.ols(formula='MEDV ~ CRIM + ZN', data=df).fit()
results.summary()

# <markdowncell>

# ##Handling categorical variables

# <codecell>

#Rooms vs Price
#Bias term

# <codecell>


# <codecell>


# <codecell>


