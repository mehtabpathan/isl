
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


# Load the data set
boston = load_boston()


# In[3]:


# Read about the data set
print(boston.DESCR)


# In[4]:


data = pd.DataFrame(data=boston.data, columns=boston.feature_names)


# In[5]:


sns.pairplot(data, x_vars=["CRIM", "LSTAT", "NOX", "INDUS"], y_vars=data.keys())


# In[6]:


# Are any of the predictors associated with per capita crime rate?
# sns.lmplot(x='AGE', y='CRIM', data=data)
sns.pairplot(data, x_vars=['AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO'], y_vars=['CRIM'], kind="reg")


# In[7]:


# Do any of the suburbs of Boston appear to have particularly high crime rates?
sns.distplot(data['CRIM'])


# In[8]:


sns.distplot(data['TAX'])


# In[9]:


sns.distplot(data['PTRATIO'])


# In[10]:


# How many of the suburbs in this data set bound the Charles river?
len(data[data['CHAS'] == 1])


# In[11]:


# What is the median pupil-teacher ratio among the towns in this data set?
data['PTRATIO'].median()


# In[12]:


# Which suburb of Boston has lowest median value of owneroccupied homes?
data['MEDV'] = boston.target
data['MEDV'].min()


# In[13]:


# What are the values of the other predictors for that suburb
data.loc[data['MEDV'].idxmin]


# In[14]:


#how do those values compare to the overall ranges for those predictors?
data.describe().append(data.loc[data['MEDV'].idxmin])


# In[15]:


#  how many of the suburbs average more than seven rooms per dwelling?
len(data[data['RM'] > 7])


# In[16]:


# More than eight rooms per dwelling?
len(data[data['RM'] > 8])


# In[17]:


# Comment on the suburbs that average more than eight rooms per dwelling.
data[data['RM'] > 7].describe()


# In[18]:


data.describe()

