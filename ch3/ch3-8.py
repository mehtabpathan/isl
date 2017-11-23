
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import pandas as pd
from statsmodels.formula.api import ols


# In[2]:


autoDataset = pd.read_csv('../datasets/auto.csv')


# In[3]:


autoDatasetNoMissing = autoDataset[['mpg', 'horsepower']].dropna()


# In[4]:


mpgHorsePowerOls = ols(formula='mpg ~ horsepower', data=autoDatasetNoMissing).fit()


# In[5]:


print(mpgHorsePowerOls.summary())

# Is there a relationship between the predictor and the response?
# - 

