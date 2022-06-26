#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


df = pd.read_csv("Linear Regression Multiple Variable.csv")


# In[3]:


df.head()


# In[4]:


import math
median_bedroom = math.floor(df.bedroom.median())


# In[5]:


median_bedroom


# In[6]:


df.bedroom=df.bedroom.fillna(median_bedroom)


# In[7]:


df


# In[8]:


reg = linear_model.LinearRegression()
reg.fit(df[['area','bedroom','age']],df.price)


# In[9]:


reg.coef_


# In[10]:


reg.intercept_


# In[11]:


reg.predict([[3000,3,15]])


# In[12]:


137.25*3000+-26025*3+-6825*15+383725.0


# In[13]:


reg.predict([[2500,4,5]])


# In[ ]:




