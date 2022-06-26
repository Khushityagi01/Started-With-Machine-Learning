#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


df = pd.read_csv("D:/ML/Dataset/houseprices.csv")


# In[3]:


df


# In[4]:


df.head(5)


# In[5]:


plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[6]:


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[7]:


reg.coef_


# In[8]:


reg.intercept_


# In[9]:


135.78767123*3300+180616.43835616432


# In[10]:


reg.predict([[3300]])


# In[11]:


reg.predict([[5000]])


# In[12]:


d = pd.read_csv("book1.csv")


# In[13]:


d


# In[14]:


d.head(4)


# In[15]:


p=reg.predict(d)


# In[16]:


d['prices'] = p


# In[17]:


d


# In[18]:


d.to_csv("predictionhouseprices.csv")


# In[19]:


plt.xlabel('area',fontsize=20)
plt.ylabel('price',fontsize=20)
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')


# In[ ]:




