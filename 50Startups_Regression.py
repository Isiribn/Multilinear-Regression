#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
start=pd.read_csv('50_Startups.csv')
start.head()


# In[2]:


start.dtypes


# In[4]:


start.shape


# In[6]:


start.isnull().any(axis=1)


# In[10]:


start.info()


# In[11]:


start.describe()


# In[12]:


start.corr()


# In[13]:


import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(start)


# In[19]:


pd.get_dummies(start[['State']])


# In[36]:


start[start.duplicated()].any()


# In[21]:


del start['State']


# In[22]:


start.head()


# In[9]:


start1=start.rename({'R&D Spend':'RD'},axis=1)
start1.head()


# In[10]:


start1=start1.rename({'Marketing Spend':'Marketing'},axis=1)
start1.head()


# In[11]:


start1=start1.rename({'Administration':'Admin'},axis=1)
start1.head()


# In[12]:


#Building a model
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf 
model = smf.ols('Profit~RD+Admin+Marketing',data=start1).fit()
model.summary()


# In[13]:


ml_a=smf.ols('Profit~Admin',data = start1).fit()  
ml_a.summary()


# In[14]:


ml_m=smf.ols('Profit~Marketing',data = start1).fit()  
ml_m.summary()


# In[15]:


ml_t=smf.ols('Profit~RD+Marketing',data = start1).fit()  
ml_t.summary()


# In[16]:


ml_al=smf.ols('Profit~Admin+Marketing',data = start1).fit()  
ml_al.summary()


# In[20]:


ml_r=smf.ols('Profit~RD+Admin',data = start1).fit()  
ml_r.summary()


# In[47]:


#prediction 
new_data=pd.DataFrame({'RD':176908.20, 'Admin':129034.50,'Marketing':178345.40}, index=[1])


# In[48]:


new_data


# In[49]:


ml_t.predict(new_data)


# In[50]:


ml_al.predict(new_data)


# In[51]:


#best model is with all the variables
model.predict(new_data)


# In[21]:


import pandas as pd
x=pd.DataFrame({'Model': ['Model with all the variables','Model with only R&D and Marketing','Model with Admin and Marketing','Model with R&D and Admin'],'Rsquare':[model.rsquared,ml_t.rsquared,ml_al.rsquared,ml_r.rsquared]})


# In[22]:


x


# In[ ]:




