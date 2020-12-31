#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from statsmodels.graphics.regressionplots import influence_plot
import matplotlib.pyplot as plt
cars = pd.read_csv("ToyotaCorolla.csv", encoding='latin1')
cars.head()


# In[6]:


cars_df=cars[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
cars_df


# In[7]:


cars_df.corr()


# In[8]:


cars_df.info()


# In[9]:


import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(cars_df)


# In[11]:


NA=cars_df.isnull().any(axis=1)


# In[12]:


NA


# In[13]:


NA.sum()


# In[23]:


DUP=cars_df[cars_df.duplicated()]


# In[24]:


DUP.shape


# In[27]:


cars_df[cars_df.duplicated()]


# In[28]:


cars_df=cars_df.drop_duplicates()


# In[29]:


cars_df


# In[36]:


cars_new=cars_df.rename({'Age_08_04':'Age'}, axis=1)


# In[37]:


cars_new=cars_new.rename({'Quarterly_Tax':'QT'},axis=1)


# In[38]:


cars_new


# In[41]:


#build a model
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
model=smf.ols("Price~Age+KM+HP+cc+Doors+Gears+QT+Weight", data=cars_new).fit()
model.summary()


# In[43]:


ml_v=smf.ols('Price~Doors',data = cars_new).fit()  
ml_v.summary()


# In[44]:


ml_c=smf.ols('Price~cc',data = cars_new).fit()  
ml_c.summary()


# In[67]:


ml_v=smf.ols('Price~Doors+cc',data = cars_new).fit()  
ml_v.summary()


# In[46]:


#Calculating VIF
rsq_age = smf.ols('Age~KM+HP+cc+Doors+Gears+QT+Weight',data=cars_new).fit().rsquared  
vif_age = 1/(1-rsq_age)

rsq_km = smf.ols('KM~HP+cc+Doors+Gears+QT+Weight+Age',data=cars_new).fit().rsquared  
vif_km = 1/(1-rsq_km) 

rsq_hp = smf.ols('HP~cc+Doors+Gears+QT+Weight+Age+KM',data=cars_new).fit().rsquared  
vif_hp = 1/(1-rsq_hp) 

rsq_cc = smf.ols('cc~Doors+Gears+QT+Weight+Age+KM+HP',data=cars_new).fit().rsquared  
vif_cc = 1/(1-rsq_cc) 

rsq_door = smf.ols('Doors~Gears+QT+Weight+Age+KM+HP+cc',data=cars_new).fit().rsquared  
vif_door = 1/(1-rsq_door) 

rsq_gear = smf.ols('Gears~QT+Weight+Age+KM+HP+cc+Doors',data=cars_new).fit().rsquared  
vif_gear = 1/(1-rsq_gear) 

rsq_qt = smf.ols('QT~Weight+Age+KM+HP+cc+Doors+Gears',data=cars_new).fit().rsquared  
vif_qt = 1/(1-rsq_qt) 

rsq_wt = smf.ols('Weight~Age+KM+HP+cc+Doors+Gears+QT',data=cars_new).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 


           # Storing vif values in a data frame
d1 = {'Variables':['Age','KM','HP','cc','Doors','Gears','QT','Weight'],'VIF':[vif_age,vif_km,vif_hp,vif_cc,vif_door,vif_gear,vif_qt,vif_wt]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[48]:


#resdiuality
res=model.resid
res


# In[49]:


#normality test using qq-plots
import statsmodels.api as sm
qqplot=sm.qqplot(res, line= 'q')
plt.title("Test for Normality of Residuals (Q-Q plot)")
plt.show


# In[51]:


list(np.where(model.resid>20))


# In[52]:


#residual plot for homoscaediscity
def get_standardized_values (vals) :
    return (vals - vals.mean())/vals.std()


# In[53]:


plt.scatter(get_standardized_values(model.fittedvalues),
           get_standardized_values(model.resid))
plt.title("Residual Plot")
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()


# In[56]:


#model deletion diagnostics
model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[57]:


c


# In[58]:


fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(cars_new)), np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[59]:


# index and value of influencer where c is more than .5
np.argmax(c), np.max(c)


# In[61]:


#high influence points
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show


# In[62]:


cars_new[cars_new.index.isin([80])]


# In[63]:


#improve the model
cars_1 = cars_new.drop(cars_new.index[[80]],axis=0).reset_index()


# In[64]:


cars_1


# In[65]:


car1=cars_1.drop(['index'], axis=1)


# In[66]:


car1


# In[68]:


#building our final model
final_ml_v= smf.ols('Price~Age+KM+HP+cc+Doors+Gears+QT+Weight',data = car1).fit()


# In[69]:


(final_ml_v.rsquared, final_ml_v.aic)


# In[79]:


#predict new data
new_data=pd.DataFrame({'Age':30, 'KM':28895, 'HP':80,'cc':1200, 'Doors':4,'Gears':5, 'QT':180, 'Weight':1102}, index=[1])


# In[80]:


new_data


# In[81]:


final_ml_v.predict(new_data)


# In[84]:


pred_y=final_ml_v.predict(car1)


# In[85]:


pred_y


# In[ ]:




