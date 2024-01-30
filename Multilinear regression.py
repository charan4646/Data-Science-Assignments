#!/usr/bin/env python
# coding: utf-8

# # Question - 1

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[52]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\multilinear\\50_Startups.csv")


# In[53]:


df


# In[54]:


df.shape


# In[56]:


df.info()


# In[57]:


df.isnull().sum()


# In[59]:


df_1=df.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
df_1


# In[60]:


df_1[df_1.duplicated()]


# In[61]:


df_1.describe()


# In[62]:


df_1.corr()


# In[64]:


#EDA
sns.set_style(style='darkgrid')
sns.pairplot(df_1)


# In[65]:


# MODEL BUILDING
# Finding Coefficient parameters
model.params


# In[66]:


# Finding tvalues and pvalues
model.tvalues , np.round(model.pvalues,5)


# In[67]:


# Finding rsquared values
model.rsquared , model.rsquared_adj  # Model accuracy is 94.75%


# In[68]:


# Build SLR and MLR models for insignificant variables 'ADMS' and 'MKTS'
# Also find their tvalues and pvalues
slr_a=smf.ols("Profit~ADMS",data=df_1).fit()
slr_a.tvalues , slr_a.pvalues  # ADMS has in-significant pvalue


# In[69]:


slr_m=smf.ols("Profit~MKTS",data=df_1).fit()
slr_m.tvalues , slr_m.pvalues  # MKTS has significant pvalue


# In[70]:


mlr_am=smf.ols("Profit~ADMS+MKTS",data=df_1).fit()
mlr_am.tvalues , mlr_am.pvalues  # varaibles have significant pvalues


# In[71]:


# MODEL VALIDATION
#1.COLLINEARITY CHECK & 2.RESIDUAL ANALYSIS
# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_r=smf.ols("RDS~ADMS+MKTS",data=df_1).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a=smf.ols("ADMS~RDS+MKTS",data=df_1).fit().rsquared
vif_a=1/(1-rsq_a)

rsq_m=smf.ols("MKTS~RDS+ADMS",data=df_1).fit().rsquared
vif_m=1/(1-rsq_m)

# Putting the values in Dataframe format
d1={'Variables':['RDS','ADMS','MKTS'],'Vif':[vif_r,vif_a,vif_m]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[72]:


# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)

sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[73]:


# MODEL DELETION DIAGNOSIS
#1.COOK'S DISTANCE & 2.LEVERAGE VALUE
# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[75]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[76]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
influence_plot(model)
plt.show()


# In[77]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=df_1.shape[1]
n=df_1.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[78]:


df_1[df_1.index.isin([49])] 


# In[79]:


# IMPROVING THE MODEL
# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
df_2=df_1.drop(df_1.index[[49]],axis=0).reset_index(drop=True)
df_2


# In[80]:


# Model Deletion Diagnostics and Final Model
while np.max(c)>0.5 :
    model=smf.ols("Profit~RDS+ADMS+MKTS",data=df_2).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    df_2=df_2.drop(df_2.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    df_2
else:
    final_model=smf.ols("Profit~RDS+ADMS+MKTS",data=df_2).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)


# In[81]:


final_model.rsquared


# # QUESTION - 2

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[3]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\multilinear\\ToyotaCorolla.csv", encoding='latin1')


# In[4]:


df


# In[5]:


# EDA
df.info()


# In[6]:


df_1 = pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],df.iloc[:,12:14],df.iloc[:,15:18]],axis=1)


# In[7]:


df_1


# In[8]:


df_2=df_1.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)


# In[9]:


df_2


# In[10]:


df_2[df_2.duplicated()]


# In[11]:


df_3=df_2.drop_duplicates().reset_index(drop=True)


# In[12]:


df_3


# In[13]:


df_3.describe()


# In[14]:


df_3.corr()


# In[15]:


sns.set_style(style='darkgrid')
sns.pairplot(df_3)


# In[16]:


Y = df_3['Price']


# In[17]:


Y


# In[18]:


X = df_3.iloc[:,1:]


# In[19]:


X


# In[91]:


Y = df_3['Price']


# In[85]:


#X = df_3[['Age']]


# In[20]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()


# In[21]:


LR.fit(X,Y)


# In[22]:


Y_pred = LR.predict(X)


# In[95]:


from sklearn.metrics import mean_squared_error                      # step:4 fit the model
mse = mean_squared_error(Y,Y_pred)
print("mean squared error:", mse.round(3))
print("Root mean squared error:", np.sqrt(mse).round(3))


# In[23]:


X = df_3[['Age','KM','HP','CC','Doors','Gears','QT','Weight']]


# Using stats model

# In[24]:


model = smf.ols('Price ~ Age+KM+HP+CC+Doors+Gears+QT+Weight', data = df_3).fit()


# In[25]:


model.summary()


# In[26]:


# Predicted values
model.fittedvalues


# In[27]:


model.resid


# In[28]:


mse1 = np.mean(model.resid ** 2)
print('mean square error', mse1)
print("Root mean squared error:", np.sqrt(mse1).round(3))


# In[29]:


# MODEL VALIDATION TECHNIQUES
# COLLINEARITY CHECK
rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=df_3).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=df_3).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=df_3).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=df_3).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=df_3).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=df_3).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=df_3).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=df_3).fit().rsquared
vif_WT=1/(1-rsq_WT)


# In[30]:


# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[31]:


# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)
sm.qqplot(model.resid,line='q') # 'q' - A line is fit through the quartiles # line = '45'- to draw the 45-degree diagonal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[36]:


# MODEL DELETION DIAGNOSTIC AND FINAL MODEL
#1. Cook's Distance
# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[37]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(df_3)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[38]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[40]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
from statsmodels.graphics.regressionplots import influence_plot
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[41]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=df_3.shape[1]
n=df_3.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[42]:


df_3[df_3.index.isin([80])] 


# In[43]:


# IMPROVING THE MODEL
# Creating a copy of data so that original dataset is not affected
toyo_new=df_3.copy()
toyo_new


# In[44]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
df_4=toyo_new.drop(toyo_new.index[[80]],axis=0).reset_index(drop=True)
df_4


# In[47]:


# Model Deletion Diagnostics and Final Model
while np.max(c)>0.5 :
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=df_4).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    df_4=df_4.drop(df_4.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    df_4
else:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=df_4).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)


# In[48]:


if np.max(c)>0.5:
    model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=df_4).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    df_4=df_4.drop(toyo5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    df_4 
elif np.max(c)<0.5:
    final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=df_4).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)


# In[49]:


final_model.rsquared

