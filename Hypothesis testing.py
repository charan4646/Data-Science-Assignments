#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# QUESTION-01
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import pandas as pd


# In[27]:


# Load the dataset
data=pd.read_csv('C:\\data science\\Cutlets.csv')


# In[28]:


data


# In[29]:


data.head()


# In[21]:


unitA=pd.Series(data.iloc[:,0])
unitA


# In[22]:


unitB=pd.Series(data.iloc[:,1])
unitB


# In[30]:


data.describe()


# In[31]:


data[data.duplicated()].shape


# In[32]:


data[data.duplicated()]


# In[33]:


data.info()


# In[42]:


# Plotting the data
plt.subplots(figsize = (9,6))
plt.subplot(121)
plt.boxplot(data['Unit A'])
plt.title('Unit A')
plt.subplot(122)
plt.boxplot(data['Unit B'])
plt.title('Unit B')
plt.show()


# In[36]:


statistic , p_value = stats.ttest_ind(data['Unit A'],data['Unit B'], alternative = 'two-sided')
print('p_value=',p_value)


# In[38]:


alpha = 0.05


# In[43]:


print('Significnace=%.5f, p=%.5f' % (alpha, p_value))


# In[44]:


if p_value<alpha:
    print("ho is rejected and h1 is accepted")
if p_value>alpha:
    print("h1 is rejected and h0 is accepted")


# # QUESTION - 2

# In[12]:


import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv('C:\\data science\\LabTAT (1).csv')


# In[5]:


data


# In[6]:


data.describe()


# In[7]:


#checking for null values
data[data.duplicated()].shape


# In[8]:


data[data.duplicated()]


# In[9]:


data.info()


# In[13]:


# Plotting the data using boxplot
plt.subplots(figsize = (16,9))
plt.subplot(221)
plt.boxplot(data['Laboratory 1'])
plt.title('Laboratory 1')
plt.subplot(222)
plt.boxplot(data['Laboratory 2'])
plt.title('Laboratory 2')
plt.subplot(223)
plt.boxplot(data['Laboratory 3'])
plt.title('Laboratory 3')
plt.subplot(224)
plt.boxplot(data['Laboratory 4'])
plt.title('Laboratory 4')
plt.show()


# In[14]:


#compare evidences with hypothesis using t-statistics
test_statistic , p_value = stats.f_oneway(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,3])
print('p_value =',p_value)


# In[15]:


alpha = 0.05


# In[16]:


print('Significnace=%.5f, p=%.5f' % (alpha, p_value))


# In[17]:


if p_value<alpha:
    print("ho is rejected and h1 is accepted")
if p_value>alpha:
    print("h1 is rejected and h0 is accepted")


# # QUESTION - 3

# In[18]:


import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import pandas as pd


# In[19]:


data = pd.read_csv("C:\\data science\\BuyerRatio (1).csv")


# In[20]:


data


# In[30]:


data_new = data.iloc[:,1:]


# In[31]:


# Applying chi-square contingency table to convert observed value into expected value
stat, p, dof, exp = stats.chi2_contingency(data_new) 
print(stat,"\n", p,"\n", dof,"\n", exp)


# In[35]:


stats.chi2_contingency(data_new) 


# In[39]:


observed = np.array([50, 142, 131, 70, 435, 1523, 1356, 750])
expected = np.array([  42.76531299 , 146.81287862 , 131.11756787 ,  72.30424052,
  442.23468701 , 1518.18712138 , 1355.88243213 , 747.69575948])


# In[40]:


# Comparing evidence with hypothesis
statistics, p_value = stats.chisquare(observed, expected, ddof = 3)
print("Statistics = ",statistics,"\n",'P_Value = ', p_value)


# In[42]:


print('Significnace=%.5f, p=%.5f' % (alpha, p_value))


# In[43]:


alpha = 0.05


# In[44]:


if p_value<alpha:
    print("ho is rejected and h1 is accepted")
if p_value>alpha:
    print("h1 is rejected and h0 is accepted")


# # QUESTION - 4

# In[45]:


import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import pandas as pd


# In[46]:


data = pd.read_csv("C:\\data science\\Costomer+OrderForm (1).csv")


# In[47]:


data


# In[48]:


data.describe()


# In[49]:


# checking for null values
data.isnull().sum()


# In[52]:


data[data.isnull().any(axis=1)]


# In[53]:


data.info()


# In[54]:


print(data['Phillippines'].value_counts(),'\n',data['Indonesia'].value_counts(),'\n',data['Malta'].value_counts(), '\n',data['India'].value_counts(), '\n')


# In[55]:


# Creating contigency table
contingency_table = [[271,267,269,280],[29,33,31,20]]
print(contingency_table)


# In[56]:


# Calculating expected values for observed data
stat, p, df, exp = stats.chi2_contingency(contingency_table)
print("Statistics = ",stat,"\n",'P_Value = ', p,'\n', 'degree of freedom =', df,'\n', 'Expected Values = ', exp)


# In[57]:


#Defining expected values and observed values
observed = np.array([271, 267, 269, 280, 29, 33, 31, 20])
expected = np.array([271.75, 271.75, 271.75, 271.75, 28.25, 28.25, 28.25, 28.25])


# In[64]:


# Compare evidence with hypothesis using t-statistic
test_statistic , p_value = stats.chisquare(observed, expected, ddof = 3)
print("Test Statistic = ",test_statistic,'\n', 'p_value =',p_value)


# In[74]:


alpha = 0.05


# In[76]:


print('Significnace=%.3f, p=%.3f' % (alpha, p_value))


# In[75]:


if p_value<alpha:
    print("ho is rejected and h1 is accepted")
if p_value>alpha:
    print("h1 is rejected and h0 is accepted")


# In[ ]:




