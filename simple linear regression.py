#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[9]:


df = pd.read_csv("C:\\data science\\delivery_time.csv")


# In[10]:


df


# In[11]:


df.describe()


# In[13]:


df.isnull().sum()


# In[16]:


dataset=df.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset


# In[24]:


# applying log transformations to 'Delivery time'
dataset['delivery_time_log'] = np.log1p(dataset['delivery_time'])


# In[25]:


dataset.describe()


# In[26]:


dataset.isnull().sum()


# In[27]:


# Visualize the relationship between Sorting_time and Delivery_time
import matplotlib.pyplot as plt
sns.scatterplot(x='delivery_time_log', y='sorting_time', data=dataset)
plt.title('Scatter plot of Log(Delivery_time)  vs Sorting_time')
plt.show()


# In[29]:


# Split the data into training and testing sets
X = dataset[['delivery_time_log']]  
y = dataset['sorting_time']  


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


# Build the linear regression model
from sklearn.linear_model import LinearRegression
LE = LinearRegression()
LE.fit(X_train, y_train)


# In[32]:


y_pred = LE.predict(X_test)


# In[33]:


# Evaluate the model
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[36]:


model=smf.ols("delivery_time~sorting_time",data=dataset).fit()


# In[37]:


# MODEL BUILDING
# Finding Coefficient parameters
model.params


# In[38]:


# Finding tvalues and pvalues
model.tvalues , model.pvalues


# In[39]:


# Finding Rsquared Values
model.rsquared , model.rsquared_adj


# # Model Predictions

# In[40]:


delivery_time = (6.582734) + (1.649020)*(5)


# In[41]:


delivery_time


# # Question - 2

# In[42]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[43]:


df = pd.read_csv("C:\\data science\\Salary_Data.csv")


# In[44]:


df


# In[46]:


df.head()


# In[47]:


df.shape


# In[48]:


df.describe()


# In[49]:


# Pairplot for initial visualization
import matplotlib.pyplot as plt
sns.pairplot(df)
plt.show()


# In[50]:


# Correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[51]:


# Data Transformation
X = df[['YearsExperience']]


# In[54]:


# Apply log transformation to the dependent variable
y = np.log1p(df['Salary'])


# In[55]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[56]:


# Initialize the model
from sklearn.linear_model import LinearRegression
LE = LinearRegression()


# In[57]:


# Train the model
LE.fit(X_train, y_train)


# In[58]:


# Make predictions
y_pred = LE.predict(X_test)


# In[59]:


# Print model coefficients and intercept
print('Coefficients:', LE.coef_)
print('Intercept:', LE.intercept_)


# In[65]:


# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[66]:


print('Mean Squared Error:', mse)
print('R-squared:', r2)


# In[ ]:




