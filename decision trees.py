#!/usr/bin/env python
# coding: utf-8

# # QUESTION - 1(company.csv file)

# In[ ]:


import pandas as pd


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Decision\\Company_Data.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[9]:


#EDA
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data = df, hue = 'ShelveLoc')


# In[10]:


# LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[11]:


Y = df['ShelveLoc']


# In[12]:


Y = LE.fit_transform(Y)


# In[13]:


Y


# In[14]:


df['ShelveLoc'] = LE.fit_transform(df['ShelveLoc'])
df['Urban'] = LE.fit_transform(df['Urban'])
df['US'] = LE.fit_transform(df['US'])


# In[15]:


df


# In[16]:


df.info()


# In[17]:


X = df[["Sales","CompPrice","Income","Advertising","Population","Price","Age","Education","Urban","US"]]


# In[18]:


Y = df['ShelveLoc']


# In[39]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion ='gini',max_depth = 3)


# In[40]:


DT.fit(X,Y)


# In[41]:


Y_pred = DT.predict(X)


# In[42]:


from sklearn.metrics import accuracy_score
ac = accuracy_score(Y,Y_pred)
print(ac)


# In[43]:


from sklearn import tree


# In[44]:


tree.plot_tree(DT)


# In[45]:


from sklearn.model_selection import train_test_split
training_accuracy = []
test_accuracy = []
for i in range(1,1001):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    DT.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test  = DT.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))


# In[47]:


import numpy as np
print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))


# # BAGGING CLASSIFIER

# In[48]:


from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=100,max_samples=0.6,max_features=0.7)


# In[49]:


from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    bag.fit(X_train,Y_train)
    Y_pred_train = bag.predict(X_train)
    Y_pred_test  = bag.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))


# In[50]:


print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))
print("Variance",np.mean(training_accuracy).round(3)-np.mean(test_accuracy).round(3))


# # RANDOM FOREST

# In[51]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_samples=0.6,max_features=0.7)


# In[52]:


from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test  = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))


# In[53]:


print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))
print("Variance",np.mean(training_accuracy).round(3)-np.mean(test_accuracy).round(3))


# # QUESTION - 2(fraud check.csv file) 

# In[54]:


import pandas as pd


# In[56]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Decision\\Fraud_Check.csv")


# In[57]:


df


# In[59]:


df.info()


# In[62]:


df = df.rename({'Undergrad':'under_grad', 'Marital.Status':'marital_status', 'Taxable.Income':'taxable_income',
                    'City.Population':'city_population', 'Work.Experience':'work_experience', 'Urban':'urban'}, axis = 1)


# In[63]:


df


# In[64]:


df['taxable_income'] = df.taxable_income.map(lambda x: 1 if x <= 30000 else 0)


# In[65]:


df


# In[66]:


# Label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[67]:


df['under_grad'] = LE.fit_transform(df['under_grad'])
df['marital_status'] = LE.fit_transform(df['marital_status'])
df['urban'] = LE.fit_transform(df['urban'])


# In[68]:


df


# In[69]:


Y = df['taxable_income']


# In[70]:


X = df[["under_grad","marital_status","city_population","work_experience","urban"]]


# In[71]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion ='gini',max_depth = 6)


# In[72]:


DT.fit(X,Y)


# In[73]:


Y_pred = DT.predict(X)


# In[74]:


from sklearn.metrics import accuracy_score
ac = accuracy_score(Y,Y_pred)
print(ac)


# In[75]:


from sklearn import tree


# In[76]:


tree.plot_tree(DT)


# In[77]:


from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []

for i in range(1,1001):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    DT.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test  = DT.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))


# In[78]:


print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))


# # BAGGING CLASSIFIER

# In[79]:


from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=100,max_samples=0.6,max_features=0.7)


# In[80]:


from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    bag.fit(X_train,Y_train)
    Y_pred_train = bag.predict(X_train)
    Y_pred_test  = bag.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))


# In[81]:


print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))


# # RANDOM FOREST

# In[83]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_samples=0.6,max_features=0.7)


# In[84]:


from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test  = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))


# In[85]:


print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))


# In[ ]:




