#!/usr/bin/env python
# coding: utf-8

# # QUESTION - 1(glass.csv file)

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\KNN\\glass.csv")


# In[3]:


df


# In[6]:


df['Type'].value_counts()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[12]:


df[df.duplicated()].shape


# In[13]:


df[df.duplicated()]


# In[14]:


data = df.drop_duplicates()


# In[15]:


data


# In[16]:


data.corr()


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
#pairwise plot of all the features
sns.pairplot(data,hue='Type')
plt.show()


# In[18]:


X = data.iloc[:,0:9]


# In[19]:


X


# In[20]:


array = X.values


# In[21]:


# Standardization
from sklearn.preprocessing import StandardScaler


# In[22]:


SS = StandardScaler().fit(array)
SS_X =SS.transform(array)


# In[23]:


data_new = pd.DataFrame(SS_X,columns=data.columns[:-1])


# In[24]:


data_new


# In[25]:


x = data_new


# In[28]:


y = data['Type']


# In[29]:


# Data partition
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.3,random_state=45)


# In[31]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
KNN = KNeighborsClassifier(n_neighbors=3)


# In[32]:


KNN.fit(x_train,y_train)


# In[33]:


#Predicting on test data
preds =KNN.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() 


# In[34]:


pd.crosstab(y_test,preds) 


# In[35]:


print("Accuracy", accuracy_score(y_test,preds)*100)
KNN.score(x_train,y_train)
print(classification_report(y_test,preds))


# In[36]:


from sklearn.model_selection import GridSearchCV
n_neighbors = np.array(range(1,15))
grid = dict(n_neighbors=n_neighbors)
KNN = KNeighborsClassifier()
grid = GridSearchCV(estimator=KNN, param_grid=grid)
grid.fit(x, y)


# In[37]:


print(grid.best_score_)
print(grid.best_params_)


# In[38]:


k_values = np.arange(1,25)


# In[39]:


train_accuracy = []
test_accuracy = []
for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))


# In[40]:


# Plot
plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.savefig('graph.png')
plt.show()


# In[41]:


print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# # QUESTION - 2(zoo.csv file)

# In[59]:


import pandas as pd


# In[60]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\KNN\\Zoo.csv")


# In[61]:


df


# In[62]:


df.info()


# In[63]:


df.describe()


# In[64]:


df['animal name'].value_counts()


# In[65]:


list(df)


# In[66]:


#check if there are duplicates in animal_name
duplicates = df['animal name'].value_counts()
duplicates[duplicates > 1]


# In[67]:


frog = df[df['animal name'] == 'frog']


# In[68]:


frog


# In[69]:


# observation: find that one frog is venomous and another one is not 
# change the venomous one into frog2 to seperate 2 kinds of frog=============== 
df['animal name'][(df['venomous'] == 1 )& (df['animal name'] == 'frog')] = "frog2"


# In[70]:


df['venomous'].value_counts()


# In[71]:


df.head(27)


# In[72]:


# finding Unique value of hair and plotting=====================================================
color_list = [("red" if i == 1 else "blue" if i == 0 else "yellow" )for i in df.hair]
unique_color = list(set(color_list))


# In[73]:


unique_color


# In[74]:


import seaborn as sns 
import matplotlib.pyplot as plt
sns.countplot(x="hair", data=df)
plt.xlabel("Hair")
plt.ylabel("Count")
plt.show()


# In[75]:


# Lets see how many animals provides us milk and plotting
df['milk'].value_counts()


# In[77]:


# Lets see species wise domestic and non-domestic animals
pd.crosstab(df['type'], df['milk']).plot(kind="bar", figsize=(10, 8), title="milk providing animals");
plt.plot();


# In[78]:


# lets find out all the aquatic animals and plotting======================================
pd.crosstab(df['type'], df['aquatic']).plot(kind="bar", figsize=(10, 8));


# In[79]:


X = df.iloc[:,1:16]
Y = df.iloc[:,16]


# In[80]:


# Data partition
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)


# In[81]:


# Label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Y = LE.fit_transform(Y)


# In[82]:


Y


# In[83]:


#K-FOLD
from sklearn.model_selection import KFold
num_folds = 10
kfold = KFold(n_splits=10)


# In[84]:


# Knn model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)


# In[85]:


#Predicting on test data
Y_preds = model.predict(X_test) 
Y_preds
pd.Series(Y_preds).value_counts()


# In[86]:


pd.crosstab(Y_test,Y_preds) 


# In[87]:


# Accuracy
import numpy as np
np.mean(Y_preds==Y_test)


# In[88]:


model.score(X_train,Y_train)


# In[89]:


# Accuracy score
from sklearn.metrics import accuracy_score
print("Accuracy", accuracy_score(Y_test,Y_preds)*100)


# In[90]:


# Cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean()*100)


# In[91]:


print(results.std()*100)


# In[92]:


# GRID SEARCH
from sklearn.model_selection import GridSearchCV
n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)
model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[93]:


print(grid.best_score_)
print(grid.best_params_)


# In[94]:


k_values = np.arange(1,25)
train_accuracy = []
test_accuracy = []
for i, k in enumerate(k_values):
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train,Y_train)
    train_accuracy.append(KNN.score(X_train,Y_train))
    test_accuracy.append(KNN.score(X_test,Y_test))


# In[95]:


plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.savefig('graph.png')
plt.show()


# In[96]:


print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# In[ ]:




