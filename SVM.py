#!/usr/bin/env python
# coding: utf-8

# # QUESTION - 1(forestfires.csv)

# In[ ]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\support vector machine\\forestfires.csv")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


# Label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[6]:


Y = df['size_category']


# In[7]:


Y = LE.fit_transform(Y)


# In[8]:


Y


# In[9]:


df['month'] = LE.fit_transform(df['month'])
df['day'] = LE.fit_transform(df['day'])


# In[10]:


df


# In[11]:


X = df.iloc[:,0:30]


# In[12]:


X


# In[13]:


Y


# In[14]:


# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)


# In[15]:


pd.DataFrame(SS_X)


# In[16]:


#STEP:4 DATA PARTITION
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size= 0.75)


# In[51]:


pip install mlxtend


# # linear

# In[65]:


# SVM
from sklearn.svm import SVC
svc = SVC(C=1.0,kernel="linear")


# In[66]:


svc.fit(X_train,Y_train)


# In[67]:


Y_pred_train = svc.predict(X_train) 
Y_pred_test = svc.predict(X_test) 


# In[68]:


# metrics
from sklearn.metrics import accuracy_score
ac1= accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:", ac1.round(3))
ac2= accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:", ac2.round(3))


# In[88]:


# Fit PCA to the data and transform it to 2D
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(X.values)
svc.fit(x_pca,Y)


# In[78]:


#Data visualiaztion
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X=x_pca, 
                      y=Y,
                      clf=svc, 
                      legend=4)


# # Polynomial

# In[84]:


# SVM
from sklearn.svm import SVC
svc = SVC(kernel='poly',degree=5)


# In[85]:


svc.fit(X_train,Y_train)


# In[86]:


Y_pred_train = svc.predict(X_train) 
Y_pred_test = svc.predict(X_test) 


# In[87]:


# metrics
from sklearn.metrics import accuracy_score
ac1= accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:", ac1.round(3))
ac2= accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:", ac2.round(3))


# In[ ]:


# Fit PCA to the data and transform it to 2D
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(X.values)
svc.fit(x_pca,Y)


# In[89]:


#Data visualiaztion
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X=x_pca, 
                      y=Y,
                      clf=svc, 
                      legend=4)


# # QUESTION - 2(salary.csv file)

# In[2]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


# In[3]:


Train = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\support vector machine\\SalaryData_Train(1).csv")


# In[4]:


Train


# In[5]:


Test = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\support vector machine\\SalaryData_Test(1).csv")


# In[6]:


Test


# In[7]:


Train.info()


# In[8]:


Train.describe()


# In[9]:


Test.info()


# In[10]:


Test.describe()


# In[11]:


Train[Train.duplicated()].shape


# In[12]:


Train[Train.duplicated()]


# In[13]:


Train =Train.drop_duplicates()


# In[14]:


Train


# In[15]:


Train.isnull().sum().sum()


# In[16]:


Test[Test.duplicated()].shape


# In[17]:


Test[Test.duplicated()]


# In[18]:


Test=Test.drop_duplicates()


# In[19]:


Test


# In[20]:


Test.isnull().sum().sum()


# In[21]:


Train['Salary'].value_counts()


# In[22]:


Test['Salary'].value_counts()


# In[25]:


pd.crosstab(Train['occupation'],Train['Salary'])


# In[26]:


pd.crosstab(Train['workclass'],Train['Salary'])


# In[27]:


pd.crosstab(Train['workclass'],Train['occupation'])


# In[28]:


# Visualize data
sns.countplot(x='Salary', data=Train)
plt.xlabel('Salary')
plt.ylabel('Count')
plt.show()


# In[29]:


sns.countplot(x='Salary',data= Test)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()


# In[30]:


Test['Salary'].value_counts()


# In[32]:


# scatter matrix to observe relationship between every colomn attribute. 
pd.plotting.scatter_matrix(Train,
                                       figsize= [20,20],
                                       diagonal='hist',
                                       alpha=1,
                                       s = 300,
                                       marker = '.',
                                       edgecolor= "black")


# In[33]:


string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


# In[34]:


##Preprocessing  categorical variables
number = LabelEncoder()
for i in string_columns:
        Train[i]= number.fit_transform(Train[i])
        Test[i]=number.fit_transform(Test[i])


# In[35]:


Train


# In[36]:


Test


# In[38]:


#column names
colnames = Train.columns
colnames


# In[39]:


#Train test split
x_train = Train[colnames[0:13]]
y_train = Train[colnames[13]]


# In[40]:


x_test = Test[colnames[0:13]]


# In[41]:


y_test = Test[colnames[13]]


# In[42]:


##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[43]:


x_train = norm_func(x_train)


# In[44]:


x_test =  norm_func(x_test)


# In[45]:


#SVM model
model_linear = SVC(kernel = "linear",random_state=40,gamma=0.1,C=1.0)
model_linear.fit(x_train,y_train)


# In[46]:


SVC(gamma=0.1, kernel='linear', random_state=40)


# In[47]:


pred_test_linear = model_linear.predict(x_test)


# In[48]:


np.mean(pred_test_linear==y_test) 


# In[49]:


# Polynomial
model_poly = SVC(kernel = "poly",random_state=40,gamma=0.1,C=1.0)
model_poly.fit(x_train,y_train)

pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test)


# In[50]:


#  RBF
model_rbf = SVC(kernel = "rbf",random_state=40,gamma=0.1,C=1.0)
model_rbf.fit(x_train,y_train)

pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test)


# In[ ]:




