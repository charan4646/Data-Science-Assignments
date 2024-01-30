#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\logistic\\bank-full.csv", delimiter=";",quotechar='"')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.hist()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize= (9,6))
sns.heatmap(data.corr(), annot=True)
plt.xticks(rotation=45)
plt.show()


# In[15]:


# lets find how many discrete and continuous feature are there in dataset bybsepeartin them in variables
discrete_feature = [feature for feature in data.columns if len(data[feature].unique())<20 and feature]
print('Discrete Variables Count: {}'.format(len(discrete_feature)))


# In[16]:


continuous_feature = [feature for feature in data.columns if data[feature].dtype!='O' and feature not in discrete_feature]
print('Continuous Feature Count {}'.format(len(continuous_feature)))


# In[17]:


# LOG transformation
for feature in continuous_feature:
    data_1 = data.copy()
    if 0 in data_1[feature].unique():
        pass
    else:
        data_1[feature] = np.log(data_1[feature])
        data_1[feature].hist(bins=15)
        plt.ylabel('Count')
        plt.title(feature)
        plt.show()


# In[18]:


# Outliers detection
outlier = data.copy() 
fig, axes = plt.subplots(7,1,figsize=(10,8), sharex=False, sharey=False)
sns.boxplot(x='age',data=outlier,palette='crest',ax=axes[0])
sns.boxplot(x='balance',data=outlier,palette='crest',ax=axes[1])
sns.boxplot(x='day',data=outlier,palette='crest',ax=axes[2])
sns.boxplot(x='duration',data=outlier,palette='crest',ax=axes[3])
sns.boxplot(x='campaign',data=outlier,palette='crest',ax=axes[4])
sns.boxplot(x='pdays',data=outlier,palette='crest',ax=axes[5])
sns.boxplot(x='previous',data=outlier,palette='crest',ax=axes[6])
plt.tight_layout(pad=2.0)


# In[19]:


# After log-transformation
for feature in continuous_feature:
    data_2 = data.copy()
    data_2[feature] = np.log(data_2[feature])
    data_2.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()


# In[103]:


#Data preprocessing
data[['job','marital','education','default','housing','loan','contact','poutcome','month','y']] = data[
    ['job','marital','education','default','housing','loan','contact','poutcome','month','y']].astype('category')


# In[104]:


data_new = data


# In[105]:


data_new


# In[106]:


data.info()


# In[107]:


#label encoding
data_new['month'] = data_new['month'].cat.codes
data_new['job'] = data_new['job'].cat.codes
data_new['marital'] = data_new['marital'].cat.codes
data_new['education'] = data_new['education'].cat.codes
data_new['default'] = data_new['default'].cat.codes
data_new['housing'] = data_new['housing'].cat.codes
data_new['loan'] = data_new['loan'].cat.codes
data_new['contact'] = data_new['contact'].cat.codes
data_new['poutcome'] = data_new['poutcome'].cat.codes
data_new['y'] = data_new['y'].cat.codes


# In[108]:


# MOdel building
x1 = data_new.drop('y', axis=1)
y1 = data_new[['y']]


# In[109]:


x1


# In[110]:


y1


# In[137]:


x_train, x_test, y_train, y_test = train_test_split(x1,y1,test_size=0.20,random_state=12)
print("Shape of X_train : ",x_train.shape)
print("Shape of X_test  : ",x_test.shape)
print("Shape of y_train : ",y_train.shape)
print("Shape of y_test  : ",y_test.shape)


# In[138]:


logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)


# In[139]:


logistic_model.coef_


# In[140]:


logistic_model.intercept_


# In[141]:


# minMax Scaler
scalar = MinMaxScaler(feature_range= (0,1))
scalar.fit(data_new)
scaled_x = scalar.transform(data_new)


# In[142]:


scaled_x


# In[143]:


classifier1 = LogisticRegression()
classifier1.fit(scaled_x,y1)


# In[144]:


classifier1.coef_


# In[145]:


proba1 = classifier1.predict_proba(scaled_x)
proba1


# In[146]:


y_pred1 = classifier1.predict(scaled_x)
y_pred1


# In[147]:


# Model testing
# Train Data
y_pred_train1 = logistic_model.predict(x_train)


# In[148]:


print(confusion_matrix(y_train, y_pred_train1))


# In[149]:


print(classification_report(y_train,y_pred_train1))


# In[150]:


accuracy_score(y_train,y_pred_train1)


# In[151]:


fpr, tpr, thresholds = roc_curve(y_train,logistic_model.predict_proba (x_train)[:,1])

auc = roc_auc_score(y_train,logistic_model.predict_proba (x_train)[:,1])
print('AUC score : {:.2f}%'.format(auc*100))

plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()


# In[152]:


classification_report1 = classification_report(y_train,y_pred_train1)
print(classification_report1)


# In[153]:


# Test Data
y_pred_test1 = logistic_model.predict(x_test)


# In[154]:


print(confusion_matrix(y_test,y_pred_test1))


# In[155]:


print(classification_report(y_test,y_pred_test1))


# In[156]:


accuracy_score(y_test,y_pred_test1)


# In[157]:


fpr, tpr, thresholds = roc_curve(y_test,logistic_model.predict_proba (x_test)[:,1])

auc = roc_auc_score(y_test,logistic_model.predict_proba (x_test)[:,1])
print('AUC score : {:.2f}%'.format(auc*100))

plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()


# In[158]:


classification_report2 = classification_report(y_test,y_pred_test1)
print(classification_report2)


# In[159]:


# Compare train set and test set accuracy
print('Training set score : {:.2f}%'.format(logistic_model.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(logistic_model.score(x_test, y_test)*100))


# In[ ]:




