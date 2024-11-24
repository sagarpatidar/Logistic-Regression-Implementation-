#!/usr/bin/env python
# coding: utf-8

# Logistic Regression Implementation 

# In[2]:


from sklearn.datasets import load_iris 


# In[3]:


dataset = load_iris()


# In[4]:


dataset


# In[5]:


print(dataset.DESCR)


# Independent Features are - 
# - sepal length in cm
#         - sepal width in cm
#         - petal length in cm
#         - petal width in cm

# In[6]:


import pandas as pd 
import numpy as np 


# In[7]:


dataset.keys()


# In[8]:


df=pd.DataFrame(dataset.data,columns=dataset.feature_names)


# In[9]:


df.head()


# # Independent and dependent features 

# In[10]:


x = df 
y = dataset.target 


# In[11]:


df['target'] = dataset.target 


# In[12]:


df.head()


# In[13]:


dataset.target 


# # binary classsification 

# In[15]:


df_copy = df[df['target']!=2]


# In[16]:


df_copy.head()


# # Independent and dependent features 

# In[17]:


X = df_copy.iloc[:,:-1]
y = df_copy.iloc[:,-1]


# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


#train test split 
from sklearn.model_selection import train_test_split


# In[20]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.33,random_state = 42)


# In[21]:


classifier = LogisticRegression()


# In[22]:


classifier.fit(X_train,y_train)
    


# In[23]:


classifier.predict_proba(X_test)


# classifier.predict_proba just comapre to the probablity of 0th class and 1st class 

# In[24]:


#Prediction
y_pred=classifier.predict(X_test)


# In[25]:


y_pred


# In[26]:


# confusion matrix,accuracy score , classification report 


# In[27]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[28]:


print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))
print (classification_report(y_pred,y_test))


# # Hyperparameter Tuning 

# Gridsearchcv

# using both gridsearchsv and crossvalidation to select best parameter in logistic Regression 

# In[29]:


from sklearn.model_selection import GridSearchCV
import warnings 


# In[30]:


warnings.filterwarnings('ignore')


# In[42]:


parameters = {'penalty':('11','12','elasticnet',None),'C':[1,10,20]}



# In[43]:


GridSearchCV(classifier,param_grid=parameters,cv=5)


# In[44]:


#Splitting of train data to validation data


# In[45]:


clf=GridSearchCV(classifier,param_grid=parameters,cv=5)


# In[46]:


clf.fit(X_train,y_train)


# In[50]:


classifier = LogisticRegression(C=1,penalty='l2')


# In[53]:


classifier.fit(X_train,y_train)


# In[58]:


#prediction 
y_pred=classifier.predict(X_test)
print (confusion_matrix(y_pred,y_test))
print (classification_report(y_pred,y_test))


# In[64]:


from sklearn.model_selection import RandomizedSearchCV


# In[69]:


random_clf =RandomizedSearchCV(LogisticRegression(),param_distributions=parameters,cv=5)


# In[70]:


random_clf.fit(X_train,y_train)


# In[71]:


random_clf.best_params_


# In[ ]:


# logistic regression 

