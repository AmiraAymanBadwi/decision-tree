#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[34]:


deabets_data = pd.read_csv(r'C:\Users\Eng.Amira\Desktop\data\deabets.csv')


# In[35]:


deabets_data.head()


# In[36]:


deabets_data.tail()


# In[6]:


deabets_data.shape


# In[7]:


deabets_data.info


# In[8]:


deabets_data.isnull().sum()


# In[9]:


deabets_data.duplicated().sum()


# In[10]:


deabets_data.drop_duplicates(inplace = True)


# In[11]:


deabets_data.duplicated().sum()


# In[12]:


deabets_data.describe()


# In[13]:


#using histogram to understand dataset data better
deabets_data.hist(figsize=(20,15));


# In[14]:


deabets_data['Diabetes_binary'].value_counts()


# In[15]:


X = deabets_data.drop(columns='Diabetes_binary', axis=1)
Y = deabets_data['Diabetes_binary']


# In[16]:


print(X)


# In[17]:


print(Y)


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[19]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)


# In[20]:


dt = DecisionTreeClassifier( max_depth= 12)
dt.fit(X_train , Y_train)


# In[21]:


# make predictions on test set
y_pred=dt.predict(X_test)

print('Training set score: {:.4f}'.format(dt.score(X_train, Y_train)))

print('Test set score: {:.4f}'.format(dt.score(X_test, Y_test)))


# In[27]:


#check MSE & RMSE 
mse =mean_squared_error(Y_test, y_pred)
print('Mean Squared Error : '+ str(mse))
rmse = math.sqrt(mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error : '+ str(rmse))


# In[30]:


matrix = classification_report(Y_test,y_pred )
print(matrix)


# In[ ]:




