#!/usr/bin/env python
# coding: utf-8

# ## Decision Tree and Random Forest

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loans = pd.read_csv('loan_data.csv')


# In[3]:


loans.info()


# In[4]:


loans.describe()


# In[5]:


loans.head()


# ### Exploratory Data Analysis
# Histogram of two FICO distributions, one for each credit.policy outcome.

# In[6]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# Histogram of two FICO distributions, one for each not.fully.paid outcome.

# In[7]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# Countplot using seaborn showing the counts of loans by purpose, with hue = not.fully.paid

# In[8]:


plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# Visualize the trend between FICO score and interest rate

# In[9]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# lmplots to see if the trend differed between not.fully.paid and credit.policy

# In[10]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# In[12]:


loans.info()


# #### Categorical Features
# 
# The "purpose" column is categorical. Hence, transform them using dummy variables so sklearn will be able to understand them. 

# In[36]:


cat_feats = ['purpose']


# Using pd.get_dummies(loans,columns=cat_feats,drop_first=True), create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data

# In[37]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[38]:


final_data.info()


# ### Train Test Split

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# #### Training the Decision Tree Model

# In[22]:


from sklearn.tree import DecisionTreeClassifier


# Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data

# In[23]:


dtree = DecisionTreeClassifier()


# In[24]:


dtree.fit(X_train,y_train)


# #### Predicting and Evaluating the Decision Tree Model

# In[25]:


predictions = dtree.predict(X_test)


# In[26]:


from sklearn.metrics import classification_report,confusion_matrix


# In[27]:


print(classification_report(y_test,predictions))


# In[28]:


print(confusion_matrix(y_test,predictions))


# #### Training the Random Forest model
# Create an instance of the RandomForestClassifier class and fit it to the training data 

# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


rfc = RandomForestClassifier(n_estimators=600)


# In[31]:


rfc.fit(X_train,y_train)


# #### Predicting and Evaluating the Random Forest Model

# In[32]:


predictions = rfc.predict(X_test)


# In[33]:


from sklearn.metrics import classification_report,confusion_matrix


# In[34]:


print(classification_report(y_test,predictions))


# In[35]:


print(confusion_matrix(y_test,predictions))

