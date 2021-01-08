#!/usr/bin/env python
# coding: utf-8

# ## Support Vector Machine
# ### The Data
# The [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set) is being used here. 

# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
iris = sns.load_dataset('iris')


# ### Exploratory Data Analysis

# Pairplot of the data set.

# In[3]:


sns.pairplot(iris,hue='species',palette='Dark2')


# kde plot of sepal_length versus sepal width for setosa species of flower

# In[4]:


setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)


# ### Train Test Split

# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# #### Training the Model

# In[7]:


from sklearn.svm import SVC


# In[8]:


svc_model = SVC()


# In[9]:


svc_model.fit(X_train,y_train)


# #### Model Evaluation

# In[10]:


predictions = svc_model.predict(X_test)


# In[11]:


from sklearn.metrics import classification_report,confusion_matrix


# In[12]:


print(confusion_matrix(y_test,predictions))


# In[13]:


print(classification_report(y_test,predictions))


# #### Gridsearch Practice

# In[14]:


from sklearn.model_selection import GridSearchCV


# Create a dictionary called param_grid and fill out some parameters for C and gamma.

# In[15]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 


# Create a GridSearchCV object and fit it to the training data

# In[16]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them

# In[17]:


grid_predictions = grid.predict(X_test)


# In[18]:


print(confusion_matrix(y_test,grid_predictions))


# In[19]:


print(classification_report(y_test,grid_predictions))


# The predictions are exactly the same as there is just one point that is too noisey to grab. Hence, it is better not to overfit model that would be able to grab that.
