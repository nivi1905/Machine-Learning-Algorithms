#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


customers = pd.read_csv('Ecommerce Customers')


# In[4]:


customers.head()


# In[5]:


customers.describe()


# In[6]:


customers.info()


# ### Exploratory Data Analysis
# 
# Jointplot to compare the Time on Website and Yearly Amount Spent columns using seaborn. 

# In[11]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# Jointplot to compare the Time on App and Yearly Amount Spent columns using seaborn. 

# In[10]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# Jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership

# In[12]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',kind='hex',data=customers)


# Pairplot using seaborn to explore the type of relationship against the entire dataset

# In[13]:


sns.pairplot(customers)


# Linear model plot of  Yearly Amount Spent vs. Length of Membership.

# ### Training and Testing Data

#  Variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

# In[14]:


customers.columns


# In[15]:


X=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[16]:


y=customers['Yearly Amount Spent']


# Use model_selection.train_test_split from sklearn to split the data into training and testing sets

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# #### Training the Model

# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


lm = LinearRegression()


# In[21]:


lm.fit(X_train,y_train)


# In[22]:


print('Coefficients: \n', lm.coef_)


# #### Predicting Test Data

# In[23]:


predictions = lm.predict(X_test)


# Scatterplot of the real test values versus the predicted values.

# In[24]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# #### Evaluating the Model

# In[25]:


from sklearn import metrics


# In[26]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# #### Residuals
# Plot a histogram of the residuals and make sure it looks normally distributed

# In[27]:


sns.distplot((y_test-predictions),bins=50);

