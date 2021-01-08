#!/usr/bin/env python
# coding: utf-8

# ## Deep Learning using Tensorflow 
# 
# This analysis is done using the [Bank Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) from the UCI repository.
# 
# The data consists of 5 columns:
# 
# * variance of Wavelet Transformed image (continuous)
# * skewness of Wavelet Transformed image (continuous)
# * curtosis of Wavelet Transformed image (continuous)
# * entropy of image (continuous)
# * class (integer)
# 
# Where class indicates whether or not a Bank Note was authentic.
# 

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('bank_note_data.csv')


# In[3]:


data.head()


# ### Exploratory Data Analysis

# In[4]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Countplot of the Classes (Authentic 1 vs Fake 0) 

# In[5]:


sns.countplot(x='Class',data=data)


# PairPlot of the Data using Seaborn with Hue = Class 

# In[6]:


sns.pairplot(data,hue='Class')


# ### Data Preparation 
# #### Standard Scaling

# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


scaler = StandardScaler()


# Fit scaler to the features.

# In[9]:


scaler.fit(data.drop('Class',axis=1))


# Use the .transform() method to transform the features to a scaled version.

# In[10]:


scaled_features = scaler.fit_transform(data.drop('Class',axis=1))


# Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.

# In[11]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# ### Train Test Split

# In[12]:


X = df_feat


# In[13]:


y = data['Class']


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ### Tensorflow

# In[16]:


import tensorflow as tf


# Create a list of feature column objects using tf.feature.numeric_column() 

# In[17]:


df_feat.columns


# In[18]:


image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')


# In[19]:


feat_cols = [image_var,image_skew,image_curt,entropy]


# Create an object called classifier which is a DNNClassifier from learn. Set it to have 2 classes and a [10,20,10] hidden unit layer structure.

# In[20]:


classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)


# Now create a tf.estimator.pandas_input_fn that takes in your X_train, y_train with batch_size = 20 and set shuffle=True. 

# In[21]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)


# Train the classifier to the input function using steps=500.

# In[22]:


classifier.train(input_fn=input_func,steps=500)


# ## Model Evaluation

# Create another pandas_input_fn that takes in the X_test data for x. Set shuffle=False since there is no need to shuffle for predictions.

# In[23]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# Use the predict method from the classifier model to create predictions from X_test 

# In[24]:


note_predictions = list(classifier.predict(input_fn=pred_fn))


# In[25]:


note_predictions[0]


# In[26]:


final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])


# #### Creating a classification report and a Confusion Matrix. 

# In[27]:


from sklearn.metrics import classification_report,confusion_matrix


# In[28]:


print(confusion_matrix(y_test,final_preds))


# In[29]:


print(classification_report(y_test,final_preds))

