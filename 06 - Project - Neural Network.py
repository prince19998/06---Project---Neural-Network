#!/usr/bin/env python
# coding: utf-8

# # Project: Neural Network
# - Diabetes Classification
# - Given a dataset of various metrics can we predict if a patient has diabetes

# ### Step 1: Import libraries

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 2: Read the data
# - Use Pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method to read **files/diabetes.csv**

# In[3]:


data = pd.read_csv('./files/diabetes.csv')
data.head()


# In[ ]:





# ### Step 3: Check for data quality
# - Check **.isna().sum()**
# - Check **.dtypes**

# In[4]:


data.isna().sum()


# In[5]:


data.dtypes


# ### Step 4: Create dataset
# - Assign **X** to all but the last column
# - Assign **y** to the last column

# In[6]:


X = data.iloc[:,:-1]
y = data.iloc[:, -1]


# In[ ]:





# ### Step 5: Create training and test set
# - Use **train_test_split** to create **X_train, X_test, y_train, y_test**.

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:





# ### Step 6: Calculate average accuracy for 10 runs
# - Create an empty list and assign it to **accuracies**
# - Loop over **i** over 10 integers.
#     - Set the random seed: **tf.random.set_seed(i)**
#     - Create a **Sequential** model
#     - Add a **Dense** layer with one exit node and **input_dim=8**, and **activation='sigmoid'**
#     - Compile the model with **loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']**
#     - Fit the model with **epochs=1500, batch_size=100, verbose=0**
#     - Calculate the accuracy with **evaluate** on **X_test** and **y_test**
#         - The second return variable is the accuracy
#     - Append the accuracy (possibly multiplied by 100) to **accuraries**
# - Calculate the average value

# In[8]:


accuracies = []

for i in range(10):
  tf.random.set_seed(i)
  model = Sequential()
  model.add(Dense(1, input_dim=8, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=1500, batch_size=100, verbose=0)
  _, accuracy = model.evaluate(X_test, y_test)
  accuracies.append(accuracy*100)


# In[9]:


sum(accuracies) / len(accuracies)


# ### Step 7: Predict values
# - Predict all values with model on **X**
# - Make it into class ids with **np.where(y_pred < 0.5, 0, 1)** *(assuming **y_pred** is the predictions)*

# In[10]:


y_pred = model.predict(X)
y_pred = np.where(y_pred < .5, 0, 1)


# In[ ]:





# ### Step 8 (Optional): Visualize correct vs incorrect predictions
# - Calculate the ones that differ
#     - **np.abs(y.to_numpy() - y_pred.T)** *(assuming the variables names are correct)*
#     - Incorrect predictions will be 1, correct will be 0
# - Make a scatter plot with the two variables and the correctness calculations as colors

# In[11]:


differ = np.abs(y.to_numpy() - y_pred.T)

fig, ax = plt.subplots()
ax.scatter(x=X['Body mass index'], y=X['Age'], c = differ, alpha = .35)


# In[ ]:




