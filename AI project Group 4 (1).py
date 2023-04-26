#!/usr/bin/env python
# coding: utf-8

# The goal of collecting the Heart Failure Prediction dataset is to raise awareness about the risk factors of the heart failure. 
#  People with cardiovascular disease or who are at high cardiovascular risk
#  (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease)
#  need early detection and management where in a machine learning model can be of great help.
# 

#  Attribute Information:Age , Sex , ChestPainType , RestingBP , Cholesterol , FastingBS ,
#  RestingECG , MaxHR , ExerciseAngina , HeartDisease
#  Link: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download

# In[168]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, PowerTransformer, FunctionTransformer
import warnings
warnings.filterwarnings('ignore')


# In[169]:


# retrive number of rows and columns
data_df = pd.read_csv('heart.csv' , index_col=0)
data_df


# In[170]:


# first 5 rows
data_df.head()


# In[171]:


# Get the number of rows and columns
num_rows = data_df.shape[0]
num_cols = data_df.shape[1]

# Print the results
print('Number of rows:', num_rows)
print('Number of columns:', num_cols)


# In[172]:


# display the variables type for each column
data_df.dtypes


# In[173]:


#Count the number of variables
num_variables = len(data_df.columns)

#print number of variables
print("Number of variables:" , num_variables )


# In[174]:


# display the statistical summaries
data_df.describe()


# In[175]:


# Remove non-numeric columns
numeric_df = data_df.select_dtypes(include=['float64', 'int64'])

# Display the variance of each column
print(numeric_df.var())


# In[176]:


# Get the labels from the dataset
labels = data_df

# Display the labels
print(labels)


# In[177]:


# Get the first 5 rows of raw data
raw_data = data_df.head()

# Display the raw data
print(raw_data)


# In[178]:


data_df.sample(3)


# In[179]:


# Plot a box plot of the Heart Disease variable
sns.boxplot(x=data_df['HeartDisease'])


# In[180]:


# Plot a box plot of the RestingBP variable
sns.boxplot(x=data_df['RestingBP'])


# In[181]:


# Plot a box plot of the Cholesterol variable
sns.boxplot(x=data_df['Cholesterol'])


# In[182]:


# Plot a box plot of the FastingBS  variable
sns.boxplot(x=data_df['FastingBS'])


# In[183]:


# Plot a box plot of the MaxHR  variable
sns.boxplot(x=data_df['MaxHR'])


# In[184]:


# Plot a box plot of the  Oldpeak  variable
sns.boxplot(x=data_df['Oldpeak'])


# In[185]:


# Compute summary statistics for each variable
var_summary = data_df.describe()

# Display the summary statistics as a table
print(var_summary)


# In[186]:


#Compute the number of patients with and without HeartDisease
diabetes_counts = data_df['HeartDisease'].value_counts()

# Create a bar chart of the diabetes data
plt.bar(diabetes_counts.index, diabetes_counts.values)
plt.xlabel('HeartDisease')
plt.ylabel('Number of patients')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()


# In[187]:


# Create a histogram of the MaxHR data
plt.hist(data_df['MaxHR'], bins=20)
plt.xlabel('MaxHR')
plt.ylabel('Frequency')
plt.show()


# In[188]:


# Create a histogram of the Cholesterol data
plt.hist(data_df['Cholesterol'], bins=20)
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()


# In[189]:


# Create a histogram of the RestingBP data
plt.hist(data_df['RestingBP'], bins=20)
plt.xlabel('RestingBP')
plt.ylabel('Frequency')
plt.show()


# In[190]:


# Create a histogram of the FastingBS data
plt.hist(data_df['FastingBS'], bins=20)
plt.xlabel('FastingBS')
plt.ylabel('Frequency')
plt.show()


# In[191]:


# Create a histogram of the Oldpeak data
plt.hist(data_df['Oldpeak'], bins=20)
plt.xlabel('Oldpeak')
plt.ylabel('Frequency')
plt.show()


# In[192]:


# Create a histogram of the HeartDisease data
plt.hist(data_df['HeartDisease'], bins=20)
plt.xlabel('HeartDisease')
plt.ylabel('Frequency')
plt.show()


# In[193]:


# Create a scatter plot of FastingBS vs. Cholesterol
plt.scatter(data_df['FastingBS'], data_df['Cholesterol'])
plt.xlabel('FastingBS')
plt.ylabel('Cholesterol')
plt.show()


# In[194]:


# Create a scatter plot of HeartDisease vs. Oldpeak
plt.scatter(data_df['HeartDisease'], data_df['Oldpeak'])
plt.xlabel('HeartDisease')
plt.ylabel('Oldpeak')
plt.show()


# In[195]:


# Compute the number of male and female patients
HeartDisease_counts = data_df['HeartDisease'].value_counts()

# Create a pie chart of the sex data
plt.pie(HeartDisease_counts.values, labels=HeartDisease_counts.index, autopct='%1.1f%%')
plt.title('HeartDisease distribution')
plt.show()


# In[196]:


# Compute the number of male and female patients
sex_counts = data_df['Sex'].value_counts()

# Create a pie chart of the sex data
plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
plt.title('Sex distribution')
plt.show()


# In[197]:


# Display the number of missing values in each column
print(data_df.isnull().sum())


# In[198]:


# Apply square root transformation to the RestingBP variable
data_df['RestingBP_sqrt'] = np.sqrt(data_df['RestingBP'])

# Print the first 5 rows of the dataset with the transformed variable
print(data_df.head())


# In[199]:


# Print the column names in the DataFrame
print(data_df.columns)


# In[200]:


# Apply logarithmic transformation to the Cholesterol variable
data_df['Cholesterol_log'] = np.log(data_df['Cholesterol'])

# Print the first 5 rows of the dataset with the transformed variable
print(data_df.head())


# In[201]:


# Remove the 'ST_Slope' variables 
data_df = data_df.drop(['ST_Slope'], axis=1)

# Print the first 5 rows of the dataset with the removed variables
print(data_df.head())


# In[208]:


# Discretize the 'RestingBP' column into 3 bins
bp_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
bp_discretized = bp_discretizer.fit_transform(data_df[['RestingBP']])

# Replace the original 'RestingBP' column with the discretized values
data_df['RestingBP'] = bp_discretized

# Print the first 5 rows of the DataFrame to see the result
print(data_df.head())


# In[ ]:




