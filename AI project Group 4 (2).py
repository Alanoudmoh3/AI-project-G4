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

# In[3]:


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


# In[4]:


# retrive number of rows and columns
data_df = pd.read_csv('heart.csv' , index_col=0)
data_df


# In[5]:


# first 5 rows
data_df.head()


# In[6]:


# Get the number of rows and columns
num_rows = data_df.shape[0]
num_cols = data_df.shape[1]

# Print the results
print('Number of rows:', num_rows)
print('Number of columns:', num_cols)


# In[7]:


# display the variables type for each column
data_df.dtypes


# In[8]:


#Count the number of variables
num_variables = len(data_df.columns)

#print number of variables
print("Number of variables:" , num_variables )


# In[9]:


# display the statistical summaries
data_df.describe()


# In[10]:


# Remove non-numeric columns
numeric_df = data_df.select_dtypes(include=['float64', 'int64'])

# Display the variance of each column
print(numeric_df.var())


# In[11]:


# Get the labels from the dataset
labels = data_df

# Display the labels
print(labels)


# In[12]:


# Get the first 5 rows of raw data
raw_data = data_df.head()

# Display the raw data
print(raw_data)


# In[13]:


data_df.sample(3)


# In[58]:


# Create a crosstab of 'ChestPainType' by 'Sex'
cp_sex_ct = pd.crosstab(data_df['ChestPainType'], data_df['Sex'])

# Print the crosstab
print(cp_sex_ct)


# In[67]:


# Create a box plot of 'Cholesterol' by 'Sex'
data_df.boxplot(column='Cholesterol', by='Sex')
plt.xlabel('Sex')
plt.ylabel('Cholesterol')
plt.title('Cholesterol by Sex')
plt.suptitle('')
plt.show()


# In[75]:


# Create a box plot of 'MaxHR' by 'HeartDisease'
data_df.boxplot(column='MaxHR', by='HeartDisease')
plt.xlabel('HeartDisease')
plt.ylabel('MaxHR')
plt.title('MaxHR by HeartDisease')
plt.suptitle('')
plt.show()


# In[76]:


# Create a box plot of 'MaxHR' by 'Sex'
data_df.boxplot(column='MaxHR', by='Sex')
plt.xlabel('Sex')
plt.ylabel('MaxHR')
plt.title('MaxHR by Sex')
plt.suptitle('')
plt.show()


# In[20]:


# Compute summary statistics for each variable
var_summary = data_df.describe()

# Display the summary statistics as a table
print(var_summary)


# In[21]:


#Compute the number of patients with and without HeartDisease
diabetes_counts = data_df['HeartDisease'].value_counts()

# Create a bar chart of the diabetes data
plt.bar(diabetes_counts.index, diabetes_counts.values)
plt.xlabel('HeartDisease')
plt.ylabel('Number of patients')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()


# In[22]:


# Create a histogram of the MaxHR data
plt.hist(data_df['MaxHR'], bins=20)
plt.xlabel('MaxHR')
plt.ylabel('Frequency')
plt.show()


# In[23]:


# Create a histogram of the Cholesterol data
plt.hist(data_df['Cholesterol'], bins=20)
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()


# In[24]:


# Create a histogram of the RestingBP data
plt.hist(data_df['RestingBP'], bins=20)
plt.xlabel('RestingBP')
plt.ylabel('Frequency')
plt.show()


# In[25]:


# Create a histogram of the FastingBS data
plt.hist(data_df['FastingBS'], bins=20)
plt.xlabel('FastingBS')
plt.ylabel('Frequency')
plt.show()


# In[26]:


# Create a histogram of the Oldpeak data
plt.hist(data_df['Oldpeak'], bins=20)
plt.xlabel('Oldpeak')
plt.ylabel('Frequency')
plt.show()


# In[27]:


# Create a histogram of the HeartDisease data
plt.hist(data_df['HeartDisease'], bins=20)
plt.xlabel('HeartDisease')
plt.ylabel('Frequency')
plt.show()


# In[63]:


# Create a scatter plot of 'Sex' vs 'MaxHR'
plt.scatter(data_df['Sex'], data_df['MaxHR'])
plt.xlabel('Sex')
plt.ylabel('Max Heart Rate')
plt.title('Sex vs Max Heart Rate')
plt.show()


# In[64]:


# Create a scatter plot of MaxHR vs. Cholesterol
plt.scatter(data_df['MaxHR'], data_df['Cholesterol'])
plt.xlabel('MaxHR')
plt.ylabel('Cholesterol')
plt.show()


# In[66]:


# Create a scatter plot of Cholesterol vs. ChestPainType
plt.scatter(data_df['Cholesterol'], data_df['ChestPainType'])
plt.xlabel('Cholesterol')
plt.ylabel('ChestPainType')
plt.show()


# In[29]:


# Create a scatter plot of HeartDisease vs. Oldpeak
plt.scatter(data_df['HeartDisease'], data_df['Oldpeak'])
plt.xlabel('HeartDisease')
plt.ylabel('Oldpeak')
plt.show()


# In[39]:


# Count the number of occurrences of each value in the 'ChestPainType' column
cp_counts = data_df['ChestPainType'].value_counts()

# Define a list of colors for the bars
colors = ['red', 'blue', 'green', 'orange']

# Create a bar plot of the 'ChestPainType' counts with different colors for each bar
plt.bar(cp_counts.index, cp_counts.values, color=colors)
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.title('Distribution of Chest Pain Type')
plt.show()


# In[42]:


# Count the number of occurrences of each value in the 'RestingECG' column
cp_counts = data_df['RestingECG'].value_counts()

# Define a list of colors for the bars
colors = ['red', 'blue', 'green']

# Create a bar plot of the 'RestingECG' counts with different colors for each bar
plt.bar(cp_counts.index, cp_counts.values, color=colors)
plt.xlabel('RestingECG')
plt.ylabel('Count')
plt.title('Distribution of RestingECG')
plt.show()


# In[31]:


# Compute the number of male and female patients
HeartDisease_counts = data_df['HeartDisease'].value_counts()

# Create a pie chart of the sex data
plt.pie(HeartDisease_counts.values, labels=HeartDisease_counts.index, autopct='%1.1f%%')
plt.title('HeartDisease distribution')
plt.show()


# In[32]:


# Compute the number of male and female patients
sex_counts = data_df['Sex'].value_counts()

# Create a pie chart of the sex data
plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
plt.title('Sex distribution')
plt.show()


# In[33]:


# Display the number of missing values in each column
print(data_df.isnull().sum())


# In[34]:


# Apply square root transformation to the RestingBP variable
data_df['RestingBP_sqrt'] = np.sqrt(data_df['RestingBP'])

# Print the first 5 rows of the dataset with the transformed variable
print(data_df.head())


# In[35]:


# Print the column names in the DataFrame
print(data_df.columns)


# In[36]:


# Apply logarithmic transformation to the Cholesterol variable
data_df['Cholesterol_log'] = np.log(data_df['Cholesterol'])

# Print the first 5 rows of the dataset with the transformed variable
print(data_df.head())


# In[37]:


# Remove the 'ST_Slope' variables 
data_df = data_df.drop(['ST_Slope'], axis=1)

# Print the first 5 rows of the dataset with the removed variables
print(data_df.head())


# In[38]:


# Discretize the 'RestingBP' column into 3 bins
bp_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
bp_discretized = bp_discretizer.fit_transform(data_df[['RestingBP']])

# Replace the original 'RestingBP' column with the discretized values
data_df['RestingBP'] = bp_discretized

# Print the first 5 rows of the DataFrame to see the result
print(data_df.head())


# In[55]:


# Compute the mean of the 'RestingBP' column
rb_mean = data_df['RestingBP'].mean()
ch_mean = data_df['Cholesterol'].mean()
fa_mean = data_df['FastingBS'].mean()
mh_mean = data_df['MaxHR'].mean()

# Print the result
print("The average value of the 'RestingBP' column is:", rb_mean)
print("The average value of the 'Cholesterol' column is:", ch_mean)
print("The average value of the 'FastingBS' column is:", fa_mean)
print("The average value of the 'MaxHR' column is:", mh_mean)


# In[56]:


# Compute the standard deviation of the 'RestingBP' column
rb_std = data_df['RestingBP'].std()
ch_std = data_df['Cholesterol'].std()
fa_std = data_df['FastingBS'].std()
mh_std = data_df['MaxHR'].std()

# Print the result
print("The standard deviation of the 'RestingBP' column is:", rb_std)
print("The standard deviation of the 'Cholesterol' column is:", ch_std)
print("The standard deviation of the 'FastingBS' column is:", fa_std)
print("The standard deviation of the 'MaxHR' column is:", mh_std)


# In[ ]:




