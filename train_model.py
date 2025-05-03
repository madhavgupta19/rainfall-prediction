#!/usr/bin/env python
# coding: utf-8

# **Importing the dependencies**

# # New Section

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


# **Data Collection and Processing**

# In[5]:


# laod the dataset to a pandas dataframe
data = pd.read_csv("Rainfall.csv")


# In[6]:


print(type(data))


# In[7]:


data.shape


# In[8]:


data.head()


# In[9]:


data.tail()


# In[10]:


data["day"].unique()


# In[11]:


print("Data Info:")
data.info()


# In[12]:


data.columns


# In[13]:


# remove extra  spaces in all columns
data.columns = data.columns.str.strip()


# In[14]:


data.columns


# In[15]:


print("Data Info:")
data.info()


# In[16]:


data = data.drop(columns=["day"])


# In[17]:


data.head()


# In[18]:


# checking the number of missing values
print(data.isnull().sum())


# In[19]:


data["winddirection"].unique()


# In[20]:


# handle missing values
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())


# In[21]:


# checking the number of missing values
print(data.isnull().sum())


# In[22]:


data["rainfall"].unique()


# In[23]:


# converting the yes & no to 1 and 0 respectively
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})


# In[24]:


data.head()


# **Exploratory Data Analysis (EDA)**

# In[25]:


data.shape


# In[26]:


# setting plot style for all the plots
sns.set(style="whitegrid")


# In[27]:


data.describe()


# In[28]:


data.columns


# In[29]:


plt.figure(figsize=(15, 10))

for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
  plt.subplot(3, 3, i)
  sns.histplot(data[column], kde=True)
  plt.title(f"Distribution of {column}")

plt.tight_layout()
plt.show()


# In[30]:


plt.figure(figsize=(6, 4))
sns.countplot(x="rainfall", data=data)
plt.title("Distribution of Rainfall")
plt.show()


# In[31]:


# correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()


# In[32]:


plt.figure(figsize=(15, 10))

for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
  plt.subplot(3, 3, i)
  sns.boxplot(data[column])
  plt.title(f"Boxplot of {column}")

plt.tight_layout()
plt.show()


# **Data Preprocessing**

# In[33]:


# drop highly correlated column
data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])


# In[34]:


data.head()


# In[35]:


print(data["rainfall"].value_counts())


# In[36]:


# separate majority and minority class
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]


# In[37]:


print(df_majority.shape)
print(df_minority.shape)


# In[38]:


# downsample majority class to match minority count
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)


# In[39]:


df_majority_downsampled.shape


# In[40]:


df_downsampled = pd.concat([df_majority_downsampled, df_minority])


# In[41]:


df_downsampled.shape


# In[42]:


df_downsampled.head()


# In[43]:


# shuffle the final dataframe
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)


# In[44]:


df_downsampled.head()


# In[45]:


df_downsampled["rainfall"].value_counts()


# In[46]:


# split features and target as X and y
X = df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]


# In[47]:


print(X)


# In[48]:


print(y)


# In[49]:


# splitting the data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# **Model Training**

# In[50]:


rf_model = RandomForestClassifier(random_state=42)

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}


# In[51]:


# Hypertuning using GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

grid_search_rf.fit(X_train, y_train)


# In[52]:


best_rf_model = grid_search_rf.best_estimator_

print("best parameters for Random Forest:", grid_search_rf.best_params_)


# **Model Evaluation**

# In[53]:


cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))


# In[54]:


# test set performance
y_pred = best_rf_model.predict(X_test)

print("Test set Accuracy:", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# **Prediction on unknown data**

# In[55]:


input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)

input_df = pd.DataFrame([input_data], columns=['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine','winddirection', 'windspeed'])


# In[56]:


input_df


# In[57]:


prediction = best_rf_model.predict(input_df)


# In[58]:


print(prediction)


# In[59]:


prediction[0]


# In[60]:


prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")


# In[61]:


# save model and feature names to a pickle file
model_data = {"model": best_rf_model, "feature_names": X.columns.tolist()}

with open("rainfall_prediction_model.pkl", "wb") as file:
  pickle.dump(model_data, file)


# **Load the saved model and file and use it for prediction**

# In[62]:


import pickle
import pandas as pd


# In[63]:


# load the trained model and feature names from the pickle file
with open("rainfall_prediction_model.pkl", "rb") as file:
  model_data = pickle.load(file)


# In[64]:


model = model_data["model"]
feature_names = model_data["feature_names"]


# In[65]:


input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)

input_df = pd.DataFrame([input_data], columns=feature_names)


# In[66]:


prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")


# In[66]:




