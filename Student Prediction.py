#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.model_selection import GridSearchCV #for hypertuning
from sklearn.linear_model import LogisticRegression


# In[12]:


#pip install scikit-learn==1.3.0
parental_education_unique = df['parental level of education'].unique()
parental_education_unique


# In[13]:


df = pd.read_csv('exams.csv')
df


# In[14]:


df.info()


# In[15]:


df.isna().any()


# In[16]:


df.describe()


# In[17]:


# Identify the categorical features
cat_cols = [col for col in df.columns if df[col].dtype=='O']
cat_cols


# In[18]:


for col in cat_cols:
    print(df[col].unique())


# In[19]:


# Get list of categorical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'O']

# Loop over categorical columns
for col in cat_cols:
    unique_vals = df[col].nunique()
    total_vals = len(df[col])
    unique_pct = unique_vals / total_vals * 100
    print(f"{col}: {unique_vals} unique values ({unique_pct:.2f} of total)")


# In[20]:


for col in cat_cols:
    df[col] = df[col].astype('category')
df.memory_usage(deep=True)


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Creating Bar chart as the Target variable is Continuous
df['writing score'].hist();


# In[22]:


plt.scatter(df['math score'],df['writing score'],marker = '*', color = 'g')
plt.scatter(df['reading score'],df['writing score'],marker = '+', color = 'b')


# In[23]:


final_cols = ['gender', 'race/ethnicity','parental level of education','lunch', 'test preparation course', 'math score','reading score']

df_final = df[final_cols]
X = df_final[final_cols]
y = df['writing score']
X
y


# In[24]:


num_cols = ['math score', 'reading score']


# In[25]:


from sklearn.pipeline import Pipeline
# Create a pipeline for categorical data
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
categorical_cols = ['gender',
 'race/ethnicity',
 'parental level of education',
 'lunch',
 'test preparation course']

 # Apply the pipeline to the categorical columns
categorical_df = categorical_pipeline.fit_transform(df[categorical_cols])

# Convert the sparse matrix to a Pandas DataFrame
categorical_df = pd.DataFrame(categorical_df.toarray())

# Concatenate the categorical data with the original DataFrame
df = pd.concat([df.drop(categorical_cols, axis=1), categorical_df], axis=1)


# In[26]:


# define the preprocessing pipelines for numerical and categorical features
num_cols = ['math score', 'reading score']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_cols = ['gender',
 'race/ethnicity',
 'parental level of education',
 'lunch',
 'test preparation course']

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())])


# In[27]:


# convert all column names to strings
df.columns = df.columns.astype(str)
df


# In[28]:


num_pipeline = Pipeline([
    ('num_smoothening',PowerTransformer())
])

# define the column transformer to preprocess both numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, categorical_cols)])


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

# check the shapes of the training and test data
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')
X_train


# In[30]:


# define the final pipeline that includes the column transformer and a logistic regression model
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', LinearRegression())])


# In[31]:


# fit the pipeline to the training data
pipe.fit(X_train, y_train)

# evaluate the pipeline on the test data
score = pipe.score(X_test, y_test)
print(f'Test score: {score:.2f}')


# In[32]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np


# In[33]:


# List of regression models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'XGBoost Regressor': XGBRegressor(),
    #'LightGBM Regressor': LGBMRegressor(),
    'CatBoost Regressor': CatBoostRegressor(silent=True)  # Set silent=True to suppress output
}
# Create a LightGBM Regressor with verbose=-1
lightgbm_regressor = LGBMRegressor(verbose=-1)

# Add it to your models dictionary
models['LightGBM Regressor'] = lightgbm_regressor


# Dictionary to store mean MSE scores for each model
mse_scores = {}

# Evaluate each model using cross-validation
for model_name, model in models.items():
    # Create a pipeline for the current model
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Convert negative MSE scores to positive
    mse_scores[model_name] = -np.mean(cv_scores)

# Find the model with the lowest MSE
best_model = min(mse_scores, key=mse_scores.get)
best_mse = mse_scores[best_model]

print(f"Mean Squared Error (MSE) for each model:")
for model_name, mse in mse_scores.items():
    print(f"{model_name}: {mse:.2f}")

print(f"\nThe best model is: {best_model} with MSE: {best_mse:.2f}")

# Now, you can fit the best model on the full training data and evaluate it on the test data.
best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', models[best_model])])
best_model_pipeline.fit(X_train, y_train)
test_predictions = best_model_pipeline.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
print(f"\nTest MSE for the best model on the test data: {test_mse:.2f}")


# In[34]:





# In[35]:


import matplotlib.pyplot as plt

# List of regression model names
model_names = list(mse_scores.keys())

# List of MSE scores
mse_values = list(mse_scores.values())

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.barh(model_names, mse_values, color='skyblue')
plt.xlabel('Mean Squared Error (MSE)')
plt.title('Comparison of Regression Models')
plt.gca().invert_yaxis()  # Invert the y-axis to display the best model at the top
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Annotate each bar with its MSE value
for i, v in enumerate(mse_values):
    plt.text(v, i, f'{v:.2f}', va='center', fontsize=12, color='black', fontweight='bold')

plt.tight_layout()
plt.show()


# In[42]:


# Get feature coefficients for Ridge Regression
feature_coefficients = best_model_pipeline.named_steps['regressor'].coef_

# Example of hyperparameter tuning using GridSearchCV for Ridge Regression
param_grid = {'regressor__alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(best_model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


# In[43]:


# Calculate residuals
residuals = y_test - test_predictions

# Create residual plots (e.g., scatterplot of residuals vs. predicted values)
plt.scatter(test_predictions, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")


# In[38]:


# Use the trained Ridge Regression model to make predictions
predictions = best_model_pipeline.predict(X_test)

# Create a DataFrame to store the actual and predicted values
predictions_df = pd.DataFrame({'Actual Writing Score': y_test, 'Predicted Writing Score': predictions})

# Display the first few rows of the predictions DataFrame
print(predictions_df.head())


# In[39]:


import joblib

# Save the trained Ridge Regression model to a file
model_filename = 'ridge_regression_model1.pkl'
joblib.dump(best_model_pipeline, model_filename)

print(f"Model saved as {model_filename}")


# In[40]:


import pandas as pd
from sklearn.pipeline import Pipeline

# Create a new DataFrame with the input data (replace this with your actual new data)
new_data = pd.DataFrame({
    'gender': ['female', 'male'],  # Replace with your values
    'race/ethnicity': ['group A', 'group B'],  # Replace with your values
    'parental level of education': ["bachelor's degree", "associate's degree"],  # Replace with your values
    'lunch': ['standard', 'free/reduced'],  # Replace with your values
    'test preparation course': ['completed', 'none'],  # Replace with your values
    'math score': [75, 80],  # Replace with your values
    'reading score': [85, 90]  # Replace with your values
})

# Use the preprocessor and Ridge Regression model to make predictions
predicted_scores = best_model_pipeline.predict(new_data)

# Add the predicted scores to the new_data DataFrame
new_data['Predicted Writing Score'] = predicted_scores

# Print the DataFrame with predicted scores
print(new_data[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'Predicted Writing Score']])


# In[ ]:





# In[41]:


import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

# Load the pre-trained Ridge Regression model
ridge_model = joblib.load('ridge_regression_model1.pkl')

# Define a function to make predictions
def predict_writing_score(input_data):
    # Create a DataFrame from the input data
    new_data = pd.DataFrame(input_data)

    # Use the preprocessor and Ridge Regression model to make predictions
    predicted_scores = ridge_model.predict(new_data)

    # Add the predicted scores to the new_data DataFrame
    new_data['Predicted Writing Score'] = predicted_scores

    return new_data[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'Predicted Writing Score']]

# Example input data (replace this with user input)
input_data = {
    'gender': ['female', 'male'],
    'race/ethnicity': ['group A', 'group B'],
    'parental level of education': ["bachelor's degree", "associate's degree"],
    'lunch': ['standard', 'free/reduced'],
    'test preparation course': ['completed', 'none'],
    'math score': [75, 80],
    'reading score': [85, 90]
}

# Make predictions using the function
predictions = predict_writing_score(input_data)

# Print the predictions
print(predictions)


# In[66]:





# In[ ]:





# In[ ]:




