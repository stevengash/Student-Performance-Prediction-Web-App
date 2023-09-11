#!/usr/bin/env python
# coding: utf-8

# In[8]:


#pip install Flask


# In[9]:


from flask import Flask, render_template, request
import pandas as pd
import joblib


# In[10]:


app = Flask(__name__)


# In[11]:


model_filename = 'ridge_regression_model1.pkl'
best_model_pipeline = joblib.load(model_filename)


# In[12]:


@app.route('/')
def home():
    return render_template('index.html')


# In[13]:


@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input from the form
    gender = request.form['gender']
    race = request.form['race']
    education = request.form['parental_education']
    lunch = request.form['lunch']
    prep_course = request.form['test_prep']
    math_score = float(request.form['math_score'])
    reading_score = float(request.form['reading_score'])
    
    # Create a DataFrame with the user input
    new_data = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [education],
        'lunch': [lunch],
        'test preparation course': [prep_course],
        'math score': [math_score],
        'reading score': [reading_score]
    })
    
    # Use the model to make predictions
    predicted_score = best_model_pipeline.predict(new_data)
    # Round the predicted score to 2 decimal 
    rounded_score = round(predicted_score[0], 2)
    # Display the rounded prediction on a new page
    return render_template('result.html', prediction=rounded_score)

import warnings
from app import app  # Import your Flask app instance here (assuming your app is named app.py)

# Suppress all warnings (not recommended for production use)
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    app.run(debug=True)

# In[ ]:




