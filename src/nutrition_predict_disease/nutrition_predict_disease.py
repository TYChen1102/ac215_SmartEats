# Import necessary libraries
import os
import pandas as pd
import numpy as np
import requests
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from google.cloud import storage
import json
import joblib

gcp_project = "ac215-project"
bucket_name = "ac215smarteat"

# download the nutrition information
def extract_input():
    print("Extract input")

    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('shared_results/step2_output.json')
    step2_json = blob.download_as_text()
    step2_output = json.loads(step2_json)

    print(step2_output)
    return step2_output

def transform_data(json_data):
    nutritional_columns = ['Carbohydrate (G)', 'Energy (KCAL)', 'Protein (G)', 'Fat (G)']
    transformed_data = {
        'Carbohydrate (G)': float(json_data['Carbohydrate'].split()[0]),  # Extract number
        'Energy (KCAL)': float(json_data['Energy'].split()[0]),            # Extract number
        'Protein (G)': float(json_data['Protein'].split()[0]),             # Extract number
        'Fat (G)': float(json_data['Fat'].split()[0])                      # Extract number
    }
    df_transformed = pd.DataFrame([transformed_data])
    print(df_transformed)
    return df_transformed

def send_output(output):
    # Write the result to a shared folder
    with open('step3_output.json', 'w') as outfile:
        json.dump(output, outfile)

    # Upload output to bucket shared_results folder
    print("upload")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('shared_results/step3_output.json')
    blob.upload_from_filename('step3_output.json')

    print('Step3 output uploaded to GCP bucket.')

## Run the above functions
step2_output = extract_input()
step2_output_df = transform_data(step2_output)

# # Use machine learning models to predict disease risk
# # Load trained models
obesity_model = joblib.load('models/Obesity_model.pkl')
diabetes_model = joblib.load('models/Diabetes_model.pkl')
high_cholesterol_model = joblib.load('models/High Cholesterol_model.pkl')
hypertension_model = joblib.load('models/Hypertension_model.pkl')

# Predict risks
# Predict the probability of each disease
obesity_prob = obesity_model.predict_proba(step2_output_df)[0][1]  # Probability of having obesity
diabetes_prob = diabetes_model.predict_proba(step2_output_df)[0][1]  # Probability of having diabetes
high_cholesterol_prob = high_cholesterol_model.predict_proba(step2_output_df)[0][1]  # Probability of high cholesterol
hypertension_prob = hypertension_model.predict_proba(step2_output_df)[0][1]  # Probability of having hypertension

output = {
    'Obesity': round(obesity_prob, 4).astype(str),
    'Diabetes': round(diabetes_prob, 4).astype(str),
    'High Cholesterol': round(high_cholesterol_prob, 4).astype(str),
    'Hypertension': round(hypertension_prob, 4).astype(str)
}

send_output(output)
