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

# Load trained models
obesity_model = joblib.load('/models/Obesity_model.pkl')
diabetes_model = joblib.load('/models/Diabetes_model.pkl')
high_cholesterol_model = joblib.load('/models/High Cholesterol_model.pkl')
hypertension_model = joblib.load('/models/Hypertension_model.pkl')

# Output risks


# # Save preditions and upload output to bucket shared_results folder
# def send_output(label, probabilities):

#     output = {
#         'food': label,
#         'weight': probabilities.tolist()
#     }

#     with open('step3_output.json', 'w') as outfile:
#         json.dump(output, outfile)
    
#     print("upload")
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob('shared_results/step3_output.json')
#     blob.upload_from_filename('step3_output.json')

#     print('Step3 output uploaded to GCP bucket.')



## Run the above functions
step2_output = extract_input()
step2_output
