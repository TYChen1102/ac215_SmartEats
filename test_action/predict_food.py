import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

#import os
#import shutil
#import glob
from google.cloud import storage
#import argparse
import json

gcp_project = "ac215-project"
bucket_name = "ac215smarteat"

# Download test image
def download():
    storage_client = storage.Client(project=gcp_project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('shared_results/test_food.png')
    blob.download_to_filename('test_food.png')

    print('Step1 input downloaded from GCP bucket.')

# Read and process the image
def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict on the image
def make_prediction(img):
    num_to_food = {
        0:'Baked Potato',
        1:'Crispy Chicken',
        2:'Donut',
        3:'Fries',
        4:'Hot Dog',
        5:'Sandwich',
        6:'Taco', 
        7:'Taquito', 
        8:'apple_pie', 
        9:'burger', 
        10:'butter_naan', 
        11:'chai', 
        12:'chapati', 
        13:'cheesecake', 
        14:'chicken_curry', 
        15:'chole_bhature', 
        16:'dal_makhani', 
        17:'dhokla', 
        18:'fried_rice', 
        19:'ice_cream', 
        20:'idli', 
        21:'jalebi', 
        22:'kaathi_rolls', 
        23:'kadai_paneer', 
        24:'kulfi', 
        25:'masala_dosa', 
        26:'momos', 
        27:'omelette', 
        28:'paani_puri', 
        29:'pakode', 
        30:'pav_bhaji', 
        31:'pizza', 
        32:'samosa', 
        33:'sushi'
    }
    predictions = model.predict(img)[0]
    pred_class = np.argmax(predictions)
    pred_class = num_to_food[pred_class]
    return pred_class, np.max(predictions)

# Save preditions and upload output to bucket shared_results folder
def send_output(label, probability):

    output = {
        'food': label,
        'prob': float(probability)
    }

    with open('step1_output.json', 'w') as outfile:
        json.dump(output, outfile)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('shared_results/step1_output.json')
    blob.upload_from_filename('step1_output.json')

    print('Step1 output uploaded to GCP bucket.')

model = load_model('food_model_EfficientNet.h5')
download()
img = process_image('test_food.png')
label, probability = make_prediction(img)
send_output(label, probability)
