from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import File, Form
from tempfile import TemporaryDirectory
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import APIRouter

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from google.cloud import storage
import json
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
import joblib
import subprocess

gcp_project = "ac215-project"
bucket_name = "ac215smarteat"

storage_client = storage.Client(project=gcp_project)
bucket = storage_client.bucket(bucket_name)

api_key = 'HKDAbEhIHFiO9tKvNa4KmtHdiolSBIg5bf20cZvD'

router = APIRouter()


print("loading models")
model = load_model('models/food_model_EfficientNet.h5')

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
    return pred_class, np.max(predictions),num_to_food


def food_to_nutrition_step_2(food_item, weight, USDA_API_key):
    response = requests.get(
    f"https://api.nal.usda.gov/fdc/v1/foods/search",
    params={
    "api_key": USDA_API_key,
    "query": food_item
    })

    if response.status_code == 200:
        data = response.json()

        # List to store the relevant nutrients data
        nutrients_data = []

        # Iterate through the foods returned by the API
        for food in data['foods']:
            # Check if there's serving size information available
            serving_size = None
            serving_size_unit = None

            if 'servingSize' in food:
                serving_size = food['servingSize']
                serving_size_unit = food.get('servingSizeUnit', '')

                # Extract the nutrients per 100g
                for nutrient in food['foodNutrients']:
                    if nutrient['nutrientName'] in ['Energy', 'Carbohydrate, by difference', 'Total lipid (fat)', 'Protein']:
                        # Append the nutrient details to the list
                        nutrients_data.append({
                        'Description': food['description'],
                        'Nutrient': nutrient['nutrientName'],
                        'Amount': nutrient['value'],
                        'Unit': nutrient['unitName'],
                        'Serving Size': f"{serving_size} {serving_size_unit}" if serving_size else 'N/A'
                        })
                        # Create a DataFrame from the list of nutrients
                        nutrients_df = pd.DataFrame(nutrients_data)
    else:
        print(f"Error: {response.status_code}")


    nutrients_df['Amount'] = nutrients_df['Amount'].astype(str) + " " + nutrients_df['Unit']
    nutrients_df_wide = nutrients_df.pivot_table(index='Description', columns='Nutrient', values='Amount', aggfunc="first").reset_index()

    serving_size = nutrients_df[['Description', 'Serving Size']].drop_duplicates(subset='Description')
    nutrient_components = pd.merge(nutrients_df_wide, serving_size, on="Description", how="left")

    # Implement cosine similarity to find the closest matched description in the resulting dataframe with the food item predicted
    combined_strings = pd.concat([pd.Series([food_item]), nutrient_components['Description']]) # Combine query string with the Pandas Series
    vectorizer = TfidfVectorizer().fit_transform(combined_strings) # Convert strings to vectors using TfidfVectorizer
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten() # Calculate cosine similarity between the query and the list of strings
    closest_match_index = cosine_sim.argmax() # Find the index of the string with the highest similarity

    nutrient_components['Cosine similarity score'] = pd.Series(cosine_sim, index=nutrient_components.index) # extract the closet matching row
    output = nutrient_components.iloc[closest_match_index].to_frame().T # switch row with column

    output['Carbohydrate (numeric)'] = output['Carbohydrate, by difference'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
    output['Energy (numeric)'] = output['Energy'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
    output['Protein (numeric)'] = output['Protein'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
    output['Fat (numeric)'] = output['Total lipid (fat)'].str.extract(r'(\d+\.\d+|\d+)').astype(float)

    carbo_unit = output['Carbohydrate, by difference'].str.split().str[1].iloc[0]
    energy_unit = output['Energy'].str.split().str[1].iloc[0]
    protein_unit = output['Protein'].str.split().str[1].iloc[0]
    fat_unit = output['Total lipid (fat)'].str.split().str[1].iloc[0]


    output_final = output[['Description', 'Cosine similarity score']].copy()
    if output['Serving Size'].eq('N/A').any():
        new_carbo = weight / 100 * output['Carbohydrate (numeric)'].iloc[0]
        output_final['Carbohydrate'] = (new_carbo).astype(str) + " " + carbo_unit

        new_energy = weight / 100 * output['Energy (numeric)'].iloc[0]
        output_final['Energy'] = (new_energy).astype(str) + " " + energy_unit

        new_protein = weight / 100 * output['Protein (numeric)'].iloc[0]
        output_final['Protein'] = (new_protein).astype(str) + " " + protein_unit

        new_fat = weight / 100 * output['Fat (numeric)'].iloc[0]
        output_final['Fat'] = (new_fat).astype(str) + " " + fat_unit

    else:
        serving_size = output['Serving Size'].str.extract(r'(\d+\.\d+|\d+)').astype(float)

        new_carbo = weight / serving_size * output['Carbohydrate (numeric)'].iloc[0]
        output_final['Carbohydrate'] = (new_carbo).astype(str) + " " + carbo_unit

        new_energy = weight / serving_size * output['Energy (numeric)'].iloc[0]
        output_final['Energy'] = (new_energy).astype(str) + " " + energy_unit

        new_protein = weight / serving_size * output['Protein (numeric)'].iloc[0]
        output_final['Protein'] = (new_protein).astype(str) + " " + protein_unit

        new_fat = weight / serving_size * output['Fat (numeric)'].iloc[0]
        output_final['Fat'] = (new_fat).astype(str) + " " + fat_unit

    print("Nutrition components: ", output_final)

    output_json = {
    'Description': output_final['Description'].iloc[0],
    'Cosine similarity score': output_final['Cosine similarity score'].iloc[0],
    'Carbohydrate': output_final['Carbohydrate'].iloc[0],
    'Energy': output_final['Energy'].iloc[0],
    'Protein': output_final['Protein'].iloc[0],
    'Fat': output_final['Fat'].iloc[0]
    }
    return output_json



def transform_data_step_3(json_data):
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



# Routes
@router.post("/predict_nutrition")
async def predict_food_from_image(file: bytes = File(...), weight: float = Form()):
    # Save the image
    with open("test.png", "wb") as output:
        output.write(file)

    # Step 1 image food classification
    img = cv2.imread("test.png")
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    label, probability, num_to_food = make_prediction(img)

    print(label, probability)  # step 1 classification output

    # Step 2 food nutrition prediction
    food_input = label
    weight_input = weight if weight is not None else 500
    step2_output = food_to_nutrition_step_2(food_input, weight_input, api_key)

    # step 3 predict disease risk
    step2_output_df = transform_data_step_3(step2_output)

    # Load trained models and predict risks
    obesity_model = joblib.load('models/Obesity_model.pkl')
    diabetes_model = joblib.load('models/Diabetes_model.pkl')
    high_cholesterol_model = joblib.load('models/High Cholesterol_model.pkl')
    hypertension_model = joblib.load('models/Hypertension_model.pkl')

    obesity_prob = obesity_model.predict_proba(step2_output_df)[0][1]
    diabetes_prob = diabetes_model.predict_proba(step2_output_df)[0][1]
    high_cholesterol_prob = high_cholesterol_model.predict_proba(step2_output_df)[0][1]
    hypertension_prob = hypertension_model.predict_proba(step2_output_df)[0][1]

    disease_output = {
        'Obesity': round(float(obesity_prob), 4),
        'Diabetes': round(float(diabetes_prob), 4),
        'High Cholesterol': round(float(high_cholesterol_prob), 4),
        'Hypertension': round(float(hypertension_prob), 4)
    }

    # Convert all numpy float32 types to Python native float wherever necessary
   
   
    response_data = {
        'label': label,
        'probability': float(probability) if isinstance(probability, (np.float32, np.float64)) else probability,
        'nutritional_info': {
            k: float(v) if isinstance(v, (np.float32, np.float64)) else v
            for k, v in step2_output.items()
        },
        'disease_risks': disease_output
    }

    return JSONResponse(content=response_data)



