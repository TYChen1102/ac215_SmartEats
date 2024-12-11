from fastapi import FastAPI
from fastapi import File, Form
from tempfile import TemporaryDirectory
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routers import image_to_nutrition, llm_rag_chat
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

gcp_project = "ac215-smarteat-437821"
bucket_name = "ac215smarteat"

storage_client = storage.Client(project=gcp_project)
bucket = storage_client.bucket(bucket_name)

api_key = 'HKDAbEhIHFiO9tKvNa4KmtHdiolSBIg5bf20cZvD'

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/")
async def get_index():
    return {"message": "API Server launched!"}

# Additional routers here
app.include_router(image_to_nutrition.router, prefix="/image_to_nutrition")
app.include_router(llm_rag_chat.router, prefix="/llm-rag")





#other things to delete


#print("loading models")
#model = load_model('models/food_model_EfficientNet.h5')
#
## Predict on the image
#def make_prediction(img):
#    num_to_food = {
#        0:'Baked Potato',
#        1:'Crispy Chicken',
#        2:'Donut',
#        3:'Fries',
#        4:'Hot Dog',
#        5:'Sandwich',
#        6:'Taco', 
#        7:'Taquito', 
#        8:'apple pie',
#        9:'burger',
#        10:'butter naan',
#        11:'chai',
#        12:'chapati', 
#        13:'cheesecake', 
#        14:'chicken curry',
#        15:'chole bhature',
#        16:'dal makhani',
#        17:'dhokla',
#        18:'fried rice',
#        19:'ice cream',
#        20:'idli',
#        21:'jalebi', 
#        22:'kaathi rolls',
#        23:'kadai paneer',
#        24:'kulfi',
#        25:'masala dosa',
#        26:'momos',
#        27:'omelette', 
#        28:'paani puri',
#        29:'pakode',
#        30:'pav bhaji', 
#        31:'pizza', 
#        32:'samosa', 
#        33:'sushi'
#    }
#    predictions = model.predict(img)[0]
#    pred_class = np.argmax(predictions)
#    pred_class = num_to_food[pred_class]
#    return pred_class, np.max(predictions),num_to_food
#
## Save preditions and upload output to bucket shared_results folder
#def send_output(label, probability):
#
#    output = {
#        'food': label,
#        'prob': float(probability)
#    }
#
#    with open('step1_output.json', 'w') as outfile:
#        json.dump(output, outfile)
#
#    storage_client = storage.Client()
#    bucket = storage_client.bucket(bucket_name)
#    blob = bucket.blob('shared_results/step1_output.json')
#    blob.upload_from_filename('step1_output.json')
#
#    print('Step1 output uploaded to GCP bucket.')
#
#def extract_input_step_2():
#    # Initialize the Google Cloud Storage client
#    storage_client = storage.Client()
#    bucket = storage_client.bucket(bucket_name)
#    # read step1_output.json
#    blob = bucket.blob('shared_results/step1_output.json')
#    step1_json = blob.download_as_text()
#    step1_output = json.loads(step1_json)
#    
#    # read Weight.json
#    w_blob = bucket.blob('shared_results/Weight.json')
#    weight_json = w_blob.download_as_text()
#    weight = json.loads(weight_json)
#    step1_output.update(weight)
#    return step1_output
#
#
#
#def food_to_nutrition_step_2(food_item, weight, USDA_API_key):
#    response = requests.get(
#    f"https://api.nal.usda.gov/fdc/v1/foods/search",
#    params={
#    "api_key": USDA_API_key,
#    "query": food_item
#    })
#
#    if response.status_code == 200:
#        data = response.json()
#
#        # List to store the relevant nutrients data
#        nutrients_data = []
#
#        # Iterate through the foods returned by the API
#        for food in data['foods']:
#            # Check if there's serving size information available
#            serving_size = None
#            serving_size_unit = None
#
#            if 'servingSize' in food:
#                serving_size = food['servingSize']
#                serving_size_unit = food.get('servingSizeUnit', '')
#
#                # Extract the nutrients per 100g
#                for nutrient in food['foodNutrients']:
#                    if nutrient['nutrientName'] in ['Energy', 'Carbohydrate, by difference', 'Total lipid (fat)', 'Protein']:
#                        # Append the nutrient details to the list
#                        nutrients_data.append({
#                        'Description': food['description'],
#                        'Nutrient': nutrient['nutrientName'],
#                        'Amount': nutrient['value'],
#                        'Unit': nutrient['unitName'],
#                        'Serving Size': f"{serving_size} {serving_size_unit}" if serving_size else 'N/A'
#                        })
#                        # Create a DataFrame from the list of nutrients
#                        nutrients_df = pd.DataFrame(nutrients_data)
#    else:
#        print(f"Error: {response.status_code}")
#
#
#    nutrients_df['Amount'] = nutrients_df['Amount'].astype(str) + " " + nutrients_df['Unit']
#    nutrients_df_wide = nutrients_df.pivot_table(index='Description', columns='Nutrient', values='Amount', aggfunc="first").reset_index()
#
#    serving_size = nutrients_df[['Description', 'Serving Size']].drop_duplicates(subset='Description')
#    nutrient_components = pd.merge(nutrients_df_wide, serving_size, on="Description", how="left")
#
#    # Implement cosine similarity to find the closest matched description in the resulting dataframe with the food item predicted
#    combined_strings = pd.concat([pd.Series([food_item]), nutrient_components['Description']]) # Combine query string with the Pandas Series
#    vectorizer = TfidfVectorizer().fit_transform(combined_strings) # Convert strings to vectors using TfidfVectorizer
#    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten() # Calculate cosine similarity between the query and the list of strings
#    closest_match_index = cosine_sim.argmax() # Find the index of the string with the highest similarity
#
#    nutrient_components['Cosine similarity score'] = pd.Series(cosine_sim, index=nutrient_components.index) # extract the closet matching row
#    output = nutrient_components.iloc[closest_match_index].to_frame().T # switch row with column
#
#    output['Carbohydrate (numeric)'] = output['Carbohydrate, by difference'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
#    output['Energy (numeric)'] = output['Energy'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
#    output['Protein (numeric)'] = output['Protein'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
#    output['Fat (numeric)'] = output['Total lipid (fat)'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
#
#    carbo_unit = output['Carbohydrate, by difference'].str.split().str[1].iloc[0]
#    energy_unit = output['Energy'].str.split().str[1].iloc[0]
#    protein_unit = output['Protein'].str.split().str[1].iloc[0]
#    fat_unit = output['Total lipid (fat)'].str.split().str[1].iloc[0]
#
#
#    output_final = output[['Description', 'Cosine similarity score']].copy()
#    if output['Serving Size'].eq('N/A').any():
#        new_carbo = weight / 100 * output['Carbohydrate (numeric)'].iloc[0]
#        output_final['Carbohydrate'] = (new_carbo).astype(str) + " " + carbo_unit
#
#        new_energy = weight / 100 * output['Energy (numeric)'].iloc[0]
#        output_final['Energy'] = (new_energy).astype(str) + " " + energy_unit
#
#        new_protein = weight / 100 * output['Protein (numeric)'].iloc[0]
#        output_final['Protein'] = (new_protein).astype(str) + " " + protein_unit
#
#        new_fat = weight / 100 * output['Fat (numeric)'].iloc[0]
#        output_final['Fat'] = (new_fat).astype(str) + " " + fat_unit
#
#    else:
#        serving_size = output['Serving Size'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
#
#        new_carbo = weight / serving_size * output['Carbohydrate (numeric)'].iloc[0]
#        output_final['Carbohydrate'] = (new_carbo).astype(str) + " " + carbo_unit
#
#        new_energy = weight / serving_size * output['Energy (numeric)'].iloc[0]
#        output_final['Energy'] = (new_energy).astype(str) + " " + energy_unit
#
#        new_protein = weight / serving_size * output['Protein (numeric)'].iloc[0]
#        output_final['Protein'] = (new_protein).astype(str) + " " + protein_unit
#
#        new_fat = weight / serving_size * output['Fat (numeric)'].iloc[0]
#        output_final['Fat'] = (new_fat).astype(str) + " " + fat_unit
#
#    print("Nutrition components: ", output_final)
#    return output_final
#
#def send_output_step_2(output_final):
#    output = {
#    'Description': output_final['Description'].iloc[0],
#    'Cosine similarity score': output_final['Cosine similarity score'].iloc[0],
#    'Carbohydrate': output_final['Carbohydrate'].iloc[0],
#    'Energy': output_final['Energy'].iloc[0],
#    'Protein': output_final['Protein'].iloc[0],
#    'Fat': output_final['Fat'].iloc[0]
#    }
#
#    # Write the result to a shared folder
#    with open('step2_output.json', 'w') as outfile:
#        json.dump(output, outfile)
#
#    # Upload output to bucket shared_results folder
#    storage_client = storage.Client()
#    bucket = storage_client.bucket(bucket_name)
#    blob = bucket.blob('shared_results/step2_output.json')
#    blob.upload_from_filename('step2_output.json')
#    print('Step2 output uploaded to GCP bucket.')
#
## download the nutrition information
#def extract_input_step_3():
#    print("Extract input")
#
#    # Initialize the Google Cloud Storage client
#    storage_client = storage.Client()
#    bucket = storage_client.bucket(bucket_name)
#    blob = bucket.blob('shared_results/step2_output.json')
#    step2_json = blob.download_as_text()
#    step2_output = json.loads(step2_json)
#
#    print(step2_output)
#    return step2_output
#
#def transform_data_step_3(json_data):
#    nutritional_columns = ['Carbohydrate (G)', 'Energy (KCAL)', 'Protein (G)', 'Fat (G)']
#    transformed_data = {
#        'Carbohydrate (G)': float(json_data['Carbohydrate'].split()[0]),  # Extract number
#        'Energy (KCAL)': float(json_data['Energy'].split()[0]),            # Extract number
#        'Protein (G)': float(json_data['Protein'].split()[0]),             # Extract number
#        'Fat (G)': float(json_data['Fat'].split()[0])                      # Extract number
#    }
#    df_transformed = pd.DataFrame([transformed_data])
#    print(df_transformed)
#    return df_transformed
#
#def send_output_step_3(output):
#    # Write the result to a shared folder
#    with open('step3_output.json', 'w') as outfile:
#        json.dump(output, outfile)
#
#    # Upload output to bucket shared_results folder
#    print("upload")
#    storage_client = storage.Client()
#    bucket = storage_client.bucket(bucket_name)
#    blob = bucket.blob('shared_results/step3_output.json')
#    blob.upload_from_filename('step3_output.json')
#
#    print('Step3 output uploaded to GCP bucket.')
#
#
#
## Routes
#@app.post("/predict")
#async def predict_food_from_image(file: bytes = File(...),weight:float = Form()):
#    # Save the image
#    with open("test.png", "wb") as output:
#        output.write(file)
#
#    # Step 1 image food classification
#    img = cv2.imread("test.png")
#    img = cv2.resize(img, (224, 224))  # Resize to 224x224
#    img = np.expand_dims(img, axis=0)  # Add batch dimension
#
#    label, probability,num_to_food = make_prediction(img)
#    send_output(label, probability)
#
#    print(label,probability)
#
#    ## Step 2 food nutrition preditcion
#    step1_output = extract_input_step_2()
#    food_input = step1_output['food']
#    if weight is None:
#        weight_input = step1_output['Weight']
#    else:
#        weight_input=weight
#    step2_output = food_to_nutrition_step_2(food_input, weight_input, api_key)
#    send_output_step_2(step2_output)
#
#    # step 3 predict diease risk
#    step2_output = extract_input_step_3()
#    step2_output_df = transform_data_step_3(step2_output)
#
#    # # Use machine learning models to predict disease risk
#    # # Load trained models
#    obesity_model = joblib.load('models/Obesity_model.pkl')
#    diabetes_model = joblib.load('models/Diabetes_model.pkl')
#    high_cholesterol_model = joblib.load('models/High Cholesterol_model.pkl')
#    hypertension_model = joblib.load('models/Hypertension_model.pkl')
#
#    # Predict risks
#    # Predict the probability of each disease
#    obesity_prob = obesity_model.predict_proba(step2_output_df)[0][1]  # Probability of having obesity
#    diabetes_prob = diabetes_model.predict_proba(step2_output_df)[0][1]  # Probability of having diabetes
#    high_cholesterol_prob = high_cholesterol_model.predict_proba(step2_output_df)[0][1]  # Probability of high cholesterol
#    hypertension_prob = hypertension_model.predict_proba(step2_output_df)[0][1]  # Probability of having hypertension
#
#    output = {
#        'Obesity': round(obesity_prob, 4).astype(str),
#        'Diabetes': round(diabetes_prob, 4).astype(str),
#        'High Cholesterol': round(high_cholesterol_prob, 4).astype(str),
#        'Hypertension': round(hypertension_prob, 4).astype(str)
#    }
#
#    send_output_step_3(output)
#    
#
#
#    # Step 1: Load the JSON content from the previous model's output
#    def extract_input_LLM():
#        # Initialize the Google Cloud Storage client
#        storage_client = storage.Client()
#        bucket = storage_client.bucket(bucket_name)
#        # read step1_output.json
#        blob = bucket.blob('shared_results/step2_output.json')
#        step2_json = blob.download_as_text()
#        step2_output = json.loads(step2_json)
#        blob2 = bucket.blob('shared_results/step3_output.json')
#        step3_json = blob2.download_as_text()
#        step3_output = json.loads(step3_json)
#
#        
#        meal_info = (
#        f"This is the nutrition content and calories of the user's meal: "
#        f"{step2_output['Description']}.\n"
#        f"Energy: {step2_output['Energy']}, Carbohydrates: {step2_output['Carbohydrate']}, "
#        f"Protein: {step2_output['Protein']}, Fat: {step2_output['Fat']}."
#        f"The risk of 4 potential relavant diseases are Obesity: {step3_output['Obesity']}, Diabetes:{step3_output['Diabetes']}, \n"
#        f"High Cholesterol: {step3_output['High Cholesterol']}, Hypertension: {step3_output['Hypertension']}. Could you give us some dietary advice based on these information?"
#        )
#    
#        print("Formatted Meal Information for LLM Input:\n", meal_info)
#        return meal_info
#
#
#
#    meal_info=extract_input_LLM()
#
#
#    # Step 3: Prepare the RAG system call
#    # Assume cli.py is your entry point for the LLM chat system.
#
#    def interact_with_llm(prompt):
#        """Send the prompt to the LLM using RAG via cli.py."""
#            
#        # Use subprocess to call the Python script with the necessary arguments
#        try:
#            command = [ "python", "./cli.py", "--chat", "--chunk_type", "char-split", "--query_text", str(meal_info) ] 
#            res = subprocess.run(
#                command
#            )
#        except subprocess.CalledProcessError as e:
#            print(f"Error interacting with LLM: {e.stderr}")
#
#    # Step 4: Send the formatted meal information to the LLM
#    res=interact_with_llm(meal_info)
#    output_file_name = 'final_step_LLM_output.txt'
#
#
#    blob = bucket.blob(f"shared_results/{output_file_name}")
#    generated_text = blob.download_as_text()
#    return {
#        "meal_info": meal_info,
#        "text": generated_text
#    } 
#
#
