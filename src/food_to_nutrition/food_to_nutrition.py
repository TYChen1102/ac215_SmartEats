
"""
Module that contains the command line app.
"""
import os
import json
from google.cloud import storage
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


gcp_project = "ac215-project"
bucket_name = "ac215smarteat"
api_key = 'HKDAbEhIHFiO9tKvNa4KmtHdiolSBIg5bf20cZvD'


def extract_input():
    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    # read step1_output.json
    blob = bucket.blob('shared_results/step1_output.json')
    step1_json = blob.download_as_text()
    step1_output = json.loads(step1_json)

    # read Weight.json
    w_blob = bucket.blob('shared_results/Weight.json')
    weight_json = w_blob.download_as_text()
    weight = json.loads(weight_json)
    step1_output.update(weight)
    return step1_output



def food_to_nutrition(food_item, weight, USDA_API_key):
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
    return output_final



def send_output(output_final):
    output = {
    'Description': output_final['Description'].iloc[0],
    'Cosine similarity score': output_final['Cosine similarity score'].iloc[0],
    'Carbohydrate': output_final['Carbohydrate'].iloc[0],
    'Energy': output_final['Energy'].iloc[0],
    'Protein': output_final['Protein'].iloc[0],
    'Fat': output_final['Fat'].iloc[0]
    }

    # Write the result to a shared folder
    with open('step2_output.json', 'w') as outfile:
        json.dump(output, outfile)

    # Upload output to bucket shared_results folder
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob('shared_results/step2_output.json')
    blob.upload_from_filename('step2_output.json')
    print('Step2 output uploaded to GCP bucket.')


## Run the above functions
step1_output = extract_input()
food_input = step1_output['food']
weight_input = step1_output['Weight']
step2_output = food_to_nutrition(food_input, weight_input, api_key)
send_output(step2_output)
