import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

### Extract input from step1 output
input = "sushi"
weight = 200



### Define a function to query API to extract nutrition components
def food_to_nutrition(food_name, USDA_API_key):
    """
    """
    response = requests.get(
    f"https://api.nal.usda.gov/fdc/v1/foods/search",
    params={
    "api_key": USDA_API_key,
    "query": food_name
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
    nutrients_df_wide_final = pd.merge(nutrients_df_wide, serving_size, on="Description", how="left")

    return nutrients_df_wide_final



### Run the defined function with the input
api_key = 'HKDAbEhIHFiO9tKvNa4KmtHdiolSBIg5bf20cZvD'
nutrient_components = food_to_nutrition(input, api_key)

# Implement cosine similarity to find the closest matched description in the resulting dataframe with the food item predicted
combined_strings = pd.concat([pd.Series([input]), nutrient_components['Description']]) # Combine query string with the Pandas Series
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


output_final = output[['Description', 'Cosine similarity score']]
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


print("Step2 output: ", output_final)
