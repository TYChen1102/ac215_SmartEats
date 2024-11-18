import json
import subprocess
from google.cloud import storage

bucket_name = "ac215smarteat"

# Step 1: Load the JSON content from the previous model's output
def extract_input():
    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    # read step1_output.json
    blob = bucket.blob('shared_results/step2_output.json')
    step2_json = blob.download_as_text()
    step2_output = json.loads(step2_json)
    blob2 = bucket.blob('shared_results/step3_output.json')
    step3_json = blob2.download_as_text()
    step3_output = json.loads(step3_json)

    
    meal_info = (
    f"This is the nutrition content and calories of the user's meal: "
    f"{step2_output['Description']}.\n"
    f"Energy: {step2_output['Energy']}, Carbohydrates: {step2_output['Carbohydrate']}, "
    f"Protein: {step2_output['Protein']}, Fat: {step2_output['Fat']}."
    f"The risk of 4 potential relavant diseases are Obesity: {step3_output["Obesity"]}, Diabetes:{step3_output["Diabetes"]}, \n"
    f"High Cholesterol: {step3_output["High Cholesterol"]}, Hypertension: {step3_output["Hypertension"]}. Could you give us some dietary advice based on these information?"
    )
   
    print("Formatted Meal Information for LLM Input:\n", meal_info)
    return meal_info



meal_info=extract_input()


# Step 3: Prepare the RAG system call
# Assume cli.py is your entry point for the LLM chat system.

def interact_with_llm(prompt):
    """Send the prompt to the LLM using RAG via cli.py."""
        # Upgrade chromadb before running the script
    try:
        subprocess.run(["pip", "install", "--upgrade", "chromadb"], check=True)
        print("Successfully upgraded chromadb.")
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading chromadb: {e.stderr}")
        return
        
    # Use subprocess to call the Python script with the necessary arguments
    try:
        command = [ "python", "./cli.py", "--chat", "--chunk_type", "char-split", "--query_text", str(meal_info) ] 
        result = subprocess.run(
            command
        )
    except subprocess.CalledProcessError as e:
        print(f"Error interacting with LLM: {e.stderr}")

# Step 4: Send the formatted meal information to the LLM
interact_with_llm(meal_info)


