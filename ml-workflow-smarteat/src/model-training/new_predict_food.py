import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import zipfile
import cv2
import pandas as pd
from google.cloud import storage

# Initialize GCP Bucket Info
gcp_project = os.environ.get("GCP_PROJECT", "ml-workflow-cnn")
bucket_name = os.environ.get("GCS_BUCKET_NAME", "ml-workflow-cnn")

# Load pretrained model
model = load_model('food_model_EfficientNet.h5')

# Mapping of class indices to food labels
num_to_food = {
    0: 'Baked Potato', 1: 'Crispy Chicken', 2: 'Donut', 3: 'Fries', 4: 'Hot Dog',
    5: 'Sandwich', 6: 'Taco', 7: 'Taquito', 8: 'apple_pie', 9: 'burger',
    10: 'butter_naan', 11: 'chai', 12: 'chapati', 13: 'cheesecake',
    14: 'chicken_curry', 15: 'chole_bhature', 16: 'dal_makhani', 17: 'dhokla',
    18: 'fried_rice', 19: 'ice_cream', 20: 'idli', 21: 'jalebi', 22: 'kaathi_rolls',
    23: 'kadai_paneer', 24: 'kulfi', 25: 'masala_dosa', 26: 'momos',
    27: 'omelette', 28: 'paani_puri', 29: 'pakode', 30: 'pav_bhaji',
    31: 'pizza', 32: 'samosa', 33: 'sushi'
}

# Step 1: Download and unzip the file
def download_and_unzip():
    storage_client = storage.Client(project=gcp_project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob("clean.zip")
    
    # Download clean.zip
    local_zip = "clean.zip"
    blob.download_to_filename(local_zip)
    print(f"[INFO] clean.zip downloaded from bucket {bucket_name}")
    
    # Unzip the contents
    unzip_dir = "images"
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print(f"[INFO] Unzipped clean.zip into {unzip_dir}")
    return unzip_dir

# Step 2: Process each image
def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize
    return img

# Step 3: Predict labels and probabilities
def make_predictions(image_folder):
    results = []
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        if os.path.isfile(img_path):
            img = process_image(img_path)
            predictions = model.predict(img)[0]
            label_idx = np.argmax(predictions)
            label = num_to_food[label_idx]
            probability = round(float(np.max(predictions)), 4)
            results.append({"image": img_name, "label": label, "probability": probability})
            print(f"[INFO] Prediction for {img_name}: {label} ({probability})")
    return results

# Step 4: Output results to a table
def save_results_to_table(results):
    df = pd.DataFrame(results)
    df.to_csv("predictions.csv", index=False)
    print("[INFO] Predictions saved to predictions.csv")
    print(df)

# Main execution
if __name__ == "__main__":
    # Download and unzip images
    image_folder = download_and_unzip()

    # Make predictions
    results = make_predictions(f"{image_folder}/pizza")

    # Save and display results
    save_results_to_table(results)
