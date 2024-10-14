import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

model = load_model('food_model_EfficientNet.h5')

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

# Read and process the image
def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict on the image
def make_prediction(img_path):
    img = process_image(img_path)
    predictions = model.predict(img)[0]
    pred_class = np.argmax(predictions)
    pred_class = num_to_food[pred_class]
    return pred_class, predictions

img_path = 'test_food.png'
label, probabilities = make_prediction(img_path)
print("Predicted food:")
print(label)
print("Predicted probability:")
print(probabilities)
# Save preditions
df = pd.DataFrame({
    'Predicted': [label]
})

for class_index, class_name in num_to_food.items():
  df[f'Prob_{class_name}'] = probabilities[class_index]

df.to_csv('predictions_test.csv', index=True)