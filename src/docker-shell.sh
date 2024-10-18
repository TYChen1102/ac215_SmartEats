#!/bin/bash

# Step 1: Run the food classification container
docker build -t food_classification ./food-classification
docker run --rm -ti food_classification
# Wait for Step 1 to complete (You can check logs or specific conditions as needed)
echo "Step 1 completed."


# Step 2: Run the food to nutrition container
docker build -t food_to_nutrition ./food_to_nutrition
docker run --rm -ti food_to_nutrition
echo "Step 2 completed."

# Step 2: Run the food to nutrition container
docker build -t nutrition_predict_disease ./nutrition_predict_disease
docker run --rm -ti nutrition_predict_disease
echo "Step 3 completed."

# Completion message
echo "The first steps have been completed."
