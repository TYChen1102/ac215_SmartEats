#!/bin/bash
# python service.py

cd food-classification
python predict_food.py
echo "Food classification completed."

cd ../food_to_nutrition
python food_to_nutrition.py
echo "Food to nutrition completed."

cd ../nutrition_predict_disease
python nutrition_predict_disease.py
echo "Nutrition predict disease completed."

cd ..
