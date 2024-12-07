#!/bin/bash

python predict_food.py
echo "Food classification completed."

python food_to_nutrition.py
echo "Food to nutrition completed."

python nutrition_predict_disease.py
echo "Nutrition predict disease completed."

python llm-main.py
echo "LLM RAG completed."
