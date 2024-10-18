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

# Step 3: Run the food to nutrition container
docker build -t nutrition_predict_disease ./nutrition_predict_disease
docker run --rm -ti nutrition_predict_disease
echo "Step 3 completed."

# Step 4: Run the RAG container
set -e

# Set vairables
export BASE_DIR=$(pwd)
#export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/secrets/
export GCP_PROJECT="ac215-smarteat-437821" # CHANGE TO YOUR PROJECT ID
#export GOOGLE_APPLICATION_CREDENTIALS="/secrets/llm-service-account.json"
#export GOOGLE_APPLICATION_CREDENTIALS="$SECRETS_DIR/ac215-smarteat-437821-7bf746ea309f.json"
export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/secrets/llm-service-account.json
export IMAGE_NAME="llm-rag-cli"


# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME ./llm-rag

# Run All Containers
#docker-compose run --rm --service-ports $IMAGE_NAME

#docker build -t nutrition_predict_disease ./nutrition_predict_disease
docker run --rm -ti $IMAGE_NAME
echo "Step 4 completed."

# Completion message
echo "The first steps have been completed."
