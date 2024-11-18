#!/bin/bash

set -e

export IMAGE_NAME=step1_food_classification
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/secrets/
export GCP_PROJECT="ac215-smarteat-437821"
export GCS_MODELS_BUCKET_NAME="ac215smarteat"

# Build the image
docker build -t $IMAGE_NAME -f Dockerfile .

# Debugging: Verify IMAGE_NAME
echo "Using image: $IMAGE_NAME"

# Run Container
docker run --rm --name $IMAGE_NAME -i \
  -v "$BASE_DIR":/app \
  -v "$SECRETS_DIR":/secrets \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-smarteat-437821-15c2f229e610.json \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e GCS_MODELS_BUCKET_NAME=$GCS_MODELS_BUCKET_NAME \
  $IMAGE_NAME bash -c "mkdir -p /app && chmod -R 777 /app && exec bash"
