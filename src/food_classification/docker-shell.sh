#!/bin/bash

set -e

export IMAGE_NAME=step1_food_classification
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/secrets/
export GCP_PROJECT="ac215-smarteat-437821"
export GCS_MODELS_BUCKET_NAME="ac215smarteat"

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-smarteat-437821-15c2f229e610.json
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_MODELS_BUCKET_NAME=$GCS_MODELS_BUCKET_NAME \
$IMAGE_NAME
