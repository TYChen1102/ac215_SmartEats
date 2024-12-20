#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set vairables
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="ac215-smarteat-437821" # CHANGE TO YOUR PROJECT ID
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/llm-service-account.json"
export IMAGE_NAME="llm-rag-cli"


# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME .

# Run All Containers
docker-compose run --rm --service-ports $IMAGE_NAME
