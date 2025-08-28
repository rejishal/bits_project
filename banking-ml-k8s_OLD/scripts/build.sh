#!/bin/bash

# Set variables
IMAGE_NAME="banking-ml"
IMAGE_TAG="latest"
DOCKERFILE_PATH="./docker/Dockerfile"
REGISTRY="your-container-registry"  # Replace with your container registry

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG -f $DOCKERFILE_PATH .

# Tag the image for the registry
echo "Tagging the image..."
docker tag $IMAGE_NAME:$IMAGE_TAG $REGISTRY/$IMAGE_NAME:$IMAGE_TAG

# Push the image to the container registry
echo "Pushing the image to the registry..."
docker push $REGISTRY/$IMAGE_NAME:$IMAGE_TAG

echo "Build and push completed successfully."