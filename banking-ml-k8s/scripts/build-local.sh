#!/bin/bash

# Build the Docker image for the Banking ML application
docker build -t banking-ml-app:latest -f docker/Dockerfile .

# Tag the image for the local Kind registry
docker tag banking-ml-app:latest localhost:5000/banking-ml-app:latest

# Push the image to the local Kind registry
docker push localhost:5000/banking-ml-app:latest