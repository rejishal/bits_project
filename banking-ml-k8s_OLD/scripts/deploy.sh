#!/bin/bash

# Set variables
NAMESPACE="banking-ml"
DEPLOYMENT_NAME="banking-ml-deployment"
K8S_BASE_DIR="./k8s/base"
K8S_OVERLAY_DIR="./k8s/overlays/dev"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE || echo "Namespace $NAMESPACE already exists."

# Apply base configurations
kubectl apply -f $K8S_BASE_DIR/configmap.yaml -n $NAMESPACE
kubectl apply -f $K8S_BASE_DIR/service.yaml -n $NAMESPACE
kubectl apply -f $K8S_BASE_DIR/deployment.yaml -n $NAMESPACE

# Apply overlay configurations
kubectl apply -k $K8S_OVERLAY_DIR -n $NAMESPACE

# Check the status of the deployment
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE

echo "Deployment completed successfully."