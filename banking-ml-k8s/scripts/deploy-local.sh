#!/bin/bash

# This script deploys the banking ML application to a local Kind Kubernetes cluster.

# Set the Kubernetes context to the Kind cluster
kubectl config use-context kind-kind

# Apply the ConfigMap, Deployment, Service, and Ingress resources
kubectl apply -f k8s/base/configmap.yaml
kubectl apply -f k8s/base/deployment.yaml
kubectl apply -f k8s/base/service.yaml
kubectl apply -f k8s/base/ingress.yaml

echo "Deployment to local Kind cluster completed."