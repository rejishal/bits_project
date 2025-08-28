#!/bin/bash

# This script sets up a local Kubernetes cluster using Kind and configures a local Docker registry.

# Create a Kind cluster
kind create cluster --config k8s/local-kind/kind-config.yaml

# Create a local Docker registry
kubectl apply -f k8s/local-kind/local-registry.yaml

# Set up Ingress controller
kubectl apply -f k8s/local-kind/ingress-setup.yaml

# Display cluster information
kubectl cluster-info

# Display nodes
kubectl get nodes

# Display all pods in the kube-system namespace to verify Ingress setup
kubectl get pods -n kube-system