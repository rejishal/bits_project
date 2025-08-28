# Deployment Instructions for Banking ML Pipeline on Kubernetes

This document provides a step-by-step guide to deploying the Banking Machine Learning Pipeline application on a Kubernetes cluster.

## Prerequisites

Before you begin, ensure you have the following:

- A Kubernetes cluster (local or cloud-based).
- `kubectl` command-line tool installed and configured to interact with your cluster.
- Docker installed for building images.
- Access to a container registry (e.g., Docker Hub, Google Container Registry) to push your Docker images.

## Build the Docker Image

1. Navigate to the `docker` directory:

   ```bash
   cd docker
   ```

2. Build the Docker image using the provided `Dockerfile`:

   ```bash
   docker build -t your-docker-username/banking-ml:latest .
   ```

3. Push the Docker image to your container registry:

   ```bash
   docker push your-docker-username/banking-ml:latest
   ```

## Deploy to Kubernetes

### Using Kustomize

1. Navigate to the `k8s/overlays/dev` or `k8s/overlays/prod` directory depending on your environment:

   ```bash
   cd k8s/overlays/dev
   ```

2. Apply the Kustomize configuration:

   ```bash
   kubectl apply -k .
   ```

### Using Helm

1. Navigate to the Helm chart directory:

   ```bash
   cd helm/banking-ml
   ```

2. Install the Helm chart:

   ```bash
   helm install banking-ml .
   ```

   To upgrade an existing release, use:

   ```bash
   helm upgrade banking-ml .
   ```

## Accessing the Application

- If you have configured an Ingress resource, you can access the application using the specified hostname.
- If you are using a LoadBalancer service, retrieve the external IP:

   ```bash
   kubectl get services
   ```

## Monitoring and Logs

To check the status of your deployment, use:

```bash
kubectl get deployments
```

To view logs from the application pods, use:

```bash
kubectl logs <pod-name>
```

## Cleanup

To remove the deployed resources, use:

```bash
kubectl delete -k k8s/overlays/dev
```

Or if using Helm:

```bash
helm uninstall banking-ml
```

## Conclusion

You have successfully deployed the Banking Machine Learning Pipeline application on Kubernetes. For further customization and scaling, refer to the Kubernetes documentation and Helm chart values.