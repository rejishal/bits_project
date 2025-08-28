# Banking ML Kubernetes Deployment

This project provides a complete setup for deploying a Banking Machine Learning application using Kubernetes with Kind (Kubernetes in Docker). It includes Docker configurations, Kubernetes manifests, and scripts for building and deploying the application locally.

## Project Structure

```
banking-ml-k8s
├── docker
│   ├── Dockerfile          # Dockerfile for building the application image
│   └── requirements.txt    # Python dependencies for the application
├── k8s
│   ├── base
│   │   ├── configmap.yaml  # ConfigMap for application configuration
│   │   ├── deployment.yaml  # Deployment manifest for application pods
│   │   ├── service.yaml     # Service manifest for exposing the application
│   │   └── ingress.yaml     # Ingress manifest for routing external traffic
│   └── local-kind
│       ├── kind-config.yaml         # Kind cluster configuration
│       ├── local-registry.yaml      # Local Docker registry setup
│       └── ingress-setup.yaml       # Ingress controller setup
├── scripts
│   ├── setup-kind.sh       # Script to set up the Kind cluster
│   ├── build-local.sh      # Script to build the Docker image locally
│   └── deploy-local.sh     # Script to deploy the application to Kind
├── README.md               # Project documentation
└── .env.local              # Local environment variables
```

## Prerequisites

- Docker installed on your machine
- Kind installed for managing Kubernetes clusters
- kubectl installed for interacting with the Kubernetes cluster

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd banking-ml-k8s
   ```

2. **Build the Docker Image**
   Use the provided script to build the Docker image for the application.
   ```bash
   ./scripts/build-local.sh
   ```

3. **Set Up the Kind Cluster**
   Run the setup script to create a Kind cluster and configure the local registry.
   ```bash
   ./scripts/setup-kind.sh
   ```

4. **Deploy the Application**
   Deploy the application to the Kind cluster using the deployment script.
   ```bash
   ./scripts/deploy-local.sh
   ```

## Usage

After deployment, you can access the application through the configured Ingress. Make sure to check the Ingress rules defined in `k8s/base/ingress.yaml` for the correct host and path.

## Environment Variables

The `.env.local` file contains environment variables specific to your local setup. Ensure to configure it with the necessary values, such as API keys and database credentials.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.