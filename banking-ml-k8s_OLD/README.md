# Banking Machine Learning Pipeline

This project is a machine learning pipeline designed for banking applications, deployed using Docker and Kubernetes. It includes various components for building, deploying, and managing the application in different environments.

## Project Structure

```
banking-ml-k8s
├── docker
│   ├── Dockerfile          # Instructions to build the Docker image
│   └── requirements.txt    # Python dependencies for the application
├── k8s
│   ├── base
│   │   ├── configmap.yaml  # ConfigMap for application configuration
│   │   ├── deployment.yaml  # Deployment resource for the application
│   │   ├── service.yaml     # Service resource to expose the application
│   │   └── ingress.yaml     # Ingress resource for external access
│   ├── overlays
│   │   ├── dev
│   │   │   ├── kustomization.yaml  # Kustomize for development overlay
│   │   │   └── patch-deployment.yaml # Patches for development deployment
│   │   └── prod
│   │       ├── kustomization.yaml  # Kustomize for production overlay
│   │       └── patch-deployment.yaml # Patches for production deployment
│   └── secrets
│       └── .gitignore       # Git ignore for sensitive files
├── scripts
│   ├── deploy.sh            # Script to deploy to Kubernetes
│   └── build.sh             # Script to build Docker image
├── helm
│   └── banking-ml
│       ├── Chart.yaml       # Helm chart metadata
│       ├── values.yaml      # Default values for Helm chart
│       ├── templates
│       │   ├── configmap.yaml # Helm template for ConfigMap
│       │   ├── deployment.yaml # Helm template for Deployment
│       │   ├── service.yaml    # Helm template for Service
│       │   └── ingress.yaml    # Helm template for Ingress
│       └── values-prod.yaml  # Production-specific values for Helm chart
├── ci
│   ├── .github
│   │   └── workflows
│   │       ├── build.yaml    # GitHub Actions workflow for building
│   │       └── deploy.yaml    # GitHub Actions workflow for deploying
│   └── Jenkinsfile           # Jenkins pipeline for CI/CD
├── docs
│   └── deployment.md         # Documentation for deployment process
├── kustomization.yaml         # Kustomize configuration for Kubernetes resources
└── README.md                 # Project documentation
```

## Getting Started

### Prerequisites

- Docker
- Kubernetes (Minikube, GKE, EKS, etc.)
- Helm
- Kustomize

### Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd banking-ml-k8s
   ```

2. **Build the Docker image:**
   ```
   cd docker
   ./build.sh
   ```

3. **Deploy to Kubernetes:**
   ```
   cd scripts
   ./deploy.sh
   ```

4. **Access the application:**
   Use the configured Ingress or Service to access the application.

### Usage

- The application can be accessed via the configured endpoints.
- Refer to the `docs/deployment.md` for detailed deployment instructions and configurations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.