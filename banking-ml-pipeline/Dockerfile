# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p models logs data outputs

# Expose port for API
EXPOSE 5000

# Default command (can be overridden)
CMD ["python", "-m", "src.api.app"]

---

# docker-compose.yml
version: '3.8'

services:
  banking-api:
    build: .
    container_name: banking-ml-api
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - FLASK_ENV=production
      - API_HOST=0.0.0.0
      - API_PORT=5000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  training:
    build: .
    container_name: banking-ml-training
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./data:/app/data
    command: ["python", "scripts/train_model.py", "--synthetic", "--n_samples", "10000"]
    profiles:
      - training

  monitoring:
    image: prom/prometheus:latest
    container_name: banking-ml-monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    profiles:
      - monitoring

---

# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.gitignore
.mypy_cache
.pytest_cache
.hypothesis
notebooks/.ipynb_checkpoints
*.ipynb_checkpoints
.DS_Store
*.egg-info/
dist/
build/