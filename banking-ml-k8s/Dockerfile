FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY docker/requirements.txt .

# Install build tools
RUN apt-get update && apt-get install -y build-essential

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY models/ /app/models/

# Expose the port the app runs on
EXPOSE 6000

# Set the environment variable for Python path
ENV PYTHONPATH=/app

# Set the command to run the application
CMD ["python", "src/api/app.py"]