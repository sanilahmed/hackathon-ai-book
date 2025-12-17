# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI application with uvicorn for Hugging Face Spaces
CMD ["uvicorn", "rag_agent_api.main:app", "--host", "0.0.0.0", "--port", "7860"]