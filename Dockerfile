# Use an official Python runtime as a parent image
FROM python:3.8

# Install system-level dependencies
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /application

# Copy only the requirements file first for better caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the necessary port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Run the application with Flask's development server
CMD ["python", "application.py"]
