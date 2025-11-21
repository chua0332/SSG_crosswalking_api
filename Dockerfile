# Use official Python image
FROM python:3.10-slim

# Create working directory
WORKDIR /app

# Install OS dependencies (optional but helpful)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY required data files
COPY ./data/ssg_skills.csv ./data/ssg_skills.csv
COPY ./data/ssg_skill_embeddings.pt ./data/ssg_skill_embeddings.pt

# Copying in the src folder stuffs
COPY ./src ./src

# Copying the main file
COPY ./main.py ./main.py

# Expose port FastAPI will run on
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
