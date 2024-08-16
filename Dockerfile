# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5000

# Environment variables
## Address of weaviate
ENV WEAVIATE_SERVER=http://weaviate:8080
## Public port number of CognifyVault
ENV COGNIFY_VAULT_PORT=5000
## Storage destination for knowledge in WEAVIATE
ENV ARTICLE_NAME=Article
## OpenAI API key
ENV OPENAI_API_KEY=
## LLM model for handling critical tasks
ENV LLM_MODEL=gpt-4o-mini
## LLM model for support tasks
ENV SUPPORT_LLM_MODEL=gpt-4o-mini

# Run the application
CMD ["python", "app.py"]
