# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

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
ENV ARTICLE_NAME=ArticleV2
## OpenAI API key
ENV OPENAI_API_KEY=
## LLM model for handling critical tasks
ENV LLM_MODEL=gpt-4o-mini
## LLM model for support tasks
ENV SUPPORT_LLM_MODEL=gpt-4o-mini
## Speech-to-text model
ENV SPEECH_TO_TEXT_MODEL=whisper-1
## Which determines the closeness of the match to the search keywords
ENV WEAVIATE_SEARCH_DISTANCE=0.2
## Limits the number of references returned in search results
ENV WEAVIATE_SEARCH_LIMIT=3

# Run the application
CMD ["python", "app.py"]
