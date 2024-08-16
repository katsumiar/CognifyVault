[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/katsumiar/CognifyVault/blob/main/LICENSE)

![image](https://github.com/user-attachments/assets/b26e440e-69eb-4e3c-8875-ae36e4eb77e2)

# CognifyVault

Register knowledge and files in a vector database, and generate information based on questions using the OpenAI API. This knowledge management and search support tool efficiently searches for relevant knowledge through vector search powered by Weaviate.

## Features
- **Knowledge Registration**: Register knowledge by directly entering text or uploading files (supports `.txt` and `.pdf` formats).
- **Knowledge Extraction**: Ask questions and get responses based on the registered knowledge.
- **File Summarization**: Automatically generate summaries for uploaded files using the OpenAI API.
- **Vector Search**: Efficiently search through knowledge using Weaviate's vector-based search capabilities.

## Supported File Formats
- **Text Files (`.txt`)**
- **PDF Files (`.pdf`)**

## Prerequisites
- Docker
- Docker Compose
- OpenAI API Key

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/katsumiar/cognifyvault.git
   cd cognifyvault
   ```

2. **Set Your OpenAI API Key**
   - Open the `Dockerfile` and set your OpenAI API key in the line `ENV OPENAI_API_KEY=`.
   - Example: `ENV OPENAI_API_KEY=your_openai_api_key_here`

3. **Build the Docker Image**
   ```bash
   docker-compose build
   ```

4. **Start the Docker Container**
   ```bash
   docker-compose up -d
   ```

5. **Access the Application**
   - Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage

### Registering Knowledge
- On the `CognifyVault` page, use the "Register your knowledge" section to enter a title and content directly or upload a file (`.txt` or `.pdf`) to register your knowledge.

### Extracting Knowledge
- Use the "Extract knowledge" section to ask questions and receive responses based on the registered knowledge.

## Environment Variables
- `WEAVIATE_SERVER`: The URL for the Weaviate server. Default is `http://localhost:8080`.
- `ARTICLE_NAME`: The class name used in Weaviate for storing articles. Default is `Article`.
- `LLM_MODEL`: The name of the OpenAI model to be used. Default is `gpt-4o-mini`.
- `SUPPORT_LLM_MODEL`: The support model name for OpenAI. Default is `gpt-4o-mini`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
