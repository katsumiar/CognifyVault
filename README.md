[![Go to Wiki](https://img.shields.io/badge/Go%20to-Wiki-brightgreen)](https://github.com/katsumiar/CognifyVault/wiki)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/katsumiar/CognifyVault/blob/main/LICENSE)

![image](https://github.com/user-attachments/assets/913b2c45-54de-489f-a08b-22fcc15eedd6)

# CognifyVault

Register knowledge and files in a vector database, and generate information based on questions using the OpenAI API. This knowledge management and search support tool efficiently searches for relevant knowledge through vector search powered by Weaviate.

# Functionality Improvements
- **Title Warning**: When registering or editing a title, if the entered title matches an existing one, a red warning is displayed to alert the user of the duplication.
- **Duplicate Title Check**: Before saving a new title, the system checks for any existing titles with the same name and prompts the user to confirm if a duplicate is found.
- **Duplicate File Check**: When uploading a file, the system checks if the content matches an existing file and warns the user of the duplication before proceeding.
- **Improved PDF Text Extraction**: Enhanced the accuracy of text extraction from PDF files by removing unnecessary line breaks and spaces.
- **Enhanced AI Prompts**: Optimized the interaction with OpenAI API, leading to more accurate and relevant responses based on user queries.
- **Optimized Reference Handling**: Improved the consistency and accuracy of search results by preventing the referencing of duplicate files.

## Features
- **Knowledge Registration**: Register knowledge by directly entering text or uploading files (supports `.txt`, `.pdf`, and `.md` formats).
  - **Title Warning**: When registering or editing a title, if the entered title matches an existing one, a red warning is displayed to alert the user of the duplication.
  - **Duplicate Title Check**: Before saving a new title, the system checks for any existing titles with the same name and prompts the user to confirm if a duplicate is found.
- **Knowledge Extraction**: Ask questions and get responses based on the registered knowledge.
  - **Enhanced AI Prompts**: Optimized the interaction with OpenAI API, leading to more accurate and relevant responses based on user queries.
- **File Summarization**: Automatically generate summaries for uploaded files using the OpenAI API.
  - **Duplicate File Check**: When uploading a file, the system checks if the content matches an existing file and warns the user of the duplication before proceeding.
  - **Improved PDF Text Extraction**: Enhanced the accuracy of text extraction from PDF files by removing unnecessary line breaks and spaces.
- **Vector Search**: Efficiently search through registered knowledge using vector search powered by Weaviate.
  - **Optimized Reference Handling**: Improved the consistency and accuracy of search results by preventing the referencing of duplicate files.

## Report Generation Feature

This application offers an advanced report generation feature that creates detailed reports based on user requests. Unlike simple text generation, this feature intelligently analyzes the provided documents and articles to produce reports that align closely with the user's intent.

- 1. Information Extraction Using Vector Search
First, the application automatically extracts relevant information from the provided materials (such as articles or documents). This process utilizes vector search technology, which considers the semantic relationships between words and phrases, ensuring that the most relevant content is selected in response to the user's request.

- 2. Understanding and Reflecting User Intent
Next, the application interprets the user's request to understand their intent. This step goes beyond surface-level processing and delves into what the user is truly asking for, ensuring that the report is constructed in a way that accurately reflects the user's needs.

- 3. Report Generation and Proofreading
Based on the extracted information and the interpreted user intent, the application generates a report. The generated report is then further proofread to verify the accuracy of numbers, names, translation quality, and the appropriateness of the format. This process ensures that the final document is of high quality.

- 4. Accuracy and Cost
This approach involves multiple invocations of large language models (LLMs), which increases processing costs. However, the precision and quality of the resulting reports are significantly enhanced, meeting the user's expectations. While the cost is higher, the end result is a highly reliable document.

## Language Switching

The application supports multiple languages to enhance user experience. By default, the application will detect and select the language based on your browser's locale settings. However, you can manually switch to a different language using the following steps:

1. **Access the Language Dropdown**: On the main page, locate the language selection dropdown at the top of the page.
2. **Select Your Preferred Language**: Click on the dropdown menu and choose your preferred language from the list of available options.
3. **Automatic Refresh**: After selecting a language, the page will automatically refresh to apply the language change.
4. **Persistence**: The language setting will be saved for your session, so it will remain in your selected language as long as the session is active.

### Supported Languages

The application currently supports the following languages:
- English (`en`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Japanese (`ja`)
- Chinese (`zh`)

If your preferred language is not listed, the default language is English (`en`).

## Supported File Formats
- **Text Files (`.txt`)**
- **PDF Files (`.pdf`)**
- **Markdown Files (`.md`)**

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
