from flask import Flask, render_template, request, redirect, url_for, flash, send_file, abort
import weaviate
from openai import OpenAI
import os
import pdfplumber
import tempfile
import markdown
import secrets
from datetime import datetime

weaviate_server = os.getenv("WEAVIATE_SERVER", "http://localhost:8080")
target_class_name = os.getenv("ARTICLE_NAME", "Article")
llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
support_llm_model = os.getenv("SUPPORT_LLM_MODEL", "gpt-4o-mini")
system_role = "You are an assistant that manages documents."

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Set up Weaviate client
client = weaviate.Client(
    url=weaviate_server,
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    },
)

openai_client = OpenAI() # os.getenv("OPENAI_API_KEY")

def create_weaviate_class():
    # Create a class in Weaviate to store articles if it doesn't already exist.
    class_obj = {
        "class": target_class_name,
        "description": "A class to store articles",
        "vectorizer": "text2vec-openai",  # Specify the embedding model
        "properties": [
            {
                "name": "title",
                "dataType": ["string"],
            },
            {
                "name": "content",
                "dataType": ["text"],
            },
            {
                "name": "file_path",
                "dataType": ["string"],
            }
        ]
    }
    
    existing_classes = client.schema.get().get('classes', [])
    if not any(c['class'] == target_class_name for c in existing_classes):
        client.schema.create_class(class_obj)
        print(f"Class '{target_class_name}' created.")
    else:
        print(f"Class '{target_class_name}' already exists.")

@app.route('/')
def index():
    # Render the main index page.
    return render_template('index.html')

@app.route('/save_text', methods=['POST'])
def save_text():
    title = request.form['title']
    content = request.form['content']
    
    file = request.files.get('file')
    file_path = None

    directory = os.path.join(f"uploaded_files_{target_class_name}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    if file:
        # If a file is uploaded, use the original file name
        filename = file.filename
        
        # Add a timestamp to the filename to avoid collisions
        base_name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{base_name}_{timestamp}{ext}"
        file_path = os.path.join(directory, unique_filename)
        
        # Save the uploaded file to the directory
        file.save(file_path)
    else:
        # If no file is uploaded, create a .txt file with the title as the name
        filename = f"{title}.txt"
        file_path = os.path.join(directory, filename)
        
        # Check if the file already exists
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{title}_{timestamp}.txt"
            file_path = os.path.join(directory, unique_filename)
        
        # Save the content as a .txt file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            content = summarize_text(content)
        except Exception as e:
            flash(f"Error occurred while saving the file: {e}", "error")
    
    data_object = {
        "title": title,
        "content": content,
        "file_path": file_path
    }
    
    try:
        # Store the data object in Weaviate
        client.data_object.create(
            data_object=data_object,
            class_name=target_class_name
        )
        flash("Text and file saved successfully.", "success")
    except Exception as e:
        flash(f"Error occurred: {e}", "error")
    
    return redirect(url_for('index'))

@app.route('/summarize_upload_file', methods=['POST'])
def summarize_upload_file():
    # File upload and return its summary.
    file = request.files.get('file')
    if not file:
        return {'error': 'No file provided'}, 400
    
    # Get temporary directory path
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save file to temporary directory
        file.save(file_path)

        # Read the content of the PDF file
        file_content = read_file_content(file_path)

        # Generate summary using OpenAI API
        summary = summarize_text(file_content)
        
        return {'summary': summary}, 200
    except Exception as e:
        return {'error': f'Error occurred while generating summary: {e}'}, 500
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/download_file')
def download_file():
    # Allow users to download a specific file.
    file_path = request.args.get('file_path')
    
    if file_path:
        safe_path = os.path.join(f"uploaded_files_{target_class_name}", os.path.basename(file_path))
        
        if os.path.exists(safe_path):
            try:
                return send_file(safe_path, as_attachment=True)
            except Exception as e:
                flash(f"Error occurred while sending file: {e}", "error")
                return redirect(url_for('index'))
        else:
            flash("The specified file does not exist.", "error")
            return redirect(url_for('index'))
    else:
        flash("No file path specified.", "error")
        return redirect(url_for('index'))

@app.route('/search', methods=['POST'])
def search():
    # Handle search requests and display relevant articles.
    prompt = request.form['prompt']
    
    ward = generate_search_keywords(prompt)
    try:
        result = (
            client.query
            .get(target_class_name, ["title", "content", "file_path"])
            .with_near_text({
                "concepts": [ward],
                "distance": 0.2
            })
            .do()
        )

        articles = result.get('data', {}).get('Get', {}).get(target_class_name, [])

        if not articles:
            flash("No search results found.", "info")
            return redirect(url_for('index'))
        
        supplementary_text = "## Supplementary Materials\n"
        for article in articles:
            file_path = article.get('file_path')
            if file_path:
                file_content = read_file_content(file_path)
                file_name = os.path.basename(file_path)
                supplementary_text += f"### {file_name}\n{file_content}\n"

        # Generate a report using the search prompt and supplementary materials
        report_markdown = make_report(prompt, supplementary_text)
        report_html = markdown.markdown(report_markdown)  # Convert Markdown to HTML

        return render_template('index.html', articles=articles, report=report_html)
    except Exception as e:
        flash(f"Error occurred during search: {e}", "error")
        return redirect(url_for('index'))

def read_file_content(file_path):
    # Read the content of a file based on its extension.
    _, file_extension = os.path.splitext(file_path)
    
    try:
        if file_extension.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension.lower() == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
                return text
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        print(f"Error reading file content for {file_path}: {e}")
        raise

def summarize_text(text):
    # Generate a summary of the provided text using OpenAI.
    try:
        response = openai_client.chat.completions.create(
            model=support_llm_model,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"# Instructions\nPlease extract the following information from the document below:\n- Keywords that serve as an index for the document\n- Summarize each chapter, organizing the content into appropriate sections\n# Document\n{text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error occurred while generating summary: {e}")
        return None

def make_report(request_text, references_text):
    # Generate a report based on user request and supplementary materials using OpenAI.
    try:
        user_intent = extract_user_intent(request_text)
        if user_intent:
            response = openai_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": f"## Instructions\nRespond to the request in the requester's language based on the supporting materials provided. \n## Request\n{request_text}\n## User Intent\n{user_intent}\n## Supporting Materials\n{references_text}"}
                ]
            )
            return response.choices[0].message.content.strip()
        else:
            response = openai_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": f"## Instructions\nI will respond to the request in the requester's language based on the supporting materials provided, ensuring that no important requirements or elements are omitted.\n## Request\n{request_text}\n## Supporting Materials\n{references_text}"}
                ]
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error occurred while generating repot: {e}")
        return None

def extract_user_intent(request_text):
    # Interprets the text to clarify what the user intends.
    try:
        response = openai_client.chat.completions.create(
            model=support_llm_model,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"Please interpret the following user request and clarify the user's intent. It is okay to make educated guesses.\n## User Request\n{request_text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error occurred while generating user intent: {e}")
        return None

def generate_search_keywords(prompt):
    # Generate search keywords using OpenAI for vector database query.
    try:
        response = openai_client.chat.completions.create(
            model=support_llm_model,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"Generate suitable search keywords for querying a vector database to accomplish the following request. Separate multiple keywords with commas.\n---\n{prompt}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error occurred while generating search ward: {e}")
        return None

if __name__ == '__main__':
    # Ensure the upload directory exists
    directory = os.path.join(f"uploaded_files_{target_class_name}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    create_weaviate_class()
    port = int(os.getenv("COGNIFY_VAULT_PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
