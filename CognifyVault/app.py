from flask import Flask, render_template, request, redirect, url_for, flash, send_file, abort, session
import weaviate
from openai import OpenAI
import os
import re
import pdfplumber
import tempfile
import markdown
import secrets
import difflib
from datetime import datetime
from flask import request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from pathlib import Path

weaviate_server = os.getenv("WEAVIATE_SERVER", "http://localhost:8080")
target_class_name = os.getenv("ARTICLE_NAME", "Article")
llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
support_llm_model = os.getenv("SUPPORT_LLM_MODEL", "gpt-4o-mini")

file_headder = "file_"
file_analyze_headder = "analyze_"

LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'zh': 'Chinese',
}

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
            },
            {
                "name": "date",
                "dataType": ["date"],
            }
        ]
    }
    
    existing_classes = client.schema.get().get('classes', [])
    if not any(c['class'] == target_class_name for c in existing_classes):
        client.schema.create_class(class_obj)
        print(f"Class '{target_class_name}' created.")
    else:
        print(f"Class '{target_class_name}' already exists.")

def get_default_language():
    supported_languages = ['en', 'es', 'fr', 'de', 'ja', 'zh']

    accept_language = request.headers.get('Accept-Language')

    if accept_language:
        languages = [lang.split(';')[0] for lang in accept_language.split(',')]
        
        for lang in languages:
            lang_prefix = lang.split('-')[0]
            if lang_prefix in supported_languages:
                return lang_prefix
    
    return 'en'

@app.before_request
def set_language():
    if 'language' not in session:
        session['language'] = get_default_language()

@app.route('/set_language', methods=['POST'])
def set_language():
    language = request.form['language']
    if language in LANGUAGES:
        session['language'] = language
    return redirect(url_for('index'))

def get_system_role():
    language = session.get('language', 'en')
    system_role = f"You are an assistant responsible for communication in {LANGUAGES[language]}. Your role involves managing documents and preparing materials. You have a meticulous personality, prioritizing accuracy, and you provide responses that are both careful and nuanced."
    return system_role

@app.route('/')
def index():
    # Render the main index page.
    system_role = get_system_role()
    return render_template('index.html', system_role=system_role)

@app.route('/save_text', methods=['POST'])
def save_text():
    title = request.form['title']
    content = request.form['content']
    
    file = request.files.get('file')
    file_path = None

    directory = os.path.join(f"uploaded_files_{target_class_name}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if file:
        # If a file is uploaded, use the original file name
        filename = file.filename
        
        # Add a timestamp to the filename to avoid collisions
        base_name, ext = os.path.splitext(filename)
        unique_filename = f"{base_name}_{timestamp}{ext}"
        unique_filename_with_header = f"{file_headder}_{unique_filename}"
        file_path = os.path.join(directory, unique_filename_with_header)
        
        # Save the uploaded file to the directory
        file.save(file_path)

        if ext == ".pdf":
            file_content = read_file_content(file_path)
            analyze_filename_with_header = f"{file_analyze_headder}_{base_name}_{timestamp}.txt"
            analyze_file_path = os.path.join(directory, analyze_filename_with_header)
            analyze_text = analyze_text_format(file_content)
            with open(analyze_file_path, 'w', encoding='utf-8') as f:
                f.write(analyze_text)

    else:
        # If no file is uploaded, create a .txt file with the title as the name
        unique_filename = f"{file_headder}_{title}_{timestamp}.txt"
        file_path = os.path.join(directory, unique_filename)
        
        # Save the content as a .txt file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            content = summarize_text(content, file_path)
        except Exception as e:
            flash(f"Error occurred while saving the file: {e}", "error")
    
    # Get the current time in RFC3339 format
    current_time = datetime.now(timezone.utc).isoformat()

    # Add date information to the data object
    data_object = {
        "title": title,
        "content": f"{content}\n\nUpdated on Local Time: {current_time}",
        "file_path": file_path,
        "date": current_time
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
        summary = summarize_text(file_content, file_path)
        
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

@app.route('/check_title')
def check_title():
    title = request.args.get('title')
    existing_titles = client.query.get(target_class_name, ["title", "file_path"]).do().get('data', {}).get('Get', {}).get(target_class_name, [])
    for article in existing_titles:
        if article['title'] == title:
            return {'exists': True}, 200
    
    return {'exists': False}, 200

@app.route('/compare_similar_files', methods=['POST'])
def compare_similar_files():
    file = request.files.get('file')  # Retrieve the uploaded file

    if not file:
        return {'exists': False, 'message': 'No file uploaded or file not correctly received'}, 200

    temp_dir = tempfile.gettempdir()
    upload_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save file to temporary directory
        file.save(upload_file_path)
                    
        # Read the content of the uploaded file
        uploaded_file_content = read_file_content(upload_file_path)

        existing_titles = client.query.get(target_class_name, ["title", "file_path"]).do().get('data', {}).get('Get', {}).get(target_class_name, [])

        matching_file_count = 0
        comparison_info = ""
        for article in existing_titles:
            if article['title'] == file.filename:
                existing_file_path = article.get('file_path')
                if existing_file_path:
                    matching_file_count = matching_file_count + 1

                    existing_file_content = read_file_content(existing_file_path)
                    
                    # Compare the file contents and get the result
                    comparison_text = compare_file(os.path.getsize(existing_file_path), os.path.getsize(upload_file_path), existing_file_content, uploaded_file_content)
                    comparison_info += f"--- Compare Files No.{matching_file_count}\n{comparison_text}\n"

        if matching_file_count > 0:
            comparison_result = compare_files_with_llm(matching_file_count, comparison_info)
    
            # Construct a detailed message
            message = f"{file.filename} file already exists:\n" \
                    f"{comparison_result}\n" \
                    f"Would you like to prepare for the upload (summary)?"
            return {'exists': True, 'message': message}, 200

        return {'exists': False}, 200
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(upload_file_path):
            os.remove(upload_file_path)

def to_rfc3339(date_str, end_of_day=False):
    if end_of_day:
        return f"{date_str}T23:59:59Z"
    else:
        return f"{date_str}T00:00:00Z"

def search_articles(prompt):
    search_keywords = generate_search_keywords(prompt)

    if search_keywords is None:
        print("Failed to generate search keywords.")
        return None

    keywords = search_keywords.keywords
    objectives = search_keywords.objectives
    dates = search_keywords.dates

    try:
        # Initial query with keywords
        query = (
            client.query
            .get(target_class_name, ["title", "content", "file_path", "date"])
            .with_near_text({
                "concepts": keywords + objectives,
                "distance": 0.2  # The closer the value is to 0, the better it matches the keywords and objectives.
            })
        )

        # Add a date filter if dates are provided
        if dates:
            if len(dates) == 1:
                rfc3339_start_date = to_rfc3339(dates[0])
                rfc3339_end_date = to_rfc3339(dates[0], end_of_day=True)
                query = query.with_where({
                    "path": ["date"],
                    "operator": "GreaterThanEqual",
                    "valueDate": rfc3339_start_date
                }).with_where({
                    "path": ["date"],
                    "operator": "LessThanEqual",
                    "valueDate": rfc3339_end_date
                })
            elif len(dates) == 2:
                rfc3339_start_date = to_rfc3339(dates[0])
                rfc3339_end_date = to_rfc3339(dates[1], end_of_day=True)
                query = query.with_where({
                    "path": ["date"],
                    "operator": "GreaterThanEqual",
                    "valueDate": rfc3339_start_date
                }).with_where({
                    "path": ["date"],
                    "operator": "LessThanEqual",
                    "valueDate": rfc3339_end_date
                })
            else:
                pass

        result = query.with_limit(3).do()

        articles = result.get('data', {}).get('Get', {}).get(target_class_name, [])

        referenced_files = set()
        filtered_articles = []

        for article in articles:
            file_path = article.get('file_path')
            if not file_path:
                continue

            file_content = read_file_content(file_path)
            file_size = os.path.getsize(file_path)  # Substitute because file name comparison is not possible

            unique_file_identifier = f"{file_size}_{hash(file_content)}"
            if unique_file_identifier in referenced_files:
                continue

            referenced_files.add(unique_file_identifier)
            filtered_articles.append(article)

        return filtered_articles
    except Exception as e:
        print(f"Error occurred during search: {e}")
        return None

@app.route('/search', methods=['POST'])
def search():
    # Handle search requests and display relevant articles.
    prompt = request.form['prompt']
     
    # Get the current local time and format it
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Append the local time to the prompt with a label indicating it's local time
    prompt = f"{prompt}\n\nLocal Time: {current_time}"

    try:
        articles = search_articles(prompt)

        if not articles:
            flash("No search results found.", "info")
            return redirect(url_for('index'))

        # Generate a report using the search prompt and references text
        report_markdown = make_report(prompt, articles)
        report_html = markdown.markdown(report_markdown, extensions=['nl2br'])  # Convert Markdown to HTML

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
        elif file_extension.lower() == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension.lower() == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
                text = text.replace('-\n', '')
                text = re.sub(r'\s+', ' ', text)
                return text
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        print(f"Error reading file content for {file_path}: {e}")
        raise

def call_openai_api(model, contents=None, function_name="Unknown Function"):
    # Generalized function to call OpenAI API with given parameters.
    try:
        messages = []

        if contents:
            for i, content in enumerate(contents):
                if content is not None:
                    role = "system" if i % 2 == 0 else "user"
                    messages.append({"role": role, "content": content})

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error occurred during OpenAI API call in {function_name}: {e}")
        return None

def load_analyzed_file_content(file_path, file_headder, file_analyze_headder):
    path = Path(file_path)
    
    if path.name.startswith(file_headder):
        new_file_name = path.name.replace(file_headder, file_analyze_headder, 1)
        new_file_path = path.with_name(new_file_name).with_suffix('.txt')
        
        if new_file_path.exists():
            try:
                with new_file_path.open('r', encoding='utf-8', errors='ignore') as file:
                    data = file.read()
                return data
            except Exception as e:
                print(f"An error occurred while reading file: {e}")

    return None

def compare_texts(text1, text2):
    diff = list(difflib.ndiff(text1.splitlines(), text2.splitlines()))
    diff_info = "\n".join(diff)
    return diff_info

def compare_file(existing_size, uploaded_size, existing_content, uploaded_content):
    diff_info = compare_texts(existing_content, uploaded_content)
    diff_size = abs(existing_size - uploaded_size)
    comparison_test = (
        f"Existing file size: {existing_size} bytes\n"
        f"Uploaded file size: {uploaded_size} bytes\n"
        f"Difference in file size: {diff_size} bytes\n"
        f"Difference information:\n"
        f"{diff_info}\n"
    )
    return comparison_test

def compare_files_with_llm(matching_file_count, comparison_test):
    try:
        # Request the LLM to compare the files
        comparison_prompt = (
            "The user attempted to upload a file, but the system detected that a file with the same name is already registered."
            "The user needs to consider whether it is appropriate to register the same file again."
            "Based on the comparison information between the existing file and the file to be uploaded, please explain the differences between the existing file and the file to be uploaded.\n"
            "Note: This system allows multiple files with the same name to be registered (summaries can be set individually for each file), but if the same summary is registered, it may not be meaningful."
            "### Comparison Results Between the Existing File and the File to Be Uploaded\n"
            f"Number of existing files with the same name: {matching_file_count}\n"
            f"#### Differences in Content:\n"
            f"{comparison_test}\n"
            "### output\n"
            "(Mention that a file with the same name is already registered.)"
            "(Simply state whether there are differences or not without mentioning minor differences.)\n"
        )
        return call_openai_api(model=support_llm_model, contents=[get_system_role(), comparison_prompt], function_name="compare_files_with_llm")
    except Exception as e:
        return f"Error: An issue occurred while comparing the files: {str(e)}"

def analyze_text_format(text):
    # Generate a summary of the provided text using OpenAI.
    analyze_text_format_prompt = (
        f"## Instruction\n"
        f"Please list the characteristics of the sample text's format. Include considerations of layout elements, such as a two-column structure.\n"
        f"## Sample\n"
        f"{text}\n"
        f"## Output\n"
        f"(Analysis information only)\n"
    )
    return call_openai_api(support_llm_model, contents=[get_system_role(), analyze_text_format_prompt], function_name="analyze_text_format")

def summarize_text(text, file_path):
    # Generate a summary of the provided text using OpenAI.
    analyze_text_info = load_analyzed_file_content(file_path, file_headder, file_analyze_headder)
    if analyze_text_info:
        analyze_text_info = f"## Characteristics of the Support Materials\n{analyze_text_info}\n"
    else:
        if os.path.splitext(file_path)[1].lower() == '.pdf':
            analyze_text_info = analyze_text_format(text)
            analyze_text_info = f"## Characteristics of the Support Materials\n{analyze_text_info}\n"

    summarize_text_prompt = (
        f"## Instruction\n"
        f"Extract the following information from the document:\n"
        f"- Keywords that serve as an index for the document\n"
        f"- Summarize each chapter and organize the content into appropriate sections\n"
        f"## Document Characteristics\n"
        f"{analyze_text_info}"
        f"## Document\n"
        f"{text}"
        f"## Output\n"
        f"(Summary only)\n"
    )
    return call_openai_api(support_llm_model, contents=[get_system_role(), summarize_text_prompt], function_name="summarize_text")

def extract_and_organize_data(request_text, articles):
    report_contents = []

    for article in articles:
        file_path = article.get('file_path')
        if not file_path:
            continue

        file_content = read_file_content(file_path)

        analyze_text_info = load_analyzed_file_content(file_path, file_headder, file_analyze_headder)
        if analyze_text_info:
            analyze_text_info = f"## Characteristics of the Support Materials\n{analyze_text_info}\n"

        extract_matching_content = (
            f"## Instruction\n"
            f"Extract information from the support materials that is useful for the user's request, maintaining consistency.\n"
            f"## User Request\n"
            f"{request_text}\n"
            f"{analyze_text_info}"
            f"## Support Materials\n"
            f"{file_content}\n"
        )
        file_summary = call_openai_api(support_llm_model, contents=[get_system_role(), extract_matching_content], function_name="extract_and_organize_data")
        report_contents.append(f"### file:{os.path.basename(file_path)}\n{file_summary}\n")

    return report_contents

def make_report(request_text, articles):
    report_contents = extract_and_organize_data(request_text, articles)

    base_user_content = (
        f"## Instructions\n"
        f"Respond to the request in the requester's language based on the supporting materials provided.\n"
        f"## Request\n"
        f"{request_text}\n"
    )

    if report_contents:
        base_user_content += "## Supporting Materials\n" + "\n".join(report_contents)
    else:
        base_user_content += "## Supporting Materials\nNo relevant files were found."

    first_response = call_openai_api(llm_model, contents=[get_system_role(), base_user_content], function_name="make_report")
    
    last_response = None
    if first_response:
        proofread_user_content = (
            f"After critically reviewing your response, please proofread the text with the following points in mind:\n"
            f"- Are there any mistakes in numbers, names, or words?\n"
            f"- Are there any errors in translation?\n"
            f"- Does it align with the intended purpose?\n"
            f"- Is the format appropriate?\n"
            f"- Are there any omissions or missing elements?\n"
            f"Please write only the final proofread text."
        )
        last_response = call_openai_api(llm_model, contents=[get_system_role(), base_user_content, first_response, proofread_user_content], function_name="make_report")

    return last_response

def extract_user_intent(request_text):
    # Interprets the text to clarify what the user intends.
    user_content = (
        f"## Instruction\n"
        f"Interpret the following user request and clarify the user's intent (it's okay to make educated guesses).\n"
        f"Additionally, based on the user's intent, suggest any necessary perspectives to consider.\n"
        f"## User Request\n"
        f"{request_text}\n"
    )
    return call_openai_api(support_llm_model, contents=[get_system_role(), user_content], function_name="extract_user_intent")

class SearchKeywords(BaseModel):
    objectives: list[str]
    keywords: list[str]
    dates: Optional[list[str]]

def generate_search_keywords(prompt):
    user_content = (
        f"## Instruction\n"
        f"To fulfill the user request, query the Vector database. Generate an appropriate purpose and search keywords (including a date range if necessary).\n"
        f"Consider the date range as follows: `recent` implies about 3 days, and `latest` implies about 1 month.\n"
        f"After searching with the purpose and keywords, refine the search by date range, if specified. Therefore, if you specify a date range, there is no need to include date elements in the purpose and keywords.\n"
        f"## User request\n"
        f"{prompt}\n"
        f"## Output\n"
        f"List your objectives, keywords, and any relevant dates (if applicable), separated by commas.\n"
    )

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": get_system_role()},
            {"role": "user", "content": user_content},
        ],
        response_format=SearchKeywords,
    )
    return response.choices[0].message.parsed

if __name__ == '__main__':
    # Ensure the upload directory exists
    directory = os.path.join(f"uploaded_files_{target_class_name}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    create_weaviate_class()
    port = int(os.getenv("COGNIFY_VAULT_PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
