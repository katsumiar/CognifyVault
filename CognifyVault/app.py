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
from pydub import AudioSegment
import subprocess

WEAVIATE_SERVER = os.getenv("WEAVIATE_SERVER", "http://localhost:8080")
TARGET_CLASS_NAME = os.getenv("ARTICLE_NAME", "Article")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
SUPPORT_LLM_MODEL = os.getenv("SUPPORT_LLM_MODEL", "gpt-4o-mini")
SPEECH_TO_TEXT_MODEL = os.getenv("SPEECH_TO_TEXT_MODEL", "whisper-1")

WEAVIATE_SEARCH_DISTANCE = float(os.getenv("WEAVIATE_SEARCH_DISTANCE", "0.2"))
WEAVIATE_SEARCH_LIMIT = int(os.getenv("WEAVIATE_SEARCH_LIMIT", "3"))

FILE_HEADER = "file_"
FILE_ANALYZE_HEADER = "analyze_"
TRANSCRIBED_HEADER = "Transcribed_"

TEXT_EXTENSIONS = {'.txt', '.md' }
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.flv', '.wmv'}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | {'.pdf'}

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
    url = WEAVIATE_SERVER,
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    },
)

openai_client = OpenAI() # os.getenv("OPENAI_API_KEY")

def create_weaviate_class():
    # Create a class in Weaviate to store articles if it doesn't already exist.
    class_obj = {
        "class": TARGET_CLASS_NAME,
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
    if not any(c['class'] == TARGET_CLASS_NAME for c in existing_classes):
        client.schema.create_class(class_obj)
        print(f"Class '{TARGET_CLASS_NAME}' created.")
    else:
        print(f"Class '{TARGET_CLASS_NAME}' already exists.")

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

def is_text_file(extension):
    # Determines if the file extension is one of the supported text file types.
    return extension.lower() in TEXT_EXTENSIONS

def is_supported_file(extension):
    # Determines if the file extension is one of the supported types.
    return extension.lower() in SUPPORTED_EXTENSIONS

def is_audio_file(extension):
    # Determines if the file is an audio file
    return extension.lower() in AUDIO_EXTENSIONS

def is_video_file(extension):
    # Determines if the file is a video file
    return extension.lower() in VIDEO_EXTENSIONS

@app.route('/supported_extensions', methods=['GET'])
def supported_extensions():
    # Returns a list of supported file extensions.
    return {"supported_extensions": list(SUPPORTED_EXTENSIONS)}, 200

@app.before_request
def initialize_language():
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
    supported_extensions = list(SUPPORTED_EXTENSIONS)
    return render_template('index.html', system_role=system_role, supported_extensions=supported_extensions)

@app.route('/save_text', methods=['POST'])
def save_text():
    title = request.form['title']
    content = request.form['content']
    upload_content = request.form['upload_content']
    
    file = request.files.get('file')
    file_path = None

    directory = os.path.join(f"uploaded_files_{TARGET_CLASS_NAME}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if file and file.filename:
        # If a file is uploaded, use the original file name
        filename = file.filename
        
        # Add a timestamp to the filename to avoid collisions
        base_name, ext = os.path.splitext(filename)

        if not is_supported_file(ext):
            flash("Unsupported file format.", "error")
            return redirect(url_for('index'))

        unique_filename = f"{base_name}_{timestamp}{ext}"
        unique_filename_with_header = f"{FILE_HEADER}_{unique_filename}"
        file_path = os.path.join(directory, unique_filename_with_header)
        
        # Save the uploaded file to the directory
        file.save(file_path)

        if ext == ".pdf":
            file_content = read_file_content(file_path)
            analyze_filename_with_header = f"{FILE_ANALYZE_HEADER}_{base_name}_{timestamp}.txt"
            analyze_file_path = os.path.join(directory, analyze_filename_with_header)
            analyze_text = analyze_text_format(file_content)
            if analyze_text:
                with open(analyze_file_path, 'w', encoding='utf-8') as f:
                    f.write(analyze_text)
        
        if is_audio_file(ext) or is_video_file(ext):
            transcribed_filename_with_header = f"{TRANSCRIBED_HEADER}_{base_name}_{timestamp}.txt"
            transcribed_file_path = os.path.join(directory, transcribed_filename_with_header)
            if upload_content:
                with open(transcribed_file_path, 'w', encoding='utf-8') as f:
                    f.write(upload_content)

    else:
        # If no file is uploaded, create a .txt file with the title as the name
        unique_filename = f"{FILE_HEADER}_{title}_{timestamp}.txt"
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
            class_name=TARGET_CLASS_NAME
        )
        flash("Text and file saved successfully.", "success")
    except Exception as e:
        flash(f"Error occurred: {e}", "error")
    
    return redirect(url_for('index'))

@app.route('/summarize_upload_file', methods=['POST'])
def summarize_upload_file():
    # File upload and return its summary.
    file = request.files.get('file')
    if not file or not file.filename:
        return {'error': 'No file provided'}, 400
    
    # Get temporary directory path
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save file to temporary directory
        file.save(file_path)

        # Read the content of the PDF file
        file_content = read_file_content(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        upload_content = None
        if is_audio_file(ext) or is_video_file(ext):
            upload_content = file_content.replace(' ', '\n')

        # Generate summary using OpenAI API
        summary = summarize_text(file_content, file_path)
        
        return {
            'summary': summary,
            'upload_content': upload_content
        }, 200
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
        safe_path = os.path.join(f"uploaded_files_{TARGET_CLASS_NAME}", os.path.basename(file_path))
        
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
    existing_titles = client.query.get(TARGET_CLASS_NAME, ["title", "file_path"]).do().get('data', {}).get('Get', {}).get(TARGET_CLASS_NAME, [])
    for article in existing_titles:
        if article['title'] == title:
            return {'exists': True}, 200
    
    return {'exists': False}, 200

@app.route('/compare_similar_files', methods=['POST'])
def compare_similar_files():
    file = request.files.get('file')  # Retrieve the uploaded file

    if not file or not file.filename:
        return {'exists': False, 'message': 'No file uploaded or file not correctly received'}, 200

    temp_dir = tempfile.gettempdir()
    upload_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save file to temporary directory
        file.save(upload_file_path)
                    
        # Read the content of the uploaded file
        uploaded_file_content = read_file_content(upload_file_path)

        existing_titles = client.query.get(TARGET_CLASS_NAME, ["title", "file_path"]).do().get('data', {}).get('Get', {}).get(TARGET_CLASS_NAME, [])

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
            .get(TARGET_CLASS_NAME, ["title", "content", "file_path", "date"])
            .with_near_text({
                "concepts": keywords + objectives,
                "distance": WEAVIATE_SEARCH_DISTANCE # The closer the value is to 0, the better it matches the keywords and objectives.
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

        result = query.with_limit(WEAVIATE_SEARCH_LIMIT).do()

        articles = result.get('data', {}).get('Get', {}).get(TARGET_CLASS_NAME, [])

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
        if not report_markdown:
            flash("Failed to generate the report.", "error")
            return redirect(url_for('index'))
        report_html = markdown.markdown(report_markdown, extensions=['nl2br'])  # Convert Markdown to HTML

        supported_extensions = list(SUPPORTED_EXTENSIONS)
        return render_template('index.html', articles=articles, report=report_html, supported_extensions=supported_extensions)
    except Exception as e:
        flash(f"Error occurred during search: {e}", "error")
        return redirect(url_for('index'))

def read_file_content(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    try:
        if is_text_file(file_extension):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif is_audio_file(file_extension) or is_video_file(file_extension):
            text = load_subfile_content(file_path, FILE_HEADER, TRANSCRIBED_HEADER)
            if text:
                return text
            text = transcribe_audio(file_path, "read_file_content")
            # text = transcription_correction(text)
            return text
        elif file_extension == '.pdf':
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
        if not response.choices[0].message.content:
            print(f"No content returned from OpenAI API in {function_name}")
            return None
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error occurred during OpenAI API call in {function_name}: {e}")
        return None

def convert_video_to_mp3(video_path):
    # Generate a temporary MP3 file
    mp3_path = tempfile.mktemp(suffix=".mp3")
    try:
        # Convert video to mp3 using ffmpeg command
        command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', mp3_path]
        subprocess.run(command, check=True)
        return mp3_path
    except Exception as e:
        # Ensure temporary MP3 file is deleted if conversion fails
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        raise e

def transcribe_audio(file_path, function_name="Unknown Function"):
    temp_files = []
    try:
        # Get the file extension to determine if it is a video
        _, extension = os.path.splitext(file_path)
        
        if is_video_file(extension):
            # Convert video files to MP3
            file_path = convert_video_to_mp3(file_path)
        
        language = session.get('language', 'en')

        audio = AudioSegment.from_file(file_path)
        
        ten_minutes = 10 * 60 * 1000  # Calculate 10 minutes in milliseconds
        total_duration = len(audio)  # Get the total length of the audio (in milliseconds)
        
        transcripts = []

        for start_time in range(0, total_duration, ten_minutes):
            end_time = min(start_time + ten_minutes + 5000, total_duration)
            segment = audio[start_time:end_time]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                segment.export(temp_file.name, format="mp3")
                temp_files.append(temp_file.name)

        for temp_file in temp_files:
            try:
                with open(temp_file, 'rb') as audio_file:
                    transcript = openai_client.audio.transcriptions.create(
                        model=SPEECH_TO_TEXT_MODEL,
                        file=audio_file,
                        prompt="Improve your reading by removing anything that is not necessary for understanding the conversation and correcting any typos or words that you may have misheard.",
                        language=language
                    )
                    transcripts.append(transcript.text)
            except Exception as api_error:
                print(f"API call failed for file {temp_file}: {api_error}")
        
        # Combine all text
        full_transcript = " ".join(transcripts)

        return full_transcript

    except Exception as e:
        print(f"Error occurred during OpenAI API call in {function_name}: {e}")
        return f"{e}"
    
    finally:
        # Ensure all temporary files are deleted
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError as e:
                print(f"Error removing temporary file {temp_file}: {e}")
        
        # If a video file was converted to MP3, delete the temporary MP3 file
        if is_video_file(extension):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error removing temporary MP3 file {file_path}: {e}")

def load_subfile_content(file_path, FILE_HEADER, subfile_type):
    path = Path(file_path)
    
    if path.name.startswith(FILE_HEADER):
        new_file_name = path.name.replace(FILE_HEADER, subfile_type, 1)
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
        return call_openai_api(model=SUPPORT_LLM_MODEL, contents=[get_system_role(), comparison_prompt], function_name="compare_files_with_llm")
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
    return call_openai_api(SUPPORT_LLM_MODEL, contents=[get_system_role(), analyze_text_format_prompt], function_name="analyze_text_format")

def transcription_correction(text):
    transcription_correction_prompt = (
        f"## Instruction\n"
        f"Please correct and improve the transcribed text from the audio file by fixing any typographical errors or likely misheard words to make it more readable.\n"
        f"## Transcribed Text\n"
        f"{text}"
        f"## Output\n"
        f"(Corrected Text)\n"
    )
    return call_openai_api(SUPPORT_LLM_MODEL, contents=[get_system_role(), transcription_correction_prompt], function_name="summarize_text")

def summarize_text(text, file_path):
    # Generate a summary of the provided text using OpenAI.
    ext = os.path.splitext(file_path)[1].lower()

    analyze_text_info = load_subfile_content(file_path, FILE_HEADER, FILE_ANALYZE_HEADER)
    if analyze_text_info:
        analyze_text_info = f"## Characteristics of the Support Materials\n{analyze_text_info}\n"
    else:
        if ext == '.pdf':
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
    return call_openai_api(SUPPORT_LLM_MODEL, contents=[get_system_role(), summarize_text_prompt], function_name="summarize_text")

def extract_and_organize_data(request_text, articles):
    report_contents = []

    for article in articles:
        file_path = article.get('file_path')
        if not file_path:
            continue

        file_content = read_file_content(file_path)

        analyze_text_info = load_subfile_content(file_path, FILE_HEADER, FILE_ANALYZE_HEADER)
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
        file_summary = call_openai_api(SUPPORT_LLM_MODEL, contents=[get_system_role(), extract_matching_content], function_name="extract_and_organize_data")
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

    first_response = call_openai_api(LLM_MODEL, contents=[get_system_role(), base_user_content], function_name="make_report")
    
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
        last_response = call_openai_api(LLM_MODEL, contents=[get_system_role(), base_user_content, first_response, proofread_user_content], function_name="make_report")

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
    return call_openai_api(SUPPORT_LLM_MODEL, contents=[get_system_role(), user_content], function_name="extract_user_intent")

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
    directory = os.path.join(f"uploaded_files_{TARGET_CLASS_NAME}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    create_weaviate_class()
    port = int(os.getenv("COGNIFY_VAULT_PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
