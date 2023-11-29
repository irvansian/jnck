from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
import os
import subprocess

app = Flask(__name__)

# Ensure the 'uploads' and 'edited' directories exist
if not os.path.exists('data/uploads'):
    os.makedirs('data/uploads')

if not os.path.exists('data/edited'):
    os.makedirs('data/edited')

# Set the upload folder inside the 'data' directory
app.config['UPLOAD_FOLDER'] = 'data/uploads'
# Limit upload size to 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Define allowed extensions for video files
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/edit_video', methods=['POST'])
def edit_video():
    # Check if the POST request has the file part
    if 'video' not in request.files or 'prompt' not in request.form:
        return jsonify({'error': 'Missing video file or prompt'}), 400

    video_file = request.files['video']
    prompt = request.form['prompt']

    # Check if the file is allowed
    if video_file and allowed_file(video_file.filename):
        # Save the uploaded video inside the 'data' directory
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        # Call the function to edit the video based on the prompt
        edited_video_path = edit_video_function(video_path, prompt)

        # Check if editing was successful
        if edited_video_path is not None:
            # Provide the edited video for download or streaming
            return jsonify({'edited_video_path': edited_video_path})
        else:
            # Return an error message if editing failed
            return jsonify({'error': 'Video editing failed'}), 500

    else:
        # Return an error message for invalid file type
        return jsonify({'error': 'Invalid file type'}), 400

def edit_video_function(video_path, prompt):
    # Build the command to call preprocess.py
    preprocess_command = [
        'python', 'preprocess.py',
        '--data_path', video_path,
        '--inversion_prompt', prompt
    ]

    edit_video_command = [
        'python', 'run_tokenflow_pnp.py'
    ]

    try:
        # Execute the command
        subprocess.run(preprocess_command, check=True)
        subprocess.run(edit_video_command, check=True)
        # Return the path of the edited video (adjust this based on preprocess.py behavior)
        edited_video_path = video_path.replace('uploads', 'edited')
        return edited_video_path
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Error calling preprocess.py or run_tokenflow_pnp.py: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0')
