from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import os
import subprocess
import yaml
import random

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
        extracted_filename = filename[5:13]
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], extracted_filename + '.mp4')
        video_file.save(video_path)

        # Call the function to edit the video based on the prompt
        edited_video_path = edit_video_function(video_path, prompt, extracted_filename)

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

def edit_video_function(video_path, prompt, filename):
    # Build the command to call preprocess.py
    preprocess_command = [
        'python', 'preprocess.py',
        '--data_path', video_path,
        '--inversion_prompt', prompt
    ]

    config_path = 'configs/config_' + filename + '.yaml'
    create_config_file(filename, prompt, config_path)

    edit_video_command = [
        'python', 'run_tokenflow_pnp.py',
        '--config_path', config_path
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

def create_config_file(filename, prompt, config_path):
    seed = random.randint(0, 2**32 - 1)
    config_data = {
        'seed': seed,
        'device': 'cuda',
        'output_path': 'tokenflow-results',
        'data_path': 'data/' + filename,
        'latents_path': 'latents',  # should be the same as 'save_dir' arg used in preprocess
        'n_inversion_steps': 500,  # for retrieving the latents of the inversion
        'n_frames': 40,
        'sd_version': '2.1',
        'guidance_scale': 7.5,
        'n_timesteps': 50,
        'prompt': prompt,
        'negative_prompt': "ugly, blurry, low res, unrealistic, unaesthetic",
        'batch_size': 8,
        'pnp_attn_t': 0.5,
        'pnp_f_t': 0.8
    }

    # Specify the path for your new config file
    config_path = config_path

    # Write the configuration to a YAML file
    with open(config_path, 'w') as file:
        yaml.dump(config_data, file)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

