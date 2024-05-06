import cv2
import torch
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import subprocess
import yaml
import random
import uuid
import threading

app = Flask(__name__)

if not os.path.exists('data/uploads'):
    os.makedirs('data/uploads')
if not os.path.exists('data/edited'):
    os.makedirs('data/edited')

app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv'}

job_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(job_id, video_path, prompt, inversion_prompt, extracted_filename, height, width, frame_count):
    try:
        edited_video_path, _ = edit_video_function(video_path, prompt, inversion_prompt, extracted_filename, height, width, frame_count)
        job_status[job_id] = 'completed' if edited_video_path else 'failed'
    except Exception as e:
        app.logger.error(f"Error processing video: {e}")
        job_status[job_id] = 'failed'

@app.route('/api/edit_video', methods=['POST'])
def edit_video():
    if 'video' not in request.files or 'prompt' not in request.form:
        return jsonify({'error': 'Missing video file or prompt'}), 400

    video_file = request.files['video']
    prompt = request.form['prompt']
    inversion_prompt = request.form['inversion_prompt']

    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        extracted_filename = filename[5:13]
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], extracted_filename + '.mp4')
        video_file.save(video_path)

        video = cv2.VideoCapture(video_path)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        torch.cuda.empty_cache()

        job_id = extracted_filename
        job_status[job_id] = 'processing'
        threading.Thread(target=process_video, args=(job_id, video_path, prompt, inversion_prompt, extracted_filename, height, width, frame_count)).start()

        return jsonify({'message': 'Video is being processed', 'job_id': job_id})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/status/<job_id>', methods=['GET'])
def check_status(job_id):
    status = job_status.get(job_id, 'unknown')
    return jsonify({'job_id': job_id, 'status': status})

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(os.path.join('data/edited', filename), 'tokenflow_PnP_fps_30.mp4')

def edit_video_function(video_path, prompt, inversion_prompt, filename, height, width, frame_count):
    job_id = str(uuid.uuid4())
    job_status[job_id] = 'processing'
    extracted_filename = video_path[5:13]
    latents_path = os.path.join('latents')
    save_dir_path = os.path.join('latents', extracted_filename)
    preprocess_command = [
        'python', 'preprocess.py',
        '--data_path', video_path,
        '--inversion_prompt', inversion_prompt,
        '--save_dir', latents_path,
        '--H', str(int(height)),
        '--W', str(int(width)),
        '--n_frames', str(frame_count)
    ]
    config_path = 'configs/config_' + filename + '.yaml'
    create_config_file(filename, prompt, config_path, frame_count)

    edit_video_command = [
        'python', 'run_tokenflow_pnp.py',
        '--config_path', config_path
    ]


    try:
        subprocess.run(preprocess_command, check=True)

        subprocess.run(edit_video_command, check=True)
        edited_video_path = video_path.replace('uploads', 'edited')
        job_status[job_id] = 'completed'
        return edited_video_path, job_id
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Error calling preprocess.py or run_tokenflow_pnp.py: {e}")
        job_status[job_id] = 'failed'
        return None, job_id

def create_config_file(filename, prompt, config_path, frame_number):
    seed = random.randint(0, 2 ** 32 - 1)
    config_data = {
        'seed': seed,
        'device': 'cuda',
        'output_path': 'data/edited/' + filename,
        'data_path': 'data/' + filename,
        'latents_path': 'latents',
        'n_inversion_steps': 500,
        'n_frames': frame_number,
        'sd_version': '2.1',
        'guidance_scale': 7.5,
        'n_timesteps': 50,
        'prompt': prompt,
        'negative_prompt': "ugly, blurry, low res, unrealistic, unaesthetic",
        'batch_size': 8,
        'pnp_attn_t': 0.5,
        'pnp_f_t': 0.8
    }
    with open(config_path, 'w') as file:
        yaml.dump(config_data, file)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
