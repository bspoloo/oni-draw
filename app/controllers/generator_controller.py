from flask import Blueprint, current_app, jsonify, request, send_from_directory
from app.services.file_service import FileService
from werkzeug.datastructures import FileStorage
from app.config import Config

from app.services.generator_service import GeneratorService

generator_bp = Blueprint('generator', __name__)
file_service = FileService()
generator_service = GeneratorService()

@generator_bp.route('/image-to-image', methods=['POST'])
def generate_image2image():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part'
        }), 400
    
    file : FileStorage = request.files['file']

    # En multipart/form-data, los datos JSON llegan como texto en request.form
    data_str = request.form.get('data')  # "data" es la key que envías en Postman
    data = {}
    if data_str:
        import json
        data = json.loads(data_str)

    filename: str = data.get('filename', file.filename)
    prompt: str = data.get('prompt', "")
    num_inference_steps: int = int(data.get('num_inference_steps', 50))
    strength: float = float(data.get('strength', 0.8))
    guidance_scale: float = float(data.get('guidance_scale', 7.5))

    # Guardar el archivo primero
    saved_file_result = file_service.save_file(file)

    # Procesar con el generator_service
    result = generator_service.image_to_image(
        f"{Config.UPLOAD_FOLDER}/{saved_file_result['filename']}",
        prompt,
        num_inference_steps,
        strength,
        guidance_scale
    )

    if result['status'] == 'error':
        return jsonify(result), 400
    else:
        # return send_from_directory(current_app.config['RESULT_FOLDER'], result['filename'])
        return jsonify(result), 200

@generator_bp.route('/text-to-image', methods=['POST'])
def generate_text2image():
    
    data = request.get_json()
    
    if data is {}:
    
        return jsonify({
            'status': 'error',
            'message': 'No data part'
        }), 400

    prompt: str = data.get('promp') if 'promp' in data else data.get('prompt', "")
    num_inference_steps: int = int(data.get('num_inference_steps', 50))
    strength: float = float(data.get('strength', 0.8))
    guidance_scale: float = float(data.get('guidance_scale', 7.5))
    
    result = generator_service.text_to_image(prompt, num_inference_steps, strength, guidance_scale)
    if result['status'] == 'error':
        return jsonify(result), 400
    else:
        # return send_from_directory(current_app.config['RESULT_FOLDER'], result['filename'])
        return jsonify(result), 200
    
@generator_bp.route('/result/<filename>', methods=['GET'])
def get_result_file(filename: str):
    try:
        return send_from_directory(current_app.config['RESULT_FOLDER'], filename)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al obtener la imagen "{filename}": {str(e)}'
        }), 400