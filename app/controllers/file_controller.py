

from flask import Blueprint, current_app
from app.services.file_service import FileService
from flask import Blueprint, jsonify, request
from werkzeug.datastructures import FileStorage
from app.services.file_service import FileService


file_bp = Blueprint('file', __name__)
file_service = FileService()

@file_bp.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        
        return jsonify({
            'status': 'error',
            'message': 'No file part'
        }), 400
    
    file : FileStorage = request.files['file']
    
    result = file_service.save_file(file)
    
    if result['status'] == 'error':
        return jsonify(result), 400
    else:
        return jsonify(result), 200

@file_bp.route('/<filename>', methods=['GET'])
def get_file(filename: str):
    try:
        return file_service.get_file(filename)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al obtener la imagen "{filename}": {str(e)}'
        }), 400