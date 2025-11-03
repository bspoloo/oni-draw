from flask import Flask
from .config import Config
from app.controllers.file_controller import file_bp
from app.controllers.generator_controller import generator_bp
import os

def create_app():
    # Configurar las rutas de templates y static correctamente
    # template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    # static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
    # app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    
    app = Flask(__name__)
    app.config.from_object(Config)
    app.register_blueprint(file_bp, url_prefix="/api/upload")
    app.register_blueprint(generator_bp, url_prefix="/api/generator")
    
    return app