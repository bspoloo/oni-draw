from datetime import timedelta
import os

import torch
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel

class Config:
    # SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:root@localhost/draw_db"
    # SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'dsu22uhe3iuhfiuh2iwqiudQwoejfWWE##12'
    
    # UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    # MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # LOCATION_PROJECT = r"D:\University\6.- sexto semestre\Sistemas en tiempo real\Proyecto final\confort remote\backend"
    
    # JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ccc387ba193a315cbcd1ad7d8d007e6124763894554418e7c90b7dbcd7edca23")
    # JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
    # JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)
    
    # Hyperparemeters for drawing model
    
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    BATCH_SIZE = 2
    IMAGE_SIZE = 512
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-4

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    VAE = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    UNET = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    TEXT_ENCODER = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    SCHEDULER = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ANIME_DIR = r"D:\Ciencias\Drawnime\data\train\faces"
    SKETCH_DIR = r"D:\Ciencias\Drawnime\data\train\sketches"

    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    RESULT_FOLDER = os.path.join(os.getcwd(), 'results')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}