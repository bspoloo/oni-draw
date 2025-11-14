import os
import gc
import sys
from tkinter import Image
import psutil
import torch
from classes.sketch_2_anime import SketchToAnime
from classes.text_2_anime import TextToAnime
from app.config import Config
from flask import current_app
from datetime import datetime

class GeneratorService:
    _instance = None
    _text_pipe = None
    _image_pipe = None
    _current_mode = None

    def __init__(self):
        self._load_image_model()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeneratorService, cls).__new__(cls)
        return cls._instance

    def _get_memory_info(self):
        """Obtiene información de memoria RAM y swap"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'ram_used_gb': memory.used / (1024**3),
            'ram_available_gb': memory.available / (1024**3),
            'ram_total_gb': memory.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3)
        }

    def _check_memory_sufficient(self, required_gb=2):
        """Verifica si hay suficiente memoria RAM disponible"""
        memory_info = self._get_memory_info()
        available_gb = memory_info['ram_available_gb']
        
        print(f"Memoria disponible: {available_gb:.1f}GB / Requerida: ~{required_gb}GB")
        
        if available_gb < required_gb:
            print("Memoria RAM insuficiente, liberando...")
            self._aggressive_memory_cleanup()
            
            # Verificar nuevamente
            memory_info = self._get_memory_info()
            available_gb = memory_info['ram_available_gb']
            
            if available_gb < required_gb:
                raise MemoryError(f"Memoria RAM insuficiente. Disponible: {available_gb:.1f}GB, Requerido: ~{required_gb}GB")
        
        return True

    def _aggressive_memory_cleanup(self):
        """Limpieza agresiva de memoria RAM y GPU"""
        print("Realizando limpieza agresiva de memoria...")
        
        # Liberar modelos
        if self._text_pipe is not None:
            del self._text_pipe
            self._text_pipe = None
            
        # if self._image_pipe is not None:
        #     del self._image_pipe
        #     self._image_pipe = None
        
        self._current_mode = None
        
        # Limpiar GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Forzar garbage collection
        gc.collect()
        
        # Limpiar imports y módulos temporales
        if 'functions.load_lora_model' in sys.modules:
            del sys.modules['functions.load_lora_model']
        
        print("Limpieza de memoria completada")

    def _load_text_model(self):
        """Carga el modelo de texto a imagen con verificación de memoria"""
        self._check_memory_sufficient(required_gb=2)
        
        if self._text_pipe is None or self._current_mode != 'text':
            # Liberar modelo anterior
            if self._image_pipe is not None:
                del self._image_pipe
                self._image_pipe = None
            
            self._aggressive_memory_cleanup()

            print("Cargando modelo text2img con LoRA...")
            from functions.load_lora_model import setup_text2img_with_lora
            
            # Configurar PyTorch para usar menos RAM
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            self._text_pipe = setup_text2img_with_lora(
                Config.MODEL_ID, 
                r"D:\Ciencias\Drawnime\ai_models\sketch_to_anime_lora_final4"
            )
            self._current_mode = 'text'
            
            memory_info = self._get_memory_info()
            print(f"Modelo text2img cargado. RAM usada: {memory_info['ram_used_gb']:.1f}GB")

    def _load_image_model(self):
        """Carga el modelo de imagen a imagen con verificación de memoria"""
        self._check_memory_sufficient(required_gb=3)  # 3GB estimado para el modelo
        
        if self._image_pipe is None or self._current_mode != 'image':
            # Liberar modelo anterior
            if self._text_pipe is not None:
                del self._text_pipe
                self._text_pipe = None
            
            self._aggressive_memory_cleanup()
            
            print("Cargando modelo img2img con LoRA...")
            from functions.load_lora_model import setup_img2img_with_lora
            
            # Configurar PyTorch para usar menos RAM
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            self._image_pipe = setup_img2img_with_lora(
                Config.MODEL_ID, 
                r"D:\Ciencias\Drawnime\ai_models\sketch_to_anime_lora_final4"
            )
            self._current_mode = 'image'
            
            memory_info = self._get_memory_info()
            print(f"Modelo img2img cargado. RAM usada: {memory_info['ram_used_gb']:.1f}GB")

    def text_to_image(self, prompt, num_inference_steps=30, strength=0.9, guidance_scale=7.5, number_per_prompt=1):
        """Versión optimizada con menos pasos de inferencia"""
        try:
            # Verificar memoria antes de empezar
            self._check_memory_sufficient(required_gb=1)
            
            self._load_text_model()
            
            prompt = prompt if prompt else "anime style, high quality, detailed, hair with vibrant colors, masterpiece"
            text_to_anime = TextToAnime(self._text_pipe)
            print("Generando imagen de anime desde texto...")
            # Reducir pasos de inferencia para ahorrar memoria
            results = text_to_anime.generate(
                prompt=prompt, 
                num_inference_steps=num_inference_steps,
                strength=strength, 
                guidance_scale=guidance_scale,
                number_per_prompt=number_per_prompt
            )
            filenames = self.save_images(results, current_app.config['RESULT_FOLDER'])
            del text_to_anime
            self._aggressive_memory_cleanup()

            return {
                "status": "success",
                "message": "imagen generada correctamente",
                "filenames": filenames
            }
            
        except MemoryError as e:
            print(f"Error de memoria: {e}")
            self._aggressive_memory_cleanup()
            return {
                "status": "error",
                "message": "Memoria insuficiente. Intenta nuevamente o reduce la resolución."
            }
        except Exception as e:
            print(f"Error en text_to_image: {e}")
            self._aggressive_memory_cleanup()
            return {
                "status": "error",
                "message": f"Error generando imagen: {str(e)}"
            }

    def image_to_image(self, input_image, prompt, num_inference_steps=30, strength=0.7, guidance_scale=7.5, number_per_prompt=1):
        """Versión optimizada para imagen a imagen"""
        try:
            # Verificar memoria antes de empezar
            self._check_memory_sufficient(required_gb=1)
            
            # self._load_image_model()

            prompt = prompt if prompt else "anime style, high quality, detailed, hair with vibrant colors, masterpiece"

            sketch_to_anime = SketchToAnime(self._image_pipe)
            # Reducir parámetros para ahorrar memoria
            results = sketch_to_anime.generate(
                input_image, 
                prompt=prompt, 
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                number_per_prompt=number_per_prompt
            )
            filenames = self.save_images(results, current_app.config['RESULT_FOLDER'])
            # Limpiar inmediatamente
            del sketch_to_anime
            self._aggressive_memory_cleanup()

            return {
                "status": "success",
                "message": "imagen generada correctamente",
                "filenames": filenames
            }
            
        except MemoryError as e:
            print(f"Error de memoria: {e}")
            self._aggressive_memory_cleanup()
            return {
                "status": "error",
                "message": "Memoria insuficiente. Intenta nuevamente o reduce el tamaño de la imagen."
            }
        except Exception as e:
            print(f"Error en image_to_image: {e}")
            self._aggressive_memory_cleanup()
            return {
                "status": "error",
                "message": f"Error generando imagen: {str(e)}"
            }

    def save_images(self, images: list[Image], folder: str) -> list[str]:
        """Guarda una lista de imágenes en la carpeta especificada y retorna sus nombres de archivo"""
        filenames = []
        for idx, img in enumerate(images):
            random_name = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
            url_image = os.path.join(folder, random_name)
            img.save(url_image)
            filenames.append(random_name)
        return filenames

    def unload_models(self):
        """Libera todos los modelos de memoria"""
        self._aggressive_memory_cleanup()
        print("🧹 Todos los modelos descargados de la memoria")