
from classes.sketch_2_anime import SketchToAnime
from classes.text_2_anime import TextToAnime
from app.config import Config
from flask import current_app
from datetime import datetime

class GeneratorService:
    def __init__(self):
        pass
    
    def text_to_image(self, prompt, num_inference_steps=50, strength=0.9, guidance_scale=9.5):
        
        random_name = "output_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        
        from functions.load_lora_model import setup_text2img_with_lora
        pipe = setup_text2img_with_lora(Config.MODEL_ID, r"D:\Ciencias\Drawnime\ai_models\sketch_to_anime_lora_final3")
        text_to_anime = TextToAnime(pipe)
        prompt = prompt if prompt else "anime style, high quality, detailed, hair with vibrant colors, masterpiece"
        
        result = text_to_anime.generate(prompt=prompt, num_inference_steps=50, strength=0.9, guidance_scale=9.5)
        url_image = f"{current_app.config['RESULT_FOLDER']}/{random_name}"
        result.save(url_image)
        
        return {
            "status" : "success",
            "message" : "imagen generada correctamente",
            "filename": random_name
        }
    
    def image_to_image(self,input_image, prompt, num_inference_steps=50, strength=0.9, guidance_scale=9.5):
        random_name = "output_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        
        from functions.load_lora_model import setup_img2img_with_lora
        pipe = setup_img2img_with_lora(Config.MODEL_ID, r"D:\Ciencias\Drawnime\ai_models\sketch_to_anime_lora_final3")
        sketch_to_anime = SketchToAnime(pipe)

        prompt = prompt if prompt else "anime style, high quality, detailed, hair with vibrant colors, masterpiece"
        
        result = sketch_to_anime.generate(input_image, prompt=prompt, strength=0.75, guidance_scale=9, num_inference_steps=50)
        
        url_image = f"{current_app.config['RESULT_FOLDER']}/{random_name}"
        result.save(url_image)
        
        return {
            "status" : "success",
            "message" : "imagen generada correctamente",
            "filename": random_name
        }