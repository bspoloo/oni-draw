from asyncio.windows_utils import pipe
from PIL import Image
from torchvision import transforms
import torch
from .generator import Generator
from diffusers import StableDiffusionImg2ImgPipeline

class SketchToAnime:
    def __init__(self, pipe : StableDiffusionImg2ImgPipeline):
        # super().__init__(pipe)
        self.pipe = pipe

    def generate(self, sketch_path: str, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5, strength : float = 0.7):
        """Generar usando img2img - el sketch como base"""
        print("🎨 Generando imagen de anime desde boceto...")
        # Cargar sketch
        init_image = Image.open(sketch_path).convert("RGB")
        init_image = init_image.resize((512, 512))
        
        # Generar
        result = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        return result