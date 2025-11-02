from PIL import Image
from torchvision import transforms
import torch
from .generator import Generator

class TextToAnime(Generator):
    def __init__(self, pipe):
        super().__init__(pipe)

    def generate(self, prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5, strength : float = 0.7):        
        print("🎨 Generando imagen de anime desde texto...")
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=512,
                height=512
            ).images[0]
        
        return result