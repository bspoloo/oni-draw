from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionImg2ImgPipeline
from app.config import Config
import torch
from peft import PeftModel

def setup_text2img_with_lora(base_model_id, lora_path):
    print("Cargando modelo text2img con LoRA...")
    """Configurar pipeline text2img con LoRA"""
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe = pipe.to(Config.DEVICE)
    return pipe

def setup_img2img_with_lora(base_model_id, lora_path):
    """Configurar pipeline img2img con LoRA"""
    print("Cargando modelo img2img con LoRA...")
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe = pipe.to(Config.DEVICE)
    return pipe