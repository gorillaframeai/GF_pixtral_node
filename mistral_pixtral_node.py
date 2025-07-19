import requests
import json
import base64
import os
from io import BytesIO
from PIL import Image
import torch
import numpy as np

class GF_MistralPixtralNode:
    """
    ComfyUI нода для работы с Mistral и Pixtral API
    - Если есть изображение - использует Pixtral для описания
    - Если нет изображения - использует Mistral как чат
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "Опиши эту картинку подробно"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "Ты полезный AI ассистент. Отвечай на русском языке четко и информативно."}),
                "model": (["pixtral-12b-2409", "mistral-large-latest", "mistral-small-latest"], {"default": "pixtral-12b-2409"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process"
    CATEGORY = "AI/Mistral"
    
    def load_api_key(self):
        """Загружает API ключ из файла apikey.txt"""
        try:
            # Ищем файл в той же папке, где находится нода
            current_dir = os.path.dirname(os.path.abspath(__file__))
            key_file = os.path.join(current_dir, "apikey.txt")
            
            if os.path.exists(key_file):
                with open(key_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                return None
        except Exception as e:
            print(f"Ошибка при чтении API ключа: {e}")
            return None
    
    def process(self, user_prompt, system_prompt, model, max_tokens, temperature, image=None):
        try:
            # Загружаем API ключ из файла
            api_key = self.load_api_key()
            if not api_key:
                return ("Ошибка: не найден файл apikey.txt или он пустой. Создайте файл apikey.txt в папке с нодой и вставьте туда ваш Mistral API ключ.",)
            
            # Базовый URL для Mistral API
            api_url = "https://api.mistral.ai/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Подготовка сообщений
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Если есть изображение - используем Pixtral
            if image is not None:
                # Конвертируем tensor в PIL Image
                if isinstance(image, torch.Tensor):
                    # ComfyUI обычно передает изображения как tensor [batch, height, width, channels]
                    if image.dim() == 4:
                        image = image[0]  # Берем первое изображение из батча
                    
                    # Конвертируем из [0,1] в [0,255] если нужно
                    if image.max() <= 1.0:
                        image = (image * 255).clamp(0, 255).byte()
                    
                    # Конвертируем tensor в numpy array
                    image_np = image.cpu().numpy().astype(np.uint8)
                    pil_image = Image.fromarray(image_np)
                else:
                    pil_image = image
                
                # Конвертируем изображение в base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Формируем сообщение с изображением для Pixtral
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"}
                    ]
                }
                
                # Принудительно используем Pixtral для изображений
                selected_model = "pixtral-12b-2409"
                
            else:
                # Если нет изображения - обычный текстовый чат
                user_message = {
                    "role": "user", 
                    "content": user_prompt
                }
                selected_model = model
            
            messages.append(user_message)
            
            # Подготовка данных для запроса
            data = {
                "model": selected_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Отправка запроса
            response = requests.post(api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return (result["choices"][0]["message"]["content"],)
            else:
                error_msg = f"Ошибка API: {response.status_code} - {response.text}"
                print(error_msg)
                return (error_msg,)
                
        except Exception as e:
            error_msg = f"Ошибка: {str(e)}"
            print(error_msg)
            return (error_msg,)

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "GF_MistralPixtralNode": GF_MistralPixtralNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GF_MistralPixtralNode": "GF Mistral & Pixtral"
}