# -*- coding: utf-8 -*-
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
    ComfyUI node for Mistral and Pixtral API
    - If image provided - uses Pixtral for image description
    - If no image - uses Mistral as chat
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant. Answer clearly and informatively."}),
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
    CATEGORY = "GF/AI"
    
    def load_api_key(self):
        """Load API key from apikey.txt file"""
        try:
            # Look for file in the same folder as the node
            current_dir = os.path.dirname(os.path.abspath(__file__))
            key_file = os.path.join(current_dir, "apikey.txt")
            
            if os.path.exists(key_file):
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                    
                    # Check if API key contains only ASCII characters (valid API key format)
                    if not api_key.isascii():
                        return None  # Invalid API key format
                    
                    # Check if it's not the placeholder text
                    placeholder_indicators = ['вставить', 'ключ', 'replace', 'your', 'api', 'key', 'here', 'this', 'text']
                    if len(api_key) < 10 or any(indicator in api_key.lower() for indicator in placeholder_indicators):
                        return None  # Placeholder text detected
                    
                    return api_key
            else:
                return None
        except Exception as e:
            print(f"Error reading API key: {e}")
            return None
    
    def process(self, user_prompt, system_prompt, model, max_tokens, temperature, image=None):
        try:
            # Load API key from file
            api_key = self.load_api_key()
            if not api_key:
                return ("Error: API key not found or invalid. Please:\n1. Open apikey.txt file in the node folder\n2. Replace the placeholder text with your actual Mistral API key\n3. Save the file\n\nGet your API key from: https://console.mistral.ai/",)
            
            # Base URL for Mistral API
            api_url = "https://api.mistral.ai/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # If image provided - use Pixtral
            if image is not None:
                # Convert tensor to PIL Image
                if isinstance(image, torch.Tensor):
                    # ComfyUI usually passes images as tensor [batch, height, width, channels]
                    if image.dim() == 4:
                        image = image[0]  # Take first image from batch
                    
                    # Convert from [0,1] to [0,255] if needed
                    if image.max() <= 1.0:
                        image = (image * 255).clamp(0, 255).byte()
                    
                    # Convert tensor to numpy array
                    image_np = image.cpu().numpy().astype(np.uint8)
                    pil_image = Image.fromarray(image_np)
                else:
                    pil_image = image
                
                # Convert image to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Form message with image for Pixtral
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"}
                    ]
                }
                
                # Force use Pixtral for images
                selected_model = "pixtral-12b-2409"
                
            else:
                # If no image - regular text chat
                user_message = {
                    "role": "user", 
                    "content": user_prompt
                }
                selected_model = model
            
            messages.append(user_message)
            
            # Prepare request data
            data = {
                "model": selected_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Send request
            response = requests.post(api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return (result["choices"][0]["message"]["content"],)
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                print(error_msg)
                return (error_msg,)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return (error_msg,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "GF_MistralPixtralNode": GF_MistralPixtralNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GF_MistralPixtralNode": "GF Mistral & Pixtral"
}