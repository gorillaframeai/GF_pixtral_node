# -*- coding: utf-8 -*-
"""
ComfyUI node for Mistral + Pixtral with batch support
Compatible with existing workflows
"""
import json
import base64
import os
import requests
from io import BytesIO
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image

class GF_MistralPixtralNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True,
                                           "default": "Describe this image in detail"}),
                "system_prompt": ("STRING", {"multiline": True,
                                             "default": "You are a helpful AI assistant."}),
                "model": (["pixtral-12b-2409",
                           "mistral-large-latest",
                           "mistral-small-latest"],
                          {"default": "pixtral-12b-2409"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "batch_mode": (["concatenate", "individual"], {"default": "concatenate"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("response", "response_list")
    FUNCTION = "process"
    CATEGORY = "GF/AI"

    # ---------- helpers ----------
    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
        """Convert ComfyUI batch tensor [B,H,W,C] -> list of PIL images."""
        tensor = tensor.clone()
        if tensor.dim() == 3:          # single image
            tensor = tensor.unsqueeze(0)
        # tensor is now [B,H,W,C], float 0-1
        tensor = (tensor * 255).clamp(0, 255).byte()
        tensor = tensor.cpu().numpy()
        return [Image.fromarray(img) for img in tensor]

    @staticmethod
    def _pil_to_base64(pil_img: Image.Image) -> str:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _load_api_key() -> str | None:
        try:
            key_file = os.path.join(os.path.dirname(__file__), "apikey.txt")
            if not os.path.exists(key_file):
                return None
            api_key = open(key_file, encoding="utf-8").read().strip()
            if len(api_key) < 10 or any(x in api_key.lower()
                                        for x in ["вставить", "ключ", "replace", "your"]):
                return None
            return api_key
        except Exception:
            return None

    # ---------- main ----------
    def process(self,
                user_prompt,
                system_prompt,
                model,
                max_tokens,
                temperature,
                batch_mode,
                image=None):

        api_key = self._load_api_key()
        if not api_key:
            err = ("Error: API key not found/invalid. "
                   "Place your Mistral key in apikey.txt next to this node.")
            return (err, [err])

        api_url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}

        images = []
        if image is not None:
            images = self._tensor_to_pil(image)  # List[PIL.Image]

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if images:
            selected_model = "pixtral-12b-2409"
            if batch_mode == "concatenate":
                # One user message with all images
                content = [{"type": "text", "text": user_prompt}]
                for pil in images:
                    b64 = self._pil_to_base64(pil)
                    content.append({"type": "image_url",
                                    "image_url": f"data:image/png;base64,{b64}"})
                messages.append({"role": "user", "content": content})
            else:  # individual
                # One user message per image
                for pil in images:
                    b64 = self._pil_to_base64(pil)
                    messages.append({"role": "user",
                                     "content": [{"type": "text", "text": user_prompt},
                                                 {"type": "image_url",
                                                  "image_url": f"data:image/png;base64,{b64}"}]})
        else:
            selected_model = model
            messages.append({"role": "user", "content": user_prompt})

        payload = {"model": selected_model,
                   "messages": messages,
                   "max_tokens": max_tokens,
                   "temperature": temperature}

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            if batch_mode == "concatenate" or not images:
                # One answer
                answer = data["choices"][0]["message"]["content"]
                answers = [answer]
            else:
                # Multiple choices (one per image)
                answers = [ch["message"]["content"] for ch in data["choices"]]

            # ComfyUI likes strings – join for compatibility
            full_response = "\n\n---\n\n".join(answers)
            return (full_response, answers)

        except Exception as e:
            err = f"Mistral API error: {e}"
            return (err, [err])

# Node registration
NODE_CLASS_MAPPINGS = {"GF_MistralPixtralNode": GF_MistralPixtralNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GF_MistralPixtralNode": "GF Mistral & Pixtral (batch max 10)"}