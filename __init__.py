# CogVideoX
# Created by AI Wiz Art (Stefano Flore)
# Version: 1.0
# https://stefanoflore.it
# https://ai-wiz.art

import os
import torch
import numpy as np
import cv2
from datetime import datetime
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

def download_model_if_needed(model_name, local_dir="models/CogVideoX"):
    model_dir = os.path.join(local_dir, model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        print(f"Model {model_name} not found locally. Downloading...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Model {model_name} downloaded successfully.")
    return model_dir

class CogVideoXImageToVideoNode:
    pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 30.0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "frames")
    FUNCTION = "generate_video"
    CATEGORY = "AI WizArt/CogVideoX"

    @classmethod
    def load_model(cls):
        if cls.pipe is None:
            model = "THUDM/CogVideoX-5b-I2V"
            model_dir = download_model_if_needed(model)
            cls.pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
            cls.pipe.scheduler = CogVideoXDPMScheduler.from_config(cls.pipe.scheduler.config, timestep_spacing="trailing")
            cls.pipe.enable_sequential_cpu_offload()
            cls.pipe.vae.enable_slicing()
            cls.pipe.vae.enable_tiling()
        return cls.pipe

    def generate_video(self, prompt, image, num_inference_steps, guidance_scale, seed):
        try:
            pipe = self.load_model()
            generator = torch.Generator().manual_seed(seed)

            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()

            print(f"Input image shape: {image.shape}")

            if image.ndim == 4:
                if image.shape[0] == 1:
                    image = image[0]
                else:
                    raise ValueError(f"Unsupported image format: {image.shape}")

            if image.ndim == 3:
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                elif image.shape[2] != 3:
                    raise ValueError(f"The image must have 3 color channels, found: {image.shape[2]}")

            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)
            print(f"PIL Image dimensions: {pil_image.size}")

            target_size = (720, 480)
            pil_image = resize_and_crop(pil_image, target_size)
            print(f"Image dimensions after resizing: {pil_image.size}")

            with tqdm(total=num_inference_steps, desc="Generating video") as progress_bar:
                def update_progress(step, timestep, latents):
                    progress_bar.update(1)
                
                original_step = pipe.scheduler.step
                def step_with_progress(*args, **kwargs):
                    result = original_step(*args, **kwargs)
                    update_progress(None, None, None)
                    return result
                pipe.scheduler.step = step_with_progress

                output = pipe(
                    prompt=prompt,
                    image=pil_image,
                    num_inference_steps=num_inference_steps,
                    num_frames=49,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

                pipe.scheduler.step = original_step

            if isinstance(output.frames, list):
                if all(isinstance(frame, list) for frame in output.frames):
                    video_frames = [np.array(frame) for sublist in output.frames for frame in sublist]
                elif all(isinstance(frame, np.ndarray) for frame in output.frames):
                    video_frames = output.frames
                elif all(isinstance(frame, Image.Image) for frame in output.frames):
                    video_frames = [np.array(frame) for frame in output.frames]
                else:
                    raise ValueError(f"Unexpected frame type in output list: {type(output.frames[0])}")
            elif isinstance(output.frames, np.ndarray):
                video_frames = output.frames
            else:
                raise ValueError(f"Unexpected output type: {type(output.frames)}")

            # Ensure all frames are in the correct format (H, W, C)
            video_frames = [frame if frame.ndim == 3 else frame[0] for frame in video_frames]
            
            # Convert to the format expected by ComfyUI
            comfy_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in video_frames]

            return (video_frames, comfy_frames)
        except Exception as e:
            print(f"Error during video generation: {str(e)}")
            raise

def resize_and_crop(image, target_size):
    width, height = image.size
    target_width, target_height = target_size
    
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))

class SaveVideoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "cogvideox"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "AI WizArt/CogVideoX"

    def save_video(self, video, filename_prefix, fps):
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.mp4"
        output_path = os.path.join(output_dir, filename)
        
        print(f"Received video type: {type(video)}")
        
        if isinstance(video, list):
            print(f"Video list length: {len(video)}")
            if len(video) > 0:
                print(f"First element type: {type(video[0])}")
            video = np.array(video)
        
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        
        if video.ndim == 4:
            if video.shape[1] == 3 or video.shape[1] == 4:  # (T, C, H, W)
                video = np.transpose(video, (0, 2, 3, 1))
        elif video.ndim == 3:  # Single image (H, W, C)
            video = video[np.newaxis, ...]  # Add time dimension
        
        print(f"Video shape after preprocessing: {video.shape}")
        
        if video.dtype != np.uint8:
            video = (video * 255).astype(np.uint8)
        
        num_frames, height, width, channels = video.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in video:
            if channels == 4:  # If the image has an alpha channel, remove it
                frame = frame[:, :, :3]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        print(f"Video saved to {output_path}")

        return {"ui": {"text": f"Video saved to {output_path}"}}

# Register nodes in ComfyUI
NODE_CLASS_MAPPINGS = {
    "🤖 CogVideoX ➡️ Image-2-Video": CogVideoXImageToVideoNode,
    "🤖 CogVideoX ➡️ Save Video": SaveVideoNode,
}