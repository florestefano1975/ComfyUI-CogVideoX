# CogVideoX
# Created by AI Wiz Art (Stefano Flore)
# Version: 1.3
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
from scipy.ndimage import gaussian_filter

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999999999999}),
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
        
class CogVideoXImageToVideoNodeExtended:
    pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "num_frames": ("INT", {"default": 98, "min": 49, "max": 2**31-1, "step": 49}),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 30.0}),
                "use_dynamic_cfg": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999999999999}),
                "interpolation_factor": ("INT", {"default": 1, "min": 1, "max": 7, "step": 2}),
                "flow_precision": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "motion_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smoothness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "flow_method": (["Farneback", "TV-L1", "DIS"],),
                "edge_mode": (["Replicate", "Reflect", "Wrap", "Constant"],),
                "interpolation_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("VIDEO","VIDEO",)
    RETURN_NAMES = ("normal_video","interpolated_video",)
    FUNCTION = "generate_extended_video"
    CATEGORY = "AI WizArt/CogVideoX"

    @classmethod
    def load_model(cls):
        if cls.pipe is None:
            model = "THUDM/CogVideoX-5b-I2V"
            model_dir = cls.download_model_if_needed(model)
            cls.pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
            cls.pipe.scheduler = CogVideoXDPMScheduler.from_config(cls.pipe.scheduler.config, timestep_spacing="trailing")
            cls.pipe.enable_sequential_cpu_offload()
            cls.pipe.vae.enable_slicing()
            cls.pipe.vae.enable_tiling()
        return cls.pipe

    @staticmethod
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

    def generate_extended_video(self, prompt, image, num_frames, num_inference_steps, guidance_scale, use_dynamic_cfg, seed, 
                                interpolation_factor, flow_precision, motion_threshold, smoothness, flow_method, edge_mode, interpolation_strength):
        num_frames = max(49, (num_frames // 49) * 49)
        
        try:
            pipe = self.load_model()
            generator = torch.Generator().manual_seed(seed)

            pil_image = self.preprocess_image(image)
            print(f"Preprocessed image size: {pil_image.size}")

            all_frames = []
            segment_size = 49
            last_frame = pil_image

            with tqdm(total=num_frames, desc="Generating extended video") as progress_bar:
                while len(all_frames) < num_frames:
                    frames_to_generate = min(segment_size, num_frames - len(all_frames))
                    
                    context_images = [last_frame]
                    
                    print(f"Generating segment. Last frame size: {last_frame.size}")

                    output = pipe(
                        prompt=prompt,
                        image=context_images,
                        num_inference_steps=num_inference_steps,
                        num_frames=frames_to_generate,
                        use_dynamic_cfg=use_dynamic_cfg,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        width=720,
                        height=480,
                    )

                    new_frames = self.process_output_frames(output.frames)
                    print(f"Generated {len(new_frames)} new frames")
                    
                    if new_frames:
                        all_frames.extend(new_frames)
                        last_frame = Image.fromarray(new_frames[-1])
                    else:
                        print("Warning: No new frames generated in this iteration")

                    progress_bar.update(len(new_frames))

            all_frames = all_frames[:num_frames]
            print(f"Final video length before interpolation: {len(all_frames)} frames")

            interpolated_frames = self.apply_optical_flow_interpolation(all_frames, interpolation_factor, flow_precision, 
                                                                        motion_threshold, smoothness, flow_method, edge_mode, 
                                                                        interpolation_strength)

            print(f"Final video length after interpolation: {len(interpolated_frames)} frames")

            return (all_frames,interpolated_frames,)
        except Exception as e:
            print(f"Error during extended video generation: {str(e)}")
            raise

    def apply_optical_flow_interpolation(self, frames, factor, precision, threshold, smoothness, method, edge_mode, strength):
        interpolated = []
        flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

        with tqdm(total=len(frames) - 1, desc="Applying optical flow interpolation") as pbar:
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

                if method == "Farneback":
                    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **flow_params)
                elif method == "TV-L1":
                    try:
                        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
                    except AttributeError:
                        try:
                            optical_flow = cv2.createOptFlow_DualTVL1()
                        except AttributeError:
                            print("TV-L1 optical flow not available. Using Farneback method instead.")
                            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **flow_params)
                        else:
                            flow = optical_flow.calc(gray1, gray2, None)
                    else:
                        flow = optical_flow.calc(gray1, gray2, None)
                elif method == "DIS":
                    try:
                        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray1, gray2, None)
                    except AttributeError:
                        print("DIS optical flow not available. Using Farneback method instead.")
                        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **flow_params)

                flow = self.apply_smoothness(flow, smoothness)

                flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow[flow_magnitude < threshold] = 0

                interpolated.append(frame1)
                
                for j in range(1, factor + 1):
                    t = j / (factor + 1)
                    warped = self.warp_flow(frame1, flow * t * strength, edge_mode)
                    blended = cv2.addWeighted(frame1, 1 - t, warped, t, 0)
                    interpolated.append(blended)

                pbar.update(1)

        interpolated.append(frames[-1])
        return interpolated

    def apply_smoothness(self, flow, smoothness):
        if smoothness > 0:
            sigma = smoothness * 5
            flow[:,:,0] = gaussian_filter(flow[:,:,0], sigma=sigma)
            flow[:,:,1] = gaussian_filter(flow[:,:,1], sigma=sigma)
        return flow

    def warp_flow(self, img, flow, edge_mode):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        
        if edge_mode == "Replicate":
            border_mode = cv2.BORDER_REPLICATE
        elif edge_mode == "Reflect":
            border_mode = cv2.BORDER_REFLECT
        elif edge_mode == "Wrap":
            border_mode = cv2.BORDER_WRAP
        elif edge_mode == "Constant":
            border_mode = cv2.BORDER_CONSTANT
        else:
            border_mode = cv2.BORDER_REPLICATE

        return cv2.remap(img, flow, None, cv2.INTER_LINEAR, borderMode=border_mode)

    def process_output_frames(self, frames):
        processed_frames = []
        for frame in frames:
            if isinstance(frame, list):
                for subframe in frame:
                    if isinstance(subframe, np.ndarray):
                        processed_frames.append(subframe)
                    elif isinstance(subframe, Image.Image):
                        processed_frames.append(np.array(subframe))
                    else:
                        print(f"Unexpected subframe type: {type(subframe)}")
            elif isinstance(frame, np.ndarray):
                processed_frames.append(frame)
            elif isinstance(frame, Image.Image):
                processed_frames.append(np.array(frame))
            else:
                print(f"Unexpected frame type: {type(frame)}")
        
        return processed_frames

    def preprocess_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]

        if image.ndim == 3:
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[2] != 3:
                raise ValueError(f"The image must have 3 color channels, found: {image.shape[2]}")

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        
        target_size = (720, 480)
        return resize_and_crop(pil_image, target_size)

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
            if video.shape[1] == 3 or video.shape[1] == 4:
                video = np.transpose(video, (0, 2, 3, 1))
        elif video.ndim == 3:
            video = video[np.newaxis, ...]
        
        print(f"Video shape after preprocessing: {video.shape}")
        
        if video.dtype != np.uint8:
            video = (video * 255).astype(np.uint8)
        
        num_frames, height, width, channels = video.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in video:
            if channels == 4:
                frame = frame[:, :, :3]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        print(f"Video saved to {output_path}")

        return {"ui": {"text": f"Video saved to {output_path}"}}

NODE_CLASS_MAPPINGS = {
    "🤖 CogVideoX ➡️ Image-2-Video": CogVideoXImageToVideoNode,
    "🤖 CogVideoX ➡️ Image-2-Video Extended": CogVideoXImageToVideoNodeExtended,
    "🤖 CogVideoX ➡️ Save Video": SaveVideoNode,
}