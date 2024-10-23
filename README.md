# ComfyUI-CogVideoX
Experience the CogVideoX model on ComfyUI

Original project: https://github.com/THUDM/CogVideo

**This custom node is in experimental version**

![Overview](/assets/screenshot_2.png)

## Parameters

### prompt (STRING)

- Description: The text that guides the video generation.
- Type: Multiline string.
- Impact: Directly influences the content and style of the generated video.

### image (IMAGE)

- Description: The input image from which to start the video generation.
- Type: Image
- Impact: Serves as the starting point for the video, strongly influencing the first frames.

### num_frames (INT)

- Description: The total number of frames to generate.
- Default: 49
- Range: 49 - 2^31-1 (practically unlimited)
- Impact: Determines the length of the final video.

### num_inference_steps (INT)

- Description: The number of inference steps for each frame.
- Default: 10
- Range: 1 - 1000
- Impact: Influences the quality and detail of each generated frame. More steps generally mean higher quality but longer processing times.

### guidance_scale (FLOAT)

- Description: Controls how closely the model follows the prompt.
- Default: 6.0
- Range: 0.1 - 30.0
- Impact: Higher values produce results more faithful to the prompt but can lead to artifacts.

### use_dynamic_cfg (BOOLEAN)

- Description: Enables or disables Dynamic Guided Configuration.
- Default: True
- Impact: When enabled, it can improve video consistency and quality.

### seed (INT)

- Description: The seed for random generation.
- Default: 0
- Range: 0 - 99999999999999
- Impact: Controls result reproducibility. The same seed will always produce the same output with the same parameters.
- Optical Flow Interpolation Parameters

### interpolation_factor (INT)

- Description: Determines how many intermediate frames to create between each pair of original frames.
- Default: 1
- Range: 1 - 7 (steps of 2)
- Impact: Increases video smoothness by adding interpolated frames. A value of 1 doubles the number of frames, 3 quadruples it, etc.

### flow_precision (FLOAT)

- Description: Controls the level of detail in optical flow calculation.
- Default: 0.5
- Range: 0.1 - 1.0
- Impact: Higher values produce more precise optical flow but require more computation time.

### motion_threshold (FLOAT)

- Description: Determines the minimum amount of movement required to apply interpolation.
- Default: 0.1
- Range: 0.0 - 1.0
- Impact: Lower values interpolate even small movements, higher values only significant movements.

### smoothness (FLOAT)

- Description: Controls how "smooth" the interpolated movement should be.
- Default: 0.5
- Range: 0.0 - 1.0
- Impact: Higher values produce smoother transitions but may reduce movement details.

### flow_method (COMBO)

- Description: The method used to calculate optical flow.
- Options: ["Farneback", "TV-L1", "DIS"]
- Impact: Each method has different characteristics in terms of accuracy and speed.

### edge_mode (COMBO)

- Description: How to handle image edges during interpolation.
- Options: ["Replicate", "Reflect", "Wrap", "Constant"]
- Impact: Influences the appearance of edges in interpolated frames.

### interpolation_strength (FLOAT)

- Description: How heavily to apply the calculated interpolation.
- Default: 1.0
- Range: 0.0 - 1.0
- Impact: Lower values produce subtler interpolation, higher values more pronounced interpolation.

### upscale_factor (FLOAT)

- Description: Apply an upscale factor ti frames after generation and interpolation.
- Default: 1.0
- Range: 1 - 2

## Updates

### Version 1.4

Added upscale frame after generation and interpolation

### Version 1.3

Added opticalflow frame iterpolation

### Version 1.2

Update experimental node for video duration extension

### Version 1.1

New experimental node for video duration extension

### Version 1.0

First release

