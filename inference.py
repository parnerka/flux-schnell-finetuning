import os
import sys
sys.path.append('./ai-toolkit')
from collections import OrderedDict
from diffusers import AutoPipelineForText2Image
import torch

weights_path = "./output/my_first_flux_lora_v1"

pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline.load_lora_weights(weights_path, weight_name="my_first_flux_lora_v1.safetensors")

image = pipeline('Subject is an astronaut, in ocean').images[0]
image.save(f'{weights_path}/output.png')