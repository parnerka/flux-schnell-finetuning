# import os
# import sys
# sys.path.append('./ai-toolkit')
# from toolkit.job import run_job
# from collections import OrderedDict
# from PIL import Image
# import os
# from collections import OrderedDict

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# job_to_run = OrderedDict([
#     ('job', 'extension'),
#     ('config', OrderedDict([
#         # this name will be the folder and filename name
#         ('name', 'my_first_flux_lora_v1'),
#         ('process', [
#             OrderedDict([
#                 ('type', 'sd_trainer'),
#                 ('log_dir', './logs'),
#                 ('log_config', OrderedDict([
#                     ('log_interval', 10)
#                 ])),
#                 # root folder to save training sessions/samples/weights
#                 ('training_folder', './output'),
#                 ('device', 'cuda:0'),
#                 # if a trigger word is specified, it will be added to captions of training data if it does not already exist
#                 # alternatively, in your captions you can add [trigger] and it will be replaced with the trigger word
#                 # ('trigger_word', 'image'),
#                 ('network', OrderedDict([
#                     ('type', 'lora'),
#                     ('linear', 16),
#                     ('linear_alpha', 16)
#                 ])),
#                 ('save', OrderedDict([
#                     ('dtype', 'float16'),  # precision to save
#                     ('save_every', 250),  # save every this many steps
#                     ('max_step_saves_to_keep', 4)  # how many intermittent saves to keep
#                 ])),
#                 ('datasets', [
#                     # datasets are a folder of images. captions need to be txt files with the same name as the image
#                     # for instance image2.jpg and image2.txt. Only jpg, jpeg, and png are supported currently
#                     # images will automatically be resized and bucketed into the resolution specified
#                     OrderedDict([
#                         ('folder_path', './dataset'),
#                         ('caption_ext', 'txt'),
#                         ('caption_dropout_rate', 0.05),  # will drop out the caption 5% of time
#                         ('shuffle_tokens', False), 
#                         ('cache_latents_to_disk', True),
#                         ('resolution', [512, 768, 1024])  
#                     ])
#                 ]),
#                 ('train', OrderedDict([
#                     ('batch_size', 1),
#                     ('steps', 500),  # total number of steps to train 500 - 4000 is a good range
#                     ('gradient_accumulation_steps', 1),
#                     ('train_unet', True),
#                     ('train_text_encoder', False),  # probably won't work with flux
#                     ('gradient_checkpointing', True),
#                     ('noise_scheduler', 'flowmatch'),  # for training only
#                     ('optimizer', 'adamw8bit'),
#                     ('lr', 1e-4),

#                     # ema will smooth out learning, but could slow it down. Recommended to leave on.
#                     ('ema_config', OrderedDict([
#                         ('use_ema', True),
#                         ('ema_decay', 0.99)
#                     ])),

#                     # will probably need this if gpu supports it for flux, other dtypes may not work correctly
#                     ('dtype', 'bf16')
#                 ])),
#                 ('model', OrderedDict([
#                     # huggingface model name or path
#                     ('name_or_path', 'black-forest-labs/FLUX.1-schnell'),
#                     ('assistant_lora_path', 'ostris/FLUX.1-schnell-training-adapter'),
#                     ('is_flux', True),
#                     ('quantize', True),  
#                 ])),
#                 ('sample', OrderedDict([
#                     ('sampler', 'flowmatch'),  # must match train.noise_scheduler
#                     ('sample_every', 250),  # sample every this many steps
#                     ('width', 1024),
#                     ('height', 1024),
#                     ('prompts', [
#                         # you can add [trigger] to the prompts here and it will be replaced with the trigger word
#                         #'[trigger] holding a sign that says \'I LOVE PROMPTS!\'',
#                         'woman with red hair, playing chess at the park, bomb going off in the background',
#                         'a woman holding a coffee cup, in a beanie, sitting at a cafe',
#                         'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
#                         'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
#                         'a bear building a log cabin in the snow covered mountains',
#                         'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
#                         'hipster man with a beard, building a chair, in a wood shop',
#                         'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
#                         'a man holding a sign that says, \'this is a sign\'',
#                         'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle'
#                     ]),
#                     ('neg', ''),
#                     ('seed', 42),
#                     ('walk_seed', True),
#                     ('guidance_scale', 1), 
#                     ('sample_steps', 4) # 1 - 4 works well
#                 ]))
#             ])
#         ])
#     ])),
#     # you can add any additional meta info here. [name] is replaced with config name at top
#     ('meta', OrderedDict([
#         ('name', '[name]'),
#         ('version', '1.0')
#     ]))
# ])

# run_job(job_to_run)




# Standard library imports
import os
import sys
from collections import OrderedDict
from PIL import Image

# Add Flux AI toolkit to Python path
sys.path.append('./ai-toolkit')
from toolkit.job import run_job

def set_environment_variables():
    """
    Configure environment variables for HuggingFace transfer and albumentations.
    HF_HUB_ENABLE_HF_TRANSFER: Enables HuggingFace model transfer
    NO_ALBUMENTATIONS_UPDATE: Prevents automatic updates of the albumentations library
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def get_model_config():
    """
    Define the base model configuration.
    Returns:
        OrderedDict: Configuration for the FLUX model and its training adapter
    """
    return OrderedDict([
        ('name_or_path', 'black-forest-labs/FLUX.1-schnell'),  # Base model path on HuggingFace
        ('assistant_lora_path', 'ostris/FLUX.1-schnell-training-adapter'),  # Training adapter path
        ('is_flux', True),  # Indicates this is a FLUX model
        ('quantize', True),  # Enable model quantization for reduced memory usage
    ])

def get_network_config():
    """
    Define the LoRA network configuration.
    Returns:
        OrderedDict: LoRA network parameters including rank and alpha values
    """
    return OrderedDict([
        ('type', 'lora'),  # Using LoRA
        ('linear', 16),    # LoRA rank
        ('linear_alpha', 16)  # LoRA scaling factor
    ])

def get_training_config():
    """
    Define the training hyperparameters and configuration.
    Returns:
        OrderedDict: Complete training configuration including optimization parameters
    """
    return OrderedDict([
        ('batch_size', 1),
        ('steps', 500),  # Total training steps (recommended range: 500-4000)
        ('gradient_accumulation_steps', 1),
        ('train_unet', True),  # Enable UNet training
        ('train_text_encoder', False),  # Disable text encoder training (may not work with FLUX)
        ('gradient_checkpointing', True),  # Enable gradient checkpointing
        ('noise_scheduler', 'flowmatch'),  # Noise scheduler type for training
        ('optimizer', 'adamw8bit'),  # 8-bit AdamW optimizer
        ('lr', 1e-4),  # Learning rate
        # EMA configuration
        ('ema_config', OrderedDict([
            ('use_ema', True),
            ('ema_decay', 0.99)
        ])),
        ('dtype', 'bf16')  # Using bfloat16 precision for training
    ])

def get_dataset_config():
    """
    Define the dataset configuration for training.
    Returns:
        list: Dataset configuration including path and processing parameters
    """
    return [OrderedDict([
        ('folder_path', './dataset'),  # Path to training images
        ('caption_ext', 'txt'),        # Caption file extension
        ('caption_dropout_rate', 0.05), # Probability of dropping captions during training
        ('shuffle_tokens', False),      # Disable token shuffling
        ('cache_latents_to_disk', True), # Cache latent representations for faster training
        ('resolution', [512, 768, 1024])  # Supported image resolutions for bucketing
    ])]

def get_sampling_prompts():
    """
    Define the prompts used for generating samples during training.
    Returns:
        list: Collection of diverse prompts for testing model capabilities
    """
    return [
        'woman with red hair, playing chess at the park, bomb going off in the background',
        'a woman holding a coffee cup, in a beanie, sitting at a cafe',
        'a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini',
        'a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background',
        'a bear building a log cabin in the snow covered mountains',
        'woman playing the guitar, on stage, singing a song, laser lights, punk rocker',
        'hipster man with a beard, building a chair, in a wood shop',
        'photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop',
        'a man holding a sign that says, \'this is a sign\'',
        'a bulldog, in a post apocalyptic world, with a shotgun, in a leather jacket, in a desert, with a motorcycle'
    ]

def get_sampling_config():
    """
    Define the configuration for generating samples during training.
    Returns:
        OrderedDict: Sample generation parameters including dimensions and sampling settings
    """
    return OrderedDict([
        ('sampler', 'flowmatch'),  # Must match training noise scheduler
        ('sample_every', 250),     # Generate samples every 250 steps
        ('width', 1024),           # Sample width in pixels
        ('height', 1024),          # Sample height in pixels
        ('prompts', get_sampling_prompts()),
        ('neg', ''),               # Negative prompt (empty)
        ('seed', 42),              # Initial random seed
        ('walk_seed', True),       # Enable seed walking for varied samples
        ('guidance_scale', 1),     # Classifier-free guidance scale
        ('sample_steps', 4)        # Number of sampling steps (1-4 recommended)
    ])

def create_job_config():
    """
    Create the complete job configuration by combining all components.
    Returns:
        OrderedDict: Complete job configuration including all training parameters
    """
    return OrderedDict([
        ('job', 'extension'),
        ('config', OrderedDict([
            ('name', 'my_first_flux_lora_v1'),  # Project name
            ('process', [
                OrderedDict([
                    ('type', 'sd_trainer'),  # Using stable diffusion trainer
                    ('log_dir', './logs'),   # Directory for training logs
                    ('log_config', OrderedDict([('log_interval', 10)])),  # Log every 10 steps
                    ('training_folder', './output'),  # Output directory for checkpoints and samples
                    ('device', 'cuda:0'),    # GPU device to use
                    ('network', get_network_config()),
                    ('save', OrderedDict([
                        ('dtype', 'float16'),  # Save checkpoints in float16
                        ('save_every', 250),   # Save checkpoint every 250 steps
                        ('max_step_saves_to_keep', 4)  # Keep last 4 checkpoints
                    ])),
                    ('datasets', get_dataset_config()),
                    ('train', get_training_config()),
                    ('model', get_model_config()),
                    ('sample', get_sampling_config())
                ])
            ])
        ])),
        ('meta', OrderedDict([
            ('name', '[name]'),    # Meta name (replaced with config name)
            ('version', '1.0')     # Version number
        ]))
    ])

def main():
    """
    Main entry point of the script.
    Sets up environment and runs the training job.
    """
    # Set up environment variables
    set_environment_variables()
    
    # Create and execute the training job
    job_to_run = create_job_config()
    run_job(job_to_run)

if __name__ == "__main__":
    main()