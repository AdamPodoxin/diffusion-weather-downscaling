from pathlib import Path

from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline
from diffusers.models.unets.unet_2d import UNet2DOutput
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

import torch
import xarray as xr

from tqdm import tqdm

from utils import (
    load_model_state_dict,
    get_4channel_vqvae,
    get_4channel_unet,
    get_lora_unet,
    generate_batches,
)


class WeatherLDMSuperResolutionPipeline():
    def __init__(
            self, 
            vqvae_path: Path | str="models/vqvae-trained/vqvae-trained.pt", 
            unet_path: Path | str="models/unet-trained-vanilla/unet-trained-vanilla.pt",
            batch_size=100,
        ):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Running on", self.device)

        self.vqvae = get_4channel_vqvae(self.device)
        vqvae_state_dict = load_model_state_dict(vqvae_path)
        self.vqvae.load_state_dict(vqvae_state_dict)

        self.unet = get_lora_unet(get_4channel_unet(self.device))
        unet_state_dict = load_model_state_dict(unet_path)
        self.unet.load_state_dict(unet_state_dict)

        self.scheduler = DDIMScheduler \
                        .from_pretrained("CompVis/ldm-super-resolution-4x-openimages", subfolder="scheduler")

        
        self.batch_size = batch_size


    def _process_batch(self, X: torch.Tensor, num_inference_steps=100):
        """Returns denormalized output as well as raw output (normalized)"""

        # Code derived from LDMSuperResolutionPipeline

        self.vqvae.eval()
        self.unet.eval()

        with torch.no_grad():
            height = X.shape[2]
            width = X.shape[3]

            X = X.to(self.device)

            means_base = X \
                .mean(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1)
            means_lr = means_base.expand(self.batch_size, 4, height, width)
            means_hr = means_base.expand(self.batch_size, 4, height * 4, width * 4)

            stds_base = X \
                .std(dim=(0, 2, 3)) \
                .view(1, 4, 1, 1)
            stds_lr = stds_base.expand(self.batch_size, 4, height, width)
            stds_hr = stds_base.expand(self.batch_size, 4, height * 4, width * 4)
            
            X_normalized = (X - means_lr) / stds_lr

            latents_shape = (self.batch_size, 4, height, width)
            latents_dtype = next(self.unet.parameters()).dtype

            initial_latents = randn_tensor(
                shape=latents_shape,
                dtype=latents_dtype,
                device=self.device,
            ) * self.scheduler.init_noise_sigma

            self.scheduler.set_timesteps(num_inference_steps)
            timesteps_tensor = self.scheduler.timesteps

            latents = initial_latents

            for t in tqdm(timesteps_tensor):
                latents_input = self.scheduler.scale_model_input(
                    sample=torch.cat([latents, X_normalized], dim=1),
                    timestep=t,
                )

                unet_output: UNet2DOutput = self.unet(latents_input, t)
                noise_predicted = unet_output.sample

                latents = self.scheduler.step(
                    model_output=noise_predicted, 
                    timestep=t, 
                    sample=latents,
                ).prev_sample

                torch.cuda.empty_cache()
            
            Y = self.vqvae.decode(latents).sample
            Y_denormalized = Y * stds_hr + means_hr

            del latents
            del latents_input
            del noise_predicted
            torch.cuda.empty_cache()

        return Y_denormalized, Y


    def __call__(self, data: xr.DataArray):
        batch_generator = generate_batches(data, self.batch_size)

        def loop():
            for X in batch_generator:
                yield self._process_batch(X)
                torch.cuda.empty_cache()

        Ys_denormalized, Ys = zip(*list(loop()))

        Y_denormalized = torch.cat(list(Ys_denormalized), dim=0)
        Y = torch.cat(list(Ys), dim=0)

        return Y_denormalized, Y
