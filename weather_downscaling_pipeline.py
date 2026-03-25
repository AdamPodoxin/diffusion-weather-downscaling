from pathlib import Path

from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion_superresolution import LDMSuperResolutionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import torch
import xarray as xr

from utils import (
    load_model_state_dict,
    get_4channel_vqvae,
    get_4channel_unet,
    get_lora_unet,
    normalize_across_channels, 
    denormalize_across_channels,
    generate_batches,
)


class WeatherLDMSuperResolutionPipeline():
    def __init__(
            self, 
            vqvae_path: Path | str="models/vqvae-trained/vqvae-trained.pt", 
            unet_path: Path | str="models/unet-trained-vanilla/unet-trained-vanilla.pt",
            batch_size=100
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

        self.ldm_pipeline = LDMSuperResolutionPipeline(
                                vqvae=self.vqvae, 
                                unet=self.unet, 
                                scheduler=self.scheduler
                            )
        
        self.batch_size = batch_size
    

    def _process_batch(self, X: torch.Tensor):
        X_normalized, X_means, X_stds = normalize_across_channels(X)

        Y_normalized = self.ldm_pipeline(
                            image=X_normalized, 
                            batch_size=100,
                            output_type="np.array"
                        ).images
        
        Y_normalized_tensor = torch.from_numpy(Y_normalized) \
                            .permute(0, 3, 1, 2) \
                            .to(self.device)
        
        Y = denormalize_across_channels(Y_normalized_tensor, X_means, X_stds)

        return Y
    

    def process(self, data: xr.DataArray):
        batch_generator = generate_batches(data, self.batch_size)

        def loop():
            for X in batch_generator:
                yield self._process_batch(X)
                torch.cuda.empty_cache()

        Y = torch.cat(list(loop()), dim=0)

        return Y
