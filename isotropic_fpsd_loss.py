import torch
import torch.nn.functional as F
import math

def isotropic_psd_loss(pred: torch.Tensor, target: torch.Tensor, dx, num_patches_per_side = 32, eps=1e-8):
    """
    Computes the weighted PSD loss for multichannel climate data.
    
    Args:
        pred, target: Tensors of shape (B, C, H, W)
        eps: Small constant for numerical stability in log 
        dx: physical distance per pixel (km) (IMPORTANT)
    """
    B, C, H, W = pred.shape
    device = pred.device
    # For now, we will just use a fixed patch size and stride

    patch_size =  min(H, W) // num_patches_per_side


    stride = patch_size // 2 
    print(stride)
    print(H, W)
    print(patch_size)

    # Get overlapping patches. Resulting shape: [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
    patches_p = pred.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches_t = target.unfold(2, patch_size, stride).unfold(3, patch_size, stride)  

    print(patches_p.shape)
    print(patches_t.shape)


    # Mix batches and patches into a single dimension. Resulting shape: [B * num_patches_h * num_patches_w, C, patch_size, patch_size]
    patches_p = patches_p.contiguous().view(-1, C, patch_size, patch_size)
    patches_t = patches_t.contiguous().view(-1, C, patch_size, patch_size)

    #Hann windows
    win1d = torch.hann_window(patch_size, device=device)
    win2d = torch.outer(win1d, win1d).unsqueeze(0).unsqueeze(0)

    # Apply HANN windows to each patch. 
    patches_p = patches_p * win2d
    patches_t = patches_t * win2d

    def get_psd(patches):
        f_coeff = torch.fft.fft2(patches)
        # Normalize by the patch area, not the global H * W
        return torch.abs(f_coeff)**2 / (patch_size * patch_size * dx)

    psd_pred = get_psd(patches_p)
    psd_target = get_psd(patches_t)

    # Compute Isotropic Weights based on local patch frequencies
    freq = torch.fft.fftfreq(patch_size, device=device)
    grid_h, grid_w = torch.meshgrid(freq, freq, indexing='ij')
    
    k = torch.sqrt(grid_h**2 + grid_w**2)
    k_max = k.max() # Max frequency in the patch
    
    # Higher weights toward higher frequencies 
    weights = (k / k_max)**2 
    weights = weights.unsqueeze(0).unsqueeze(0)

    log_p_pred = torch.log(psd_pred + eps)
    log_p_target = torch.log(psd_target + eps)

    diff = (log_p_pred - log_p_target)**2
    weighted_diff = diff * weights # Apply isotropic weights to the squared log differences

    return torch.sqrt(torch.mean(weighted_diff))




# Redundant function for sampling Cartesian patches from lat/lon images. We can use this to preprocess the data before feeding it into the loss function, ensuring that the input to the loss is already in a Cartesian grid format.
def sample_cartesian_patch(img_latlon, center_lat, center_lon, patch_size, dx_km):
    """
    Resamples a lat/lon image to a Cartesian grid with equal spatial resolution.
    
    Args:
        img_latlon: Tensor of shape (B, C, H_lat, W_lon) - assumes global coverage
                    Latitudes from 90 to -90 (top to bottom)
                    Longitudes from -180 to 180 (left to right)
        center_lat, center_lon: Center of the patch in degrees
        patch_size: Tuple (H_patch, W_patch) in pixels
        dx_km: Spatial resolution per pixel in km
    """
    B, C, H, W = img_latlon.shape
    device = img_latlon.device
    R_earth = 6371.0 # Radius of Earth in km

    H_patch, W_patch = patch_size
    y = torch.arange(-H_patch//2, H_patch//2, device=device) * dx_km
    x = torch.arange(-W_patch//2, W_patch//2, device=device) * dx_km
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Using a simplified equirectangular approximation
    lat_offset = (grid_y / R_earth) * (180.0 / math.pi)
    
    cos_lat = math.cos(center_lat * math.pi / 180.0)
    lon_offset = (grid_x / (R_earth * cos_lat)) * (180.0 / math.pi)
    target_lat = center_lat - lat_offset # Minus because image Y goes down
    target_lon = center_lon + lon_offset

    # Lat: 90 to -90 maps to -1 to 1
    norm_y = -(target_lat / 90.0) 
    # Lon: -180 to 180 maps to -1 to 1
    norm_x = target_lon / 180.0
    
    # Resulting shape (B, H_patch, W_patch, 2)
    grid = torch.stack((norm_x, norm_y), dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    patch_cartesian = F.grid_sample(img_latlon, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return patch_cartesian
