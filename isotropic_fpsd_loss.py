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
    # print(stride)
    # print(H, W)
    # print(patch_size)

    # Get overlapping patches. Resulting shape: [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
    patches_p = pred.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches_t = target.unfold(2, patch_size, stride).unfold(3, patch_size, stride)  

    # print(patches_p.shape)
    # print(patches_t.shape)


    # Mix batches and patches into a single dimension. Resulting shape: [B * num_patches_h * num_patches_w, C, patch_size, patch_size]
    patches_p = patches_p.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, patch_size, patch_size)
    patches_t = patches_t.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, patch_size, patch_size)

    #Hann windows
    win1d = torch.hann_window(patch_size, device=device)
    win2d = torch.outer(win1d, win1d).unsqueeze(0).unsqueeze(0)

    # Apply HANN windows to each patch. 
    patches_p = patches_p * win2d
    patches_t = patches_t * win2d

    # Handle varying dx across each instances in batch
    if isinstance(dx, torch.Tensor) and dx.numel() == B:    
        num_patches_total = patches_p.shape[0] // B
        dx_expanded = dx.repeat_interleave(num_patches_total).view(-1, 1, 1, 1) # Expand to match the number of patches and channels
    else:
        dx_expanded = dx

    def get_psd(patches):
        f_coeff = torch.fft.fft2(patches)
        return (torch.abs(f_coeff)**2) * (dx_expanded**2) / (patch_size * patch_size) # multiply by dx^2 to convert to physical units, and normalize by patch area
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





def calculate_batch_dx(center_latitudes_deg: torch.Tensor) -> torch.Tensor:
    """
    Calculates the effective dx (in km) for a batch of 128x128 images 
    each 32x32 degrees, based on their center latitudes.
    
    Args:
        center_latitudes_deg: Tensor of shape (B,) containing center latitudes in degrees.
        
    Returns:
        dx_effective: Tensor of shape (B,) containing the effective resolution per pixel in km.
    """
    R_earth = 6371.0  
    degrees_per_image = 32.0 # TO change if necessary 
    pixels_per_image = 128.0
    
    lat_rad = torch.deg2rad(center_latitudes_deg)
    dx_lat = (degrees_per_image / pixels_per_image) * (math.pi / 180.0) * R_earth 
    
     # The effective dx in the longitudinal direction is reduced by the cosine of the latitude
    dx_effective = dx_lat * torch.sqrt(torch.abs(torch.cos(lat_rad)))
    
    return dx_effective