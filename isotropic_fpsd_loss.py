import torch
import torch.nn.functional as F
import math

def isotropic_psd_loss(pred: torch.Tensor, target: torch.Tensor, dx, eps=1e-8):
    """
    Computes the weighted PSD loss for multichannel climate data.
    
    Args:
        pred, target: Tensors of shape (B, C, H, W)
        eps: Small constant for numerical stability in log 
        dx: physical distance per pixel (km) (IMPORTANT)
    """
    B, C, H, W = pred.shape
    device = pred.device

    #wavenumbers
    kh = torch.linspace(0, H - 1, H, device=device)
    kw = torch.linspace(0, W - 1, W, device=device)
    grid_h, grid_w = torch.meshgrid(kh, kw, indexing='ij')
    
    k = torch.sqrt(grid_h**2 + grid_w**2)
    k_max = torch.sqrt(torch.tensor((H - 1)**2 + (W - 1)**2, device=device))

    weights = (k / k_max)**2 

    # The paper uses the squared magnitude of fourier
    def get_psd(img):
        f_coeff = torch.fft.fft2(img)

        return torch.abs(f_coeff)**2 / (H * W * dx)

    psd_pred = get_psd(pred)
    psd_target = get_psd(target)
    log_p_pred = torch.log(psd_pred + eps)
    log_p_target = torch.log(psd_target + eps)
    
    diff = (log_p_pred - log_p_target)**2
    
    weighted_diff = diff * weights.unsqueeze(0).unsqueeze(0)

    return torch.sqrt(torch.mean(weighted_diff))




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
