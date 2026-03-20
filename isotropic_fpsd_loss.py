import torch

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
