import torch

def get_device():
    """
    Get the best available device for ML computations.
    
    Returns:
        torch.device: The best available device (MPS on Apple Silicon, CPU otherwise)
    """
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

# Global device instance
DEVICE = get_device()