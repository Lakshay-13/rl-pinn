import torch
import platform
import logging

logger = logging.getLogger(__name__)

def get_available_devices():
    """
    Returns a list of available torch.device objects.
    Priority: CUDA devices, then MPS, then CPU.
    """
    devices = []
    
    # Check CUDA
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            devices.append(torch.device(f"cuda:{i}"))
            
    # Check MPS
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append(torch.device("mps"))
        
    # Always have CPU as fallback
    devices.append(torch.device("cpu"))
    
    return devices

def get_device(prefer_cuda=True, prefer_mps=True):
    """
    Selects the best available device for PyTorch operations.
    Priority: CUDA > MPS > CPU
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    elif prefer_mps and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
             device = torch.device("mps")
             logger.info("MPS (Metal Performance Shaders) is available. Using Apple Silicon GPU.")
        else:
             logger.warning("MPS is available but PyTorch was not built with MPS support. Falling back to CPU.")
             device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info(f"No GPU detected or preferred. Using CPU. (Platform: {platform.system()})")

    return device

def get_device_str(device):
    return str(device)

def get_cuda_memory_weights():
    """
    Returns a list of free memory (in bytes) for each CUDA device.
    For non-CUDA devices, returns a list of 1s.
    """
    weights = []
    if not torch.cuda.is_available():
        return [1.0]
        
    count = torch.cuda.device_count()
    for i in range(count):
        try:
            free, total = torch.cuda.mem_get_info(i)
            # Use a small floor (100MB) to avoid division by zero or assigning to empty cards
            weights.append(max(0.1, float(free) / (1024**3))) 
        except Exception:
            weights.append(1.0)
    return weights
