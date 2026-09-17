import torch


def is_npu_available() -> bool:
    """Detect Ascend NPU availability without requiring the transformers package."""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except Exception:
        return False


def get_total_gpu_memory() -> int:
    """Returns total device memory in GBs."""
    device = infer_device()
    if device == "cuda":
        return torch.cuda.get_device_properties(0).total_memory // (1024**3)
    elif device == "xpu":
        return torch.xpu.get_device_properties(0).total_memory // (1024**3)
    elif device == "npu":
        return torch.npu.get_device_properties(0).total_memory // (1024**3)
    else:
        raise RuntimeError(f"Unsupported device: {device}")


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        return "cuda"
    # Use Ascend NPU if available (torch.npu)
    elif is_npu_available():
        return "npu"
    # XPU (Intel) if available
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"
