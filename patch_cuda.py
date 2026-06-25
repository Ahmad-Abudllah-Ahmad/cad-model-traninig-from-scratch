import os
# Set OpenMP duplicate library check BEFORE any imports to prevent macOS libomp crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np

# Monkey patch tensor and module cuda methods to redirect to cpu
torch.Tensor.cuda = lambda self, *args, **kwargs: self.to("cpu")
torch.nn.Module.cuda = lambda self, *args, **kwargs: self.to("cpu")

# Monkey patch torch.load to default to CPU map_location
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu'
    return original_load(*args, **kwargs)
torch.load = patched_load

# Map CUDA types to CPU types
torch.cuda.FloatTensor = torch.FloatTensor
torch.cuda.DoubleTensor = torch.DoubleTensor
torch.cuda.is_available = lambda: False

# Monkey patch numpy.int and numpy.float (removed in modern numpy) to standard types
np.int = int
np.float = float
# NOTE: do NOT patch np.bool — it breaks numpy.ma internals

print("[CUDA, NumPy, & OMP Patch] Applied environment configurations and device patches.")
