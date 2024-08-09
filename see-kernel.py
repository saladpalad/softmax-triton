import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import time
import os

# Set the environment variable
#os.environ['TORCH_LOGS'] = 'output_code'

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=1)

# Create a sample tensor
sample = torch.randn(100, 100)

# Compile the function 
opt_softmax = torch.compile(torch_softmax)

# Run the optimized function
opt_softmax_out = torch_softmax(sample.cuda()) 
print(opt_softmax_out)
