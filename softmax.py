import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import time
import os
#os.environ['TORCH_LOGS'] = 'output_code'
#TORCH_LOGS = "OUTPUT_CODE" 

@torch.compile
def torch_opt_softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=1)

def naive_softmax(x: torch.Tensor)-> torch.Tensor:
    """eager mode"""
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:,None]
    num = torch.exp(safe_x)
    denom = num.sum(dim=1)
    out = num/denom[:,None]
    return out

def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """ online softmax, 2.5x faster than eager """
    rows, cols = x.shape
    assert x.dim()==2, "only 2d tensors"
    y = torch.zeros_like(x)

    for r in range(rows): # i <- 1,V
        m_j = float("-inf") # row max 
        d_j = 0 # normalizer term
        for c in range(cols): #j <- 1,V
            x_j = x[r,c]
            m_prev = m_j
            m_j = max(m_prev, x_j)
            #if m_j > m_prev:
            #    print(f"updated row max is now {m_j}, row = {r}")
            prev_d = d_j
            d_j = prev_d * torch.exp(m_prev - m_j) + torch.exp(x_j - m_j)
        
        x_i = x[r,:] # update curr row
        m_V = m_j # max val of the row
        d_V = d_j # running norm after 1 row

        y[r,:] = torch.exp(x_i - m_V) / d_V

    return y

@triton.jit
def _online_softmax_fwd_kernel(x_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr):
    pid_0 = tl.program_id(0)
    log2_e = 1.44269504
    
    # batch refers to rows
    batch_range = tl.arange(0, B0) + pid_0*B0
    batch_mask = batch_range < N0

    #exp2 = lambda x: tl.tl.exp(log2_e * x)

    prev_max = tl.zeros((B0,), dtype=tl.float32)
    max_batch = tl.zeros((B0,), dtype=tl.float32)
    denom = tl.zeros((B0,), dtype=tl.float32)

    for i in range(0, T, B1):
        # normal setup
        b_offset = tl.arange(0,B1) + i
        offset_mask = b_offset < T

        x_range = batch_range[None,:] * T + b_offset[:,None]
        x_mask = batch_mask[None,:] & offset_mask[:,None]

        x_batch = tl.load(x_ptr + x_range, x_mask, other=0) #32x1

        #now we keep track of running max and norm term
        x_batch_max = tl.max(x_batch)
        prev_max = max_batch

        # update batch of max vals
        max_batch = tl.maximum(x_batch_max, prev_max) #returns a tensor

        # keep track of running normalization term
        denom = denom * tl.exp(prev_max - max_batch) + tl.sum(tl.exp(x_batch - max_batch), axis=0)

    for i in range(0, T, B1):
        # normal setup again
        b_offset = tl.arange(0,B1) + i
        offset_mask = b_offset < T

        x_range = batch_range[None,:] * T + b_offset[:,None]
        x_mask = batch_mask[None,:] & offset_mask[:,None]

        x_batch = tl.load(x_ptr + x_range, x_mask, other=0) #32x1

        ### calculate the output matrix z...
        z = tl.exp(x_batch - max_batch) / denom

        tl.store(z_ptr + x_range, z, x_mask)



def triton_online_softmax(x: torch.Tensor) -> torch.Tensor:
    """ Triton impl of online softmax """
    rows, cols = x.shape
    B0 = triton.next_power_of_2(rows)
    B1 = triton.next_power_of_2(cols) # 1000 cols -> 1024 block_size
    num_warps = 4 # 32 threads
    if B1 >= 2048:
        num_warps = 8
    if B1 >= 4096:
        num_warps = 16

    grid = (rows,) # launch a kernel for each row

    # allocate output buffer
    sm_out = torch.empty_like(x)
    #print(x.stride(0))

    _online_softmax_fwd_kernel[grid](
        x,
        sm_out,
        rows,
        cols,
        B0,
        B1
    #    block_size=block_size,
    #    num_warps=num_warps,
    )

    return sm_out


@triton.jit
def _softmax_fwd_kernel(output_ptr, stride_output_row, input_ptr, stride_input_row, num_cols, block_size: tl.constexpr):
    
    # setup input ptrs
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + (row_idx * stride_input_row)
    row_range = tl.arange(0, block_size) # col offsets

    row_mask = row_range < num_cols
   
    # load from DRAM to SRAM
    row = tl.load(row_start_ptr + row_range, mask=row_mask, other=float("-inf")) # (row_size,1)
    
    # softmax itself
    row_max = tl.max(row, axis=0)
    safe_max = row - row_max
    num = tl.exp(safe_max)
    denom = tl.sum(num, axis=0)
    sm_out = num/denom
    # write back to DRAM
    output_start_ptr = output_ptr + (row_idx * stride_output_row)
    tl.store(output_start_ptr + row_range, sm_out, row_mask)

def classic_softmax(x: torch.Tensor)->torch.Tensor:
    """ Triton impl of softmax, fwd pass only """
    rows, cols = x.shape
    assert x.dim() == 2, f"2d tensors only"
    block_size = triton.next_power_of_2(cols) # 1000 cols -> 1024 block_size
    num_warps = 4 # 32 threads
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16

    grid = (rows,) # launch a kernel for each row

    # allocate output buffer
    sm_out = torch.empty_like(x)

    _softmax_fwd_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps,
    )

    return sm_out


sample = torch.tensor([[1,2,3,4,5], [5,4,3,2,1]], dtype=torch.float32, device='cuda')

start = time.perf_counter()
torch_out = F.softmax(sample,dim=1)
end = time.perf_counter()
print(f"{torch_out=}")
torch_time = end-start


start = time.perf_counter()
eager_out = naive_softmax(sample)
end = time.perf_counter()
print(f"{eager_out=}")
eager_time = end-start

start = time.perf_counter()
online_out = online_softmax(sample)
end = time.perf_counter()
print(f"{online_out=}")
online_time = end-start

start = time.perf_counter()
triton_out = classic_softmax(sample)
end = time.perf_counter()
print(f"{triton_out=}")
triton_time = end-start

start = time.perf_counter()
triton_online_out = triton_online_softmax(sample)
end = time.perf_counter()
print(f"{triton_online_out=}")
triton_online_time = end - start

start = time.perf_counter()
torch_opt_out = torch_opt_softmax(sample) 
end = time.perf_counter()
print(f"{torch_opt_out=}")
torch_opt_time = end-start

print(f"{torch_time=}\n{eager_time=}\n{triton_time=}\n{online_time=}\n{triton_online_time=}\n{torch_opt_time=}\n")
