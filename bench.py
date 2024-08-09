import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import os

def naive_softmax(x: torch.Tensor)-> torch.Tensor:
    """ eager mode """
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
            m_j = max(m_j, curr_x)
            if m_j > m_prev:
                print(f"updated row max is now {m_j}, row = {r}")
            prev_d = d_j
            d_j = prev_d * torch.exp(m_prev - m_j) + torch.exp(x_j - m_j)
        
        x_i = x[r,:]
        m_V = m_j
        d_V = d_j

        y[r,:] = torch.exp(x_i - m_V) / d_V

    return y
            

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

def softmax(x: torch.Tensor)->torch.Tensor:
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
            "Naive",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))

def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

PATH='/home/gabe/softmax'
benchmark.run(show_plots=True, print_data=True, save_path=PATH)
