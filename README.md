# Softmax in triton (Numerically Stable and Online Version)

## Run
```
python3 softmax.py
```

## Results
```
torch_out=tensor([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364],
        [0.6364, 0.2341, 0.0861, 0.0317, 0.0117]], device='cuda:0')
eager_out=tensor([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364],
        [0.6364, 0.2341, 0.0861, 0.0317, 0.0117]], device='cuda:0')
online_out=tensor([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364],
        [0.6364, 0.2341, 0.0861, 0.0317, 0.0117]], device='cuda:0')
triton_out=tensor([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364],
        [0.6364, 0.2341, 0.0861, 0.0317, 0.0117]], device='cuda:0')
triton_online_out=tensor([[0.0115, 0.0313, 0.0850, 0.2311, 0.6283],
        [0.6283, 0.2311, 0.0850, 0.0313, 0.0115]], device='cuda:0')
torch_opt_out=tensor([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364],
        [0.6364, 0.2341, 0.0861, 0.0317, 0.0117]], device='cuda:0')

torch_time=0.2707262650001212
eager_time=1.4357375380000121
triton_time=1.0801465019999341
online_time=0.10670110000000932
triton_online_time=0.021721599999636965
torch_opt_time=1.0795402640001157

Triton online softmax speedup over torch softmax:  12.46x
```
