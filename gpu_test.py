import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
    
    # Test tensor on GPU
    tensor = torch.rand(3, 3).to("cuda")
    print("Tensor on GPU:", tensor)
else:
    print("CUDA is not available. Check your installation.")
