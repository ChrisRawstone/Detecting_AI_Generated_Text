import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. PyTorch version: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create a tensor and move it to the GPU
    tensor = torch.rand(3, 3)
    tensor_gpu = tensor.to('cuda')
    
    print("Tensor on GPU:")
    print(tensor_gpu)
    
    # Perform a simple operation on the GPU
    result = tensor_gpu * 2
    print("Result after multiplying by 2 on GPU:")
    print(result)
else:
    print("CUDA is not available. Running on CPU.")
