import torch

print("=" * 50)
print("PyTorch GPU Test")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # Test tensor on GPU
    x = torch.randn(3, 3).cuda()
    print(f"Test tensor created on GPU: {x.device}")
    print("✅ GPU is working!")
else:
    print("❌ GPU not available - running on CPU")
print("=" * 50)
