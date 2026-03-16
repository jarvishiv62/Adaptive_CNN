import torch
import sys

print("=== GPU Detection Report ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

print(f"\nCUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute: {props.major}.{props.minor}")
else:
    print("\n=== Troubleshooting ===")
    print("If you have an NVIDIA GPU, ensure:")
    print("1. NVIDIA GPU drivers are installed")
    print("2. CUDA toolkit is installed (check: nvcc --version)")
    print("3. PyTorch was installed with CUDA support")
    print("\nInstall CUDA-enabled PyTorch:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Check if we can force CUDA device
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDefault device: {device}")
    
    # Test tensor on CUDA if available
    if torch.cuda.is_available():
        x = torch.randn(1, 3, 224, 224).cuda()
        print(f"✅ CUDA tensor test: {x.device}")
except Exception as e:
    print(f"❌ CUDA test failed: {e}")
