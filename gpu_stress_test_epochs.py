import torch
import time
import sys
import os

# Filter out non-NVIDIA GPUs if any (redundant but safe)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gpu_stress_test(epochs=1000, size=10240):
    """
    Runs an EXTREME GPU stress test for 1000 epochs.
    Explicitly targets NVIDIA GeForce RTX 5070 Ti.
    """
    print(f"\n{'='*60}")
    print(f"🚀 EXTREME NVIDIA GPU STRESS TEST")
    print(f"{'='*60}")
    
    if not torch.cuda.is_available():
        print("❌ CRITICAL ERROR: NVIDIA GPU (CUDA) NOT DETECTED!")
        print("Falling back to CPU would be a failure. Aborting.")
        return

    # Explicitly select device 0 (NVIDIA RTX 5070 Ti)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    
    print(f"🔥 TARGET DEVICE: {props.name}")
    print(f"📊 TOTAL VRAM  : {props.total_memory / 1e9:.2f} GB")
    print(f"⚙️  CORES (CC)  : {props.major}.{props.minor}")
    print(f"🔄 TOTAL EPOCHS: {epochs}")
    print(f"💎 MATRIX SIZE : {size}x{size} (High Precision)")
    
    print(f"\n[1/2] Allocating massive tensors on VRAM...")
    try:
        # Using float32 for high compute load
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        print("✅ Allocation Successful.")
    except RuntimeError as e:
        print(f"❌ OUT OF MEMORY: Matrix size {size} is too large for {props.total_memory / 1e9:.2f} GB VRAM.")
        print("Reducing size to 8192...")
        size = 8192
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

    print(f"\n[2/2] Starting Execution Loop (100% Load Goal)...")
    
    total_start = time.time()
    
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Perform multiple heavy multiplications to keep the GPU pipeline full
            # This ensures Task Manager/GPU-Z shows 100% load
            for _ in range(5):
                c = torch.matmul(a, b)
                # Add some non-linear ops to diversify load
                c = torch.tanh(c)
            
            # Synchronize to wait for GPU to finish work before measuring time
            torch.cuda.synchronize()
            
            epoch_duration = time.time() - epoch_start
            
            # Print status every 10 epochs or so for cleaner log
            if epoch == 1 or epoch % 10 == 0:
                print(f"⏳ Epoch {epoch:4d}/{epochs} | Step Time: {epoch_duration:.4f}s | Status: MAX LOAD")

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🏁 STRESS TEST COMPLETE")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Speed : {epochs/total_time:.2f} epochs/sec")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    gpu_stress_test(epochs=1000)
