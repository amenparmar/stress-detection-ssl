import torch
import time
import sys

def check_cuda():
    print("Checking CUDA availability...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available. This project requires a CUDA-enabled GPU.")
    
    device_count = torch.cuda.device_count()
    print(f"CUDA is available! Found {device_count} device(s).")
    
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

def stress_test(duration=60):
    print(f"\nStarting GPU stress test for {duration} seconds...")
    print("Monitor your Task Manager/GPU usage now.")
    
    # Use a large matrix size to generate load
    size = 10000 
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    start_time = time.time()
    iterations = 0
    
    try:
        while time.time() - start_time < duration:
            # Perform matrix multiplication
            c = torch.matmul(a, b)
            # Synchronization is needed for accurate timing, but here we just want load
            torch.cuda.synchronize() 
            iterations += 1
            if iterations % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Running... {elapsed:.1f}s / {duration}s", end='\r')
                
    except KeyboardInterrupt:
        print("\nStress test stopped by user.")
        
    print(f"\nStress test completed. Performed {iterations} heavy matrix multiplications.")

if __name__ == "__main__":
    check_cuda()
    # Optional: Allow user to specify duration
    duration = 60
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            pass
    stress_test(duration)
