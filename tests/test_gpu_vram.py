import torch
import time
from src.models.hydrafusion_net import HydraFusionNet

def test_vram_and_speed():
    if not torch.cuda.is_available():
        print("CUDA not available! Test failed.")
        return
        
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    # Initialize model
    model = HydraFusionNet().to(device)
    model.train()
    
    batch_size = 16
    img_size = 256
    dummy_input = torch.rand((batch_size, 3, img_size, img_size), device=device) * 255.0
    dummy_labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"\nRunning 10 forward/backward passes with batch_size={batch_size}, AMP=fp16...")
    
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()
    
    for i in range(10):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _, _ = model(dummy_input, stage=2)
            loss = criterion(logits, dummy_labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    end_time = time.time()
    
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9
    avg_time = (end_time - start_time) / 10
    
    print(f"Peak VRAM Usage: {peak_vram:.2f} GB")
    print(f"Avg Time per Batch: {avg_time:.3f} s ({batch_size/avg_time:.1f} imgs/sec)")
    
    assert peak_vram < 5.5, f"VRAM exceeds RTX 4050 limit! Used {peak_vram:.2f} GB"
    print("\n✅ VRAM Budget Verification Passed!")

if __name__ == "__main__":
    test_vram_and_speed()
