import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mlep import MLEPExtractor
from src.utils.visualization import plot_multiscale_entropy

def generate_sample_image(size=(256, 256)):
    """Generate a sample synthetic image with some patterns."""
    img = Image.new("RGB", size, color=(80, 140, 60))
    draw = ImageDraw.Draw(img)
    for y in range(0, size[1], 16):
        for x in range(0, size[0], 16):
            if (x // 16 + y // 16) % 2 == 0:
                c = (random.randint(50, 110), random.randint(110, 170), random.randint(30, 90))
                draw.rectangle([x, y, x + 15, y + 15], fill=c)
    return img

def main():
    print("=" * 60)
    print("Executing MLEP Feature Extraction & Entropy Visualization")
    print("=" * 60)
    
    # 1. Initialize MLEP
    print("\n[1] Initializing MLEP Feature Extractor...")
    mlep = MLEPExtractor(patch_size=2)
    
    # 2. Generate Image
    print("[2] Generating Sample Input Image...")
    img = generate_sample_image()
    
    # Convert to tensor (B, C, H, W)
    img_tensor = torch.tensor(
        [list(img.getdata())], dtype=torch.float32
    ).view(1, img.height, img.width, 3).permute(0, 3, 1, 2) / 255.0

    # 3. Forward Pass through MLEP
    print("[3] Running MLEP Forward Pass (Local Shuffling -> Pyramid -> Entropy)...")
    with torch.no_grad():
        results = mlep(img_tensor)
        
    print(f"    -> Entropy Maps Generated: {len(results['entropy_maps'])} scales")
    for i, emap in enumerate(results['entropy_maps']):
        print(f"       Scale {i+1}: Shape {tuple(emap.shape)}, Max Entropy: {emap.max().item():.2f}")

    # 4. Generate Visualization
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "mlep_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mlep_entropy_heatmaps.png")
    
    print(f"\n[4] Exporting Multi-Scale Entropy Heatmaps to: {out_path}")
    plot_multiscale_entropy(results['mlep_features'], save_path=out_path)
    
    print("\n[5] Automatically opening the generated preview...")
    try:
        path_str = os.path.abspath(out_path)
        import urllib.request
        import webbrowser
        
        # Create a tiny HTML wrapper to guarantee it opens in a web browser
        # instead of the OS default image viewer (like Windows Photo Viewer).
        html_path = path_str + ".html"
        img_url = f"file:{urllib.request.pathname2url(path_str)}"
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<html><body style='margin:0;background:#222;display:flex;justify-content:center;align-items:center;height:100vh;'><img src='{img_url}' style='max-width:100%;max-height:100%;object-fit:contain;'></body></html>")
            
        file_url = f"file:{urllib.request.pathname2url(html_path)}"
        
        if os.name == 'nt':
            try:
                import subprocess
                subprocess.Popen([r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe", html_path])
            except Exception:
                os.startfile(html_path)
        else:
            webbrowser.open(file_url)
            
        print(f"    [Fallback] If it didn't pop up, please Ctrl+Click the URL below:\n    {file_url}")
    except Exception as e:
        print(f"    Failed to auto-open preview: {e}")
    
    print("\n[DONE] MLEP execution completed successfully!\n")

if __name__ == "__main__":
    main()
