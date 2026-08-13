import torch
import torch.nn as nn
import torch.fft

class LearnableFrequencyPreFilter(nn.Module):
    """
    Learnable Frequency-Domain Denoising Pre-Filter.
    Uses rFFT2 to apply a parameterized Butterworth mask, stripping high-frequency 
    JPEG quantization blockiness before it reaches the MLEP entropy calculation.
    """
    def __init__(self, img_size=256, initial_cutoff=0.8, initial_slope=10.0):
        super().__init__()
        self.img_size = img_size
        
        # Learnable parameters for the Butterworth filter
        # omega_c controls the cutoff frequency (0 to 1)
        # sigma controls the slope/sharpness of the cutoff
        self.omega_c = nn.Parameter(torch.tensor(initial_cutoff))
        self.sigma = nn.Parameter(torch.tensor(initial_slope))
        
        # Precompute normalized distance matrix for FFT frequencies
        # For rFFT2 of size (256, 256), output is (256, 129)
        fy = torch.fft.fftfreq(img_size)
        fx = torch.fft.rfftfreq(img_size)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        
        # Normalized Euclidean distance from DC component (0,0)
        # Maximum distance in normalized frequency is roughly 0.707 (sqrt(0.5^2 + 0.5^2))
        # We scale it so max freq is ~1.0
        dist = torch.sqrt(gx**2 + gy**2) / 0.7071
        
        # Register as buffer so it moves to GPU automatically
        self.register_buffer('freq_dist', dist.unsqueeze(0).unsqueeze(0)) # (1, 1, H, W)
        
    def _create_mask(self):
        """Generates the Butterworth low-pass mask based on learnable params."""
        # Formula: 1 / (1 + (D / omega_c)^(2 * sigma))
        # Clamp parameters to ensure numerical stability
        omega_c = torch.clamp(self.omega_c, min=0.01, max=1.0)
        sigma = torch.clamp(self.sigma, min=1.0, max=50.0)
        
        # Add epsilon to prevent division by zero
        ratio = self.freq_dist / (omega_c + 1e-8)
        mask = 1.0 / (1.0 + torch.pow(ratio, 2 * sigma))
        return mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        
        # Upcast to float32 for FFT (ComplexHalf is buggy on RTX 40-series)
        x_f32 = x.float()
        
        # Transform image to frequency domain
        x_fft = torch.fft.rfft2(x_f32, norm='ortho')
        
        # Generate the continuous Butterworth mask
        mask = self._create_mask()
        
        # Apply mask
        x_filtered_fft = x_fft * mask
        
        # Inverse transform back to spatial domain
        x_filtered = torch.fft.irfft2(x_filtered_fft, s=(x.shape[2], x.shape[3]), norm='ortho')
        
        # Ensure values stay in valid image range and cast back
        return torch.clamp(x_filtered, 0.0, 255.0).to(orig_dtype)
