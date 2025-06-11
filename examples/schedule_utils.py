# DashGaussian Training Scheduler for gsplat
# Adapted from the original DashGaussian implementation

import math
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

class GsplatTrainingScheduler:
    """
    DashGaussian training scheduler adapted for gsplat.
    Handles progressive resolution scheduling and primitive number control.
    """
    def __init__(
        self, 
        cfg,  # gsplat Config object
        splats: torch.nn.ParameterDict,  # gsplat's splats dictionary
        train_images: List[torch.Tensor],  # List of training images
        max_steps: Optional[int] = None,
        densify_until_iter: Optional[int] = None,
        densification_interval: Optional[int] = None,
    ):
        # Basic configuration
        self.cfg = cfg
        self.max_steps = max_steps or cfg.max_steps
        self.init_n_gaussian = splats["means"].shape[0]
        
        # Fix the densify_until_iter to be appropriate for max_steps
        if densify_until_iter is None:
            self.densify_until_iter = int(self.max_steps * 0.5)  # 3500 for 7000 steps
        else:
            # Ensure it's not beyond our training steps
            self.densify_until_iter = min(densify_until_iter, int(self.max_steps * 0.8))
        
        # Densification settings
        self.densify_mode = "freq"  # DashGaussian uses frequency-based mode
        self.densification_interval = densification_interval or 100
        
        # Resolution settings
        self.resolution_mode = "freq"  # Frequency-based resolution scheduling
        self.start_significance_factor = 4
        self.max_reso_scale = 8
        self.reso_sample_num = 32  # Must be no less than 2
        self.max_densify_rate_per_step = 0.2
        self.reso_scales = None
        self.reso_level_significance = None
        self.reso_level_begin = None
        
        # CRITICAL: Set increase_reso_until to match your training schedule
        self.increase_reso_until = self.densify_until_iter
        self.next_i = 2
        
        # Momentum-based primitive budgeting
        if hasattr(cfg, 'max_n_gaussian') and cfg.max_n_gaussian > 0:
            self.max_n_gaussian = cfg.max_n_gaussian
            self.momentum = -1
        else:
            self.momentum = 5 * self.init_n_gaussian
            self.max_n_gaussian = self.init_n_gaussian + self.momentum
            self.integrate_factor = 0.98
            self.momentum_step_cap = 1000000
        
        print(f"[DASHGAUSSIAN] Configuration:")
        print(f"  - Max steps: {self.max_steps}")
        print(f"  - Densify until: {self.densify_until_iter}")
        print(f"  - Initial gaussians: {self.init_n_gaussian}")
        print(f"  - Max gaussians: {self.max_n_gaussian}")
        
        # Initialize resolution scheduler
        self.init_reso_scheduler(train_images)
        
        # Store lr decay iteration for gsplat integration
        self.lr_decay_from_iter_value = self._compute_lr_decay_from_iter()
        
        # Print debug info about resolution schedule
        print(f"[DASHGAUSSIAN] Resolution Schedule:")
        print(f"  - Initial resolution scale: {self.get_res_scale(0)}")
        print(f"  - Resolution at 25% training: {self.get_res_scale(int(self.max_steps * 0.25))}")
        print(f"  - Resolution at 50% training: {self.get_res_scale(int(self.max_steps * 0.50))}")
        print(f"  - Resolution at 75% training: {self.get_res_scale(int(self.max_steps * 0.75))}")
        print(f"  - Resolution at final step: {self.get_res_scale(self.max_steps - 1)}")
        print(f"  - LR decay starts at: {self.lr_decay_from_iter_value}")
    
    def update_momentum(self, momentum_step: int):
        """Update momentum-based primitive budget."""
        if self.momentum == -1:
            return
        self.momentum = max(
            self.momentum,
            int(self.integrate_factor * self.momentum + min(self.momentum_step_cap, momentum_step))
        )
        self.max_n_gaussian = self.init_n_gaussian + self.momentum
    
    def get_res_scale(self, iteration: int) -> float:
        """Get resolution scale for current iteration."""
        if self.resolution_mode == "const":
            return 1.0
        elif self.resolution_mode == "freq":
            if iteration >= self.increase_reso_until:
                return 1.0
            if iteration < self.reso_level_begin[1]:
                return float(self.reso_scales[0])
            
            # Find appropriate resolution level
            current_i = self.next_i
            for idx in range(1, len(self.reso_level_begin)):
                if iteration < self.reso_level_begin[idx]:
                    current_i = idx
                    break
            
            i = current_i - 1
            if i >= len(self.reso_level_begin) - 1:
                return 1.0
                
            i_now, i_nxt = self.reso_level_begin[i:i + 2]
            s_lst, s_now = self.reso_scales[i - 1:i + 1]
            
            # Interpolate scale
            progress = (iteration - i_now) / max(i_nxt - i_now, 1)
            scale = (1 / (progress * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
            return float(int(scale))
        else:
            raise NotImplementedError(f"Resolution mode '{self.resolution_mode}' is not implemented.")
    
    def get_densify_rate(self, iteration: int, cur_n_gaussian: int, cur_scale: Optional[float] = None) -> float:
        """Get densification rate for current iteration."""
        if self.densify_mode == "free":
            return 1.0
        elif self.densify_mode == "freq":
            assert cur_scale is not None, "Current scale must be provided for frequency-based densification"
            
            if self.densification_interval + iteration < self.increase_reso_until:
                # Progressive densification based on resolution
                progress = iteration / self.densify_until_iter
                scale_factor = cur_scale**(2 - progress)
                next_n_gaussian = int(
                    (self.max_n_gaussian - self.init_n_gaussian) / scale_factor
                ) + self.init_n_gaussian
            else:
                next_n_gaussian = self.max_n_gaussian
            
            # Calculate rate with cap
            if cur_n_gaussian >= next_n_gaussian:
                return 0.0
            rate = (next_n_gaussian - cur_n_gaussian) / max(cur_n_gaussian, 1)
            return min(max(rate, 0.0), self.max_densify_rate_per_step)
        else:
            raise NotImplementedError(f"Densify mode '{self.densify_mode}' is not implemented.")
    
    def _compute_lr_decay_from_iter(self) -> int:
        """Compute iteration from which to start LR decay."""
        if self.resolution_mode == "const":
            return 1
        # Find when we reach a reasonable resolution (e.g., scale < 2)
        for i, s in zip(self.reso_level_begin, self.reso_scales):
            if s < 2:
                return min(i, int(self.max_steps * 0.7))  # Don't delay too much
        # Default to 70% of training if not found
        return int(self.max_steps * 0.7)
    
    def init_reso_scheduler(self, train_images: List[torch.Tensor]):
        """Initialize resolution scheduler based on frequency analysis of training images."""
        if self.resolution_mode != "freq":
            print(f"[ INFO ] Skipped resolution scheduler initialization, the resolution mode is {self.resolution_mode}")
            return
        
        def compute_win_significance(significance_map: torch.Tensor, scale: float) -> float:
            """Compute significance for a window at given scale."""
            h, w = significance_map.shape[-2:]
            c = ((h + 1) // 2, (w + 1) // 2)
            win_size = (max(1, int(h / scale)), max(1, int(w / scale)))
            
            # Extract window with bounds checking
            y_start = max(0, c[0] - win_size[0] // 2)
            y_end = min(h, c[0] + win_size[0] // 2)
            x_start = max(0, c[1] - win_size[1] // 2)
            x_end = min(w, c[1] + win_size[1] // 2)
            
            if y_end <= y_start or x_end <= x_start:
                return 0.0
                
            win_significance = significance_map[..., y_start:y_end, x_start:x_end].sum().item()
            return win_significance
        
        def scale_solver(significance_map: torch.Tensor, target_significance: float) -> float:
            """Binary search to find scale that gives target significance."""
            L, R, T = 0.0, 1.0, 64
            for _ in range(T):
                mid = (L + R) / 2
                win_significance = compute_win_significance(significance_map, 1 / mid)
                if win_significance < target_significance:
                    L = mid
                else:
                    R = mid
            return 1 / mid
        
        print("[ INFO ] Initializing DashGaussian resolution scheduler...")
        
        self.max_reso_scale = 8
        self.next_i = 2
        scene_freq_image = None
        
        # Process all training images
        for img_tensor in train_images:
            # Handle different image formats
            if img_tensor.dim() == 4:  # [B, C, H, W]
                img = img_tensor[0]
            elif img_tensor.dim() == 3:  # [C, H, W] or [H, W, C]
                if img_tensor.shape[-1] == 3:  # [H, W, C]
                    img = img_tensor.permute(2, 0, 1)
                else:  # [C, H, W]
                    img = img_tensor
            else:
                continue
            
            # Convert to float if needed
            if img.dtype != torch.float32:
                img = img.float()
            
            # Compute FFT
            img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
            img_fft_centered_mod = (img_fft_centered.real.square() + 
                                  img_fft_centered.imag.square()).sqrt()
            
            # Accumulate frequency information
            if scene_freq_image is None:
                scene_freq_image = img_fft_centered_mod
            else:
                scene_freq_image = scene_freq_image + img_fft_centered_mod
            
            # Update max resolution scale based on image
            e_total = img_fft_centered_mod.sum().item()
            if e_total > 0:
                e_min = e_total / self.start_significance_factor
                computed_scale = scale_solver(img_fft_centered_mod, e_min)
                self.max_reso_scale = min(self.max_reso_scale, computed_scale)
        
        # Use logarithm as modulation function
        modulation_func = math.log
        
        # Initialize schedule arrays
        self.reso_scales = []
        self.reso_level_significance = []
        self.reso_level_begin = []
        
        # Average frequency information
        scene_freq_image /= len(train_images)
        E_total = scene_freq_image.sum().item()
        E_min = compute_win_significance(scene_freq_image, self.max_reso_scale)
        
        if E_total <= 0 or E_min <= 0:
            print("[ WARNING ] Invalid frequency analysis, using default schedule")
            self.reso_scales = [8.0, 4.0, 2.0, 1.0]
            self.reso_level_begin = [0, int(self.increase_reso_until * 0.33), 
                                   int(self.increase_reso_until * 0.66), 
                                   self.increase_reso_until]
            return
        
        # Build resolution schedule compressed to fit within increase_reso_until
        self.reso_level_significance.append(E_min)
        self.reso_scales.append(self.max_reso_scale)
        self.reso_level_begin.append(0)
        
        # Use fewer levels for shorter training
        num_levels = min(self.reso_sample_num, max(4, self.increase_reso_until // 500))
        
        for i in range(1, num_levels - 1):
            # Linear interpolation of significance levels
            sig = (E_total - E_min) * i / (num_levels - 1) + E_min
            self.reso_level_significance.append(sig)
            self.reso_scales.append(scale_solver(scene_freq_image, sig))
            
            # Modulate and compute iteration boundaries
            if self.reso_level_significance[-2] > 0 and E_min > 0:
                self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
                iter_begin = int(self.increase_reso_until * self.reso_level_significance[-2] / 
                               modulation_func(E_total / E_min))
                self.reso_level_begin.append(iter_begin)
        
        # Final level (full resolution)
        self.reso_level_significance.append(modulation_func(E_total / E_min))
        self.reso_scales.append(1.0)
        if len(self.reso_level_significance) > 1 and self.reso_level_significance[-2] > 0:
            self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
            self.reso_level_begin.append(
                int(self.increase_reso_until * 0.9)  # Ensure we have some full-res training
            )
        self.reso_level_begin.append(self.increase_reso_until)
        
        # Ensure monotonic increasing iteration boundaries
        for i in range(1, len(self.reso_level_begin)):
            if self.reso_level_begin[i] <= self.reso_level_begin[i-1]:
                self.reso_level_begin[i] = self.reso_level_begin[i-1] + 100
        
        print(f"[ INFO ] Resolution schedule initialized with {len(self.reso_scales)} levels")
        print(f"[ INFO ] Max resolution scale: {self.max_reso_scale:.2f}")
        print(f"[ INFO ] Resolution will reach 1.0 at iteration {self.increase_reso_until}")
        
        # Debug: print first few resolution transitions
        for i in range(min(5, len(self.reso_scales))):
            print(f"  - Scale {self.reso_scales[i]:.1f} from iteration {self.reso_level_begin[i]}")

    def get_current_max_gaussians(self) -> int:
        """Get current maximum number of Gaussians."""
        return self.max_n_gaussian
    
    def should_densify(self, iteration: int) -> bool:
        """Check if densification should happen at this iteration."""
        return (iteration < self.densify_until_iter and 
                iteration % self.densification_interval == 0 and
                iteration > 500)  # Don't densify too early