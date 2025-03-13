import torch
import math
import numpy as np

class Model(torch.nn.Module):
    def __init__(self, params, state_dict=None):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(10, 10)  # defined but not used in forward
        self.baseline = True
        self.diff = params.get('diff', True)
        self.shift = params.get('shift', 0)
        # No need for num_cores in a GPU vectorized version

    def forward(self, x):
        """
        x: Tensor of shape (B, T) assumed to be on the correct device.
        Depending on self.diff, the Hurst exponent is computed using either the
        "random_walk" (differenced) or "change" (cumulative sum) approach.
        """
        # Choose the computation type
        kind = 'random_walk' if self.diff else 'change'
        try:
            # Compute Hurst exponents for each series (batched) on GPU
            H = compute_Hc_batch(x, kind=kind)
        except Exception as e:
            # In case of error, fallback to a default value of 1
            H = torch.ones(x.size(0), device=x.device)
        # Apply shift and ensure output has shape (B, 1)
        return (H + self.shift).unsqueeze(1)

def compute_Hc_batch(x, kind='random_walk', min_window=10):
    """
    Compute the Hurst exponent H for a batch of time series.
    
    Parameters:
      x: Tensor of shape (B, T) where T is the series length.
      kind: 'random_walk' (for differences) or 'change' (for cumulative sum).
      min_window: minimal window size (default 10).
      
    Returns:
      A tensor H of shape (B,) containing the estimated Hurst exponent per series.
    """
    B, T = x.shape
    device = x.device
    max_window = T - 1  # as in the original implementation

    # Determine window sizes similarly to the original implementation.
    ws = []
    start = math.log10(min_window)
    end = math.log10(max_window)
    step = 0.25
    cur = start
    while cur < end:
        w = int(10 ** cur)
        if w >= min_window and w <= max_window:
            ws.append(w)
        cur += step
    if T not in ws:
        ws.append(T)
    ws = sorted(set(ws))
    K = len(ws)
    # Convert window sizes to a tensor (float for later log computation)
    ws_tensor = torch.tensor(ws, dtype=torch.float32, device=device)

    eps = 1e-6  # small constant to avoid division by zero
    RS_list = []  # to collect RS value for each window size

    for w in ws:
        w = int(w)
        num_segments = T // w  # non-overlapping segments
        if num_segments < 1:
            RS_list.append(torch.full((B,), 0.0, device=device))
            continue
        # Reshape each series into segments of length w: shape (B, num_segments, w)
        segments = x[:, :num_segments * w].reshape(B, num_segments, w)
        
        if kind == 'random_walk':
            # Compute differences along each segment (mimics __to_inc)
            seg_diff = segments[..., 1:] - segments[..., :-1]  # shape (B, num_segments, w-1)
            # Compute mean increment per segment: (last-first)/(w-1)
            mean_inc = (segments[..., -1] - segments[..., 0]) / (w - 1)
            # Subtract mean increment from differences
            deviations = seg_diff - mean_inc.unsqueeze(-1)
        elif kind == 'change':
            # For the "change" case, work directly with the segment values.
            deviations = segments - segments.mean(dim=-1, keepdim=True)
        else:
            raise ValueError("Unknown kind: choose 'random_walk' or 'change'")
        
        # Compute cumulative sum along the last dimension
        Z = torch.cumsum(deviations, dim=-1)
        # Compute range R = max(Z) - min(Z) along the segment dimension
        R_val, _ = torch.max(Z, dim=-1)
        R_min, _ = torch.min(Z, dim=-1)
        R_range = R_val - R_min  # shape (B, num_segments)
        
        # Compute standard deviation of the increments used.
        if kind == 'random_walk':
            S_val = torch.std(seg_diff, dim=-1, unbiased=True)
        elif kind == 'change':
            S_val = torch.std(segments, dim=-1, unbiased=True)
        
        # RS value per segment (adding eps to avoid div-by-zero)
        RS_seg = R_range / (S_val + eps)  # shape (B, num_segments)
        # Average RS over segments for each series
        RS_avg = torch.mean(RS_seg, dim=-1)  # shape (B,)
        RS_list.append(RS_avg)
    
    # Stack RS values for each window size: shape (B, K)
    RS_tensor = torch.stack(RS_list, dim=1)
    # Take log10 of window sizes and RS values (add eps to avoid log(0))
    X = torch.log10(ws_tensor + eps)   # shape (K,)
    Y = torch.log10(RS_tensor + eps)     # shape (B, K)

    # Perform linear regression for each series:
    # For each series, we want to solve: log10(RS) = H * log10(window) + log10(c)
    meanX = X.mean()
    X_centered = X - meanX   # shape (K,)
    denom = torch.sum(X_centered ** 2)  # scalar
    # Compute the slope H for each series
    Y_mean = Y.mean(dim=1, keepdim=True)  # shape (B, 1)
    H = torch.sum((Y - Y_mean) * X_centered, dim=1) / (denom + eps)
    
    # For safety, replace any NaN or inf values with 1
    H[torch.isnan(H)] = 1.0
    H[torch.isinf(H)] = 1.0
    return H