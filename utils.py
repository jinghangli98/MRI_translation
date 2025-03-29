import torch
import numpy as np
import torch
import torchvision.transforms as T

def progressive_patch_shuffle(image, num_patches, points=None):
    """
    Apply progressive patch shuffling to an image.
    
    Args:
        image (torch.Tensor): Input image tensor of shape [B, C, H, W]
        num_patches (int or tuple): Number of patches along each dimension (e.g., 2 for 2x2, or (2,2))
        points (torch.Tensor, optional): Point coordinates to transform, shape [B, N, 2]
        
    Returns:
        torch.Tensor: Shuffled image
        torch.Tensor: Transformed points (if points is not None)
    """
    B, C, H, W = image.shape
    
    # Handle different input formats for num_patches
    if isinstance(num_patches, int):
        h_patches, w_patches = num_patches, num_patches
    else:
        h_patches, w_patches = num_patches
    
    # Calculate patch size
    patch_h, patch_w = H // h_patches, W // w_patches
    
    # Reshape image into patches
    patches = image.view(B, C, h_patches, patch_h, w_patches, patch_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(B, h_patches * w_patches, C, patch_h, patch_w)
    
    # Create random permutation indices for each batch
    indices = torch.stack([torch.randperm(h_patches * w_patches) for _ in range(B)], dim=0).to(image.device)
    
    # Shuffle patches
    shuffled_patches = torch.stack([patches[b, indices[b]] for b in range(B)], dim=0)
    
    # Reshape back to image
    shuffled_patches = shuffled_patches.view(B, h_patches, w_patches, C, patch_h, patch_w)
    shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled_image = shuffled_patches.view(B, C, H, W)
    
    # Transform point coordinates if provided
    transformed_points = None
    if points is not None:
        transformed_points = points.clone()
        for b in range(B):
            for p in range(points.shape[1]):
                if torch.all(points[b, p] == 0):  # Skip if point is [0, 0] (no point)
                    continue
                
                # Calculate which patch the point belongs to
                x, y = points[b, p]
                patch_idx_h, patch_idx_w = int(y // patch_h), int(x // patch_w)
                patch_idx = patch_idx_h * w_patches + patch_idx_w
                
                # Find where this patch went in the shuffle
                new_patch_idx = int((indices[b] == patch_idx).nonzero(as_tuple=True)[0])
                new_patch_idx_h, new_patch_idx_w = new_patch_idx // w_patches, new_patch_idx % w_patches
                
                # Calculate relative position within patch
                rel_y, rel_x = y % patch_h, x % patch_w
                
                # Calculate new absolute position
                new_y = new_patch_idx_h * patch_h + rel_y
                new_x = new_patch_idx_w * patch_w + rel_x
                
                transformed_points[b, p, 0] = new_x
                transformed_points[b, p, 1] = new_y
    
    return shuffled_image, transformed_points

def shuffle_images_identically(image1, image2, num_patches):
    """
    Shuffle two images with identical patch permutation.
    
    Args:
        image1 (torch.Tensor): First image tensor of shape [B, C, H, W]
        image2 (torch.Tensor): Second image tensor of shape [B, C, H, W]
        num_patches (int): Number of patches along each dimension
        
    Returns:
        tuple: Two shuffled image tensors
    """
    B, C1, H, W = image1.shape
    _, C2, _, _ = image2.shape
    
    # Handle different input formats for num_patches
    if isinstance(num_patches, int):
        h_patches, w_patches = num_patches, num_patches
    else:
        h_patches, w_patches = num_patches
    
    # Calculate patch size
    patch_h, patch_w = H // h_patches, W // w_patches
    
    # Create SAME random permutation indices for each batch
    # Note: We create once and reuse for both images
    indices = torch.stack([torch.randperm(h_patches * w_patches) for _ in range(B)], dim=0).to(image1.device)
    
    # First image
    patches1 = image1.view(B, C1, h_patches, patch_h, w_patches, patch_w)
    patches1 = patches1.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches1 = patches1.view(B, h_patches * w_patches, C1, patch_h, patch_w)
    
    # Shuffle patches for first image using indices
    shuffled_patches1 = torch.stack([patches1[b, indices[b]] for b in range(B)], dim=0)
    
    # Reshape back to image
    shuffled_patches1 = shuffled_patches1.view(B, h_patches, w_patches, C1, patch_h, patch_w)
    shuffled_patches1 = shuffled_patches1.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled_image1 = shuffled_patches1.view(B, C1, H, W)
    
    # Second image - same exact process but REUSE same indices
    patches2 = image2.view(B, C2, h_patches, patch_h, w_patches, patch_w)
    patches2 = patches2.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches2 = patches2.view(B, h_patches * w_patches, C2, patch_h, patch_w)
    
    # Shuffle patches for second image using SAME indices
    shuffled_patches2 = torch.stack([patches2[b, indices[b]] for b in range(B)], dim=0)
    
    # Reshape back to image
    shuffled_patches2 = shuffled_patches2.view(B, h_patches, w_patches, C2, patch_h, patch_w)
    shuffled_patches2 = shuffled_patches2.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled_image2 = shuffled_patches2.view(B, C2, H, W)
    
    return shuffled_image1, shuffled_image2