#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:32:58 2024

@author: josemacalintal
"""



#%% Packages 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
# from torchmetrics.functional import structural_similarity_index_measure as torch_ssim
from pytorch_msssim import ssim, ms_ssim

#%%  Defininig Loss / Objective Functions 

# Custom Loss Function combining MAE and MS-SSIM
class ReconstructionLoss_MAE_SSIM(nn.Module):
    def __init__(self, alpha=0.5):
        super(ReconstructionLoss_MAE_SSIM, self).__init__()
        self.alpha = alpha  # Weight for balancing MAE and MS-SSIM
        self.mae_loss = nn.L1Loss()

    def forward(self, input_image, output_image, target_image):
        # Compute the reconstruction
        # Input Image = Degraded IMage 
        # Output Image = The Output of the autoencoder 
        # Target Image = Reference Image 
        
        reconstruction = input_image + output_image

        # Mean Absolute Error (MAE) between reconstruction and target
        mae = self.mae_loss(reconstruction, target_image)
        
        # MS-SSIM Loss between reconstruction and target
        ssim_loss = 1 - ssim(reconstruction, target_image, data_range=1, size_average=True)
        
        # Combine MAE and MS-SSIM Loss
        total_loss = self.alpha * mae + (1 - self.alpha) * ssim_loss
        return total_loss
    
class ReconstructionLoss_MSE_SSIM(nn.Module):
    def __init__(self, alpha=0.5):
        super(ReconstructionLoss_MSE_SSIM, self).__init__()
        self.alpha = alpha  # Weight for balancing MSE and SSIM
        self.mae_loss = nn.MSELoss()

    def forward(self, input_image, output_image, target_image):
        # Compute the reconstruction
        # Input Image = Degraded IMage 
        # Output Image = The Output of the autoencoder 
        # Target Image = Reference Image 
        
        reconstruction = input_image + output_image

        # Mean Absolute Error (MAE) between reconstruction and target
        mae = self.mae_loss(reconstruction, target_image)
        
        # MS-SSIM Loss between reconstruction and target
        ssim_loss = 1 - ssim(reconstruction, target_image, data_range=1, size_average=True)
        
        # Combine MAE and MS-SSIM Loss
        total_loss = self.alpha * mae + (1 - self.alpha) * ssim_loss
        return total_loss
    
# Custom Loss Function Using MSE Loss 
class ReconstructionLoss_MSE(nn.Module):
    def __init__(self):
        super(ReconstructionLoss_MSE, self).__init__()
        
        self.mse_loss = nn.MSELoss()

    def forward(self, input_image, output_image, target_image):
        
        reconstruction = input_image + output_image

        # Mean Square Error (MAE) between reconstruction and target
        mse = self.mse_loss(reconstruction, target_image)
        
        return mse
    
    
# Custom Loss Function Using MSE Loss 
class ReconstructionLoss_MAE(nn.Module):
    def __init__(self):
        super(ReconstructionLoss_MAE, self).__init__()
        
        self.mae_loss = nn.L1Loss()

    def forward(self, input_image, output_image, target_image):
        
        reconstruction = input_image + output_image

        # Mean Absolute Error (MAE) between reconstruction and target
        mae = self.mae_loss(reconstruction, target_image)
        
        return mae
    
    
class Reconstruction_WeightedMSELoss(nn.Module):
    def __init__(self, high_weight=1.0, low_weight=0.0, threshold=0.0, epsilon  = 1e-8):
        super(Reconstruction_WeightedMSELoss, self).__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, input_image, output_image, target_image):
        
        # High Pass Reference
        high_pass_reference = target_image - input_image
        
        # Create a weight matrix based on the user-defined threshold
        weight = torch.where(torch.abs(high_pass_reference) > self.threshold, self.high_weight, self.low_weight)
        
        
        # Calculate the weighted MSE loss
        weighted_mse = torch.sum(weight * (output_image - high_pass_reference) ** 2) / (torch.sum(weight) + self.epsilon)

        
        return weighted_mse
    
class ReconstructionLoss_WeightedMSE_SSIM(nn.Module): 
    def __init__(self, high_weight=1.0, low_weight=0.0, threshold=0.0, alpha = 0.5, epsilon = 1e-8):
        super(ReconstructionLoss_WeightedMSE_SSIM, self).__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input_image, output_image, target_image):
        
        # High Pass Reference and reconstruction
        high_pass_reference = target_image - input_image
        reconstruction = input_image + output_image
        
        # Create a weight matrix based on the user-defined threshold
        weight = torch.where(torch.abs(high_pass_reference) > self.threshold, self.high_weight, self.low_weight)
        
        # Calculate the weighted MSE loss
        weighted_mse = torch.sum(weight * (output_image - high_pass_reference) ** 2) / (torch.sum(weight) + self.epsilon)

        # Calculate the SSIM Loss 
        ssim_loss = 1 - ssim(reconstruction, target_image, data_range = 1, size_average=True)
        # Total Loss 
        total_loss = self.alpha * weighted_mse  + (1 - self.alpha) * (ssim_loss)
        
        return total_loss
    
    
class ReconstructionLoss_WeightedMAE(nn.Module): 
    def __init__(self, high_weight = 1.0, low_weight = 0.0, threshold = 0.0, epsilon = 1e-8): 
        super(ReconstructionLoss_WeightedMAE, self).__init__()
        self.high_weight = high_weight 
        self.low_weight = low_weight 
        self.threshold = threshold 
        self.epsilon = epsilon 
        
    def forward(self, input_image, output_image, target_image): 
        
        # High Pass Reference 
        high_pass_reference = target_image - input_image 
        
        # Weighting 
        weight = torch.where(torch.abs(high_pass_reference) > self.threshold, self.high_weight, self.low_weight)
        
        #Calculate MAE Weighted Loss 
        weighted_mae = torch.sum(weight * abs(output_image - high_pass_reference)) / (torch.sum(weight) + self.epsilon)
        
        return weighted_mae
    
    
class ReconstructionLoss_WeightedMAE_SSIM(nn.Module): 
    def __init__(self, high_weight=1.0, low_weight=0.0, threshold=0.0, alpha = 0.5, epsilon = 1e-8):
        super(ReconstructionLoss_WeightedMAE_SSIM, self).__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input_image, output_image, target_image):
        
        # High Pass Reference and reconstruction
        high_pass_reference = target_image - input_image
        reconstruction = input_image + output_image
        
        # Create a weight matrix based on the user-defined threshold
        weight = torch.where(torch.abs(high_pass_reference) > self.threshold, self.high_weight, self.low_weight)
        
        # Calculate the weighted MAE loss
        weighted_mae = torch.sum(weight * abs(output_image - high_pass_reference)) / (torch.sum(weight) + self.epsilon)

        # Calculate the SSIM Loss 
        ssim_loss = 1 - ssim(reconstruction, target_image, data_range = 1, size_average=True)
        # Total Loss 
        total_loss = self.alpha * weighted_mae  + (1 - self.alpha) * (ssim_loss)
        
        return total_loss
    
    
    
class ReconstructionLoss_WeightedMSE_MSSIM(nn.Module): 
    def __init__(self, high_weight=1.0, low_weight=0.0, threshold=0.0, alpha = 0.5, epsilon = 1e-8):
        super(ReconstructionLoss_WeightedMSE_MSSIM, self).__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
        self.alpha = alpha
        self.epsilon = epsilon
        
    def forward(self, input_image, output_image, target_image):
        
        # High Pass Reference and reconstruction
        high_pass_reference = target_image - input_image
        reconstruction = input_image + output_image
        
        # Create a weight matrix based on the user-defined threshold
        weight = torch.where(torch.abs(high_pass_reference) > self.threshold, self.high_weight, self.low_weight)
        
        # Calculate the weighted MAE loss
        weighted_mse = torch.sum(weight * (output_image - high_pass_reference)**2) / (torch.sum(weight) + self.epsilon)

        # Calculate the SSIM Loss 
        ssim_loss = 1 - ms_ssim(reconstruction, target_image, data_range = 1, size_average=True)
        # Total Loss 
        total_loss = self.alpha * weighted_mse  + (1 - self.alpha) * (ssim_loss)
        
        return total_loss
    
    
class ReconstructionLoss_WeightedMAE_MSSIM(nn.Module): 
    def __init__(self, high_weight=1.0, low_weight=0.0, threshold=0.0, alpha = 0.5, epsilon = 1e-8):
        super(ReconstructionLoss_WeightedMAE_MSSIM, self).__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
        self.alpha = alpha
        self.epsilon = epsilon
        
    def forward(self, input_image, output_image, target_image):
        
        # High Pass Reference and reconstruction
        high_pass_reference = target_image - input_image
        reconstruction = input_image + output_image
        
        # Create a weight matrix based on the user-defined threshold
        weight = torch.where(torch.abs(high_pass_reference) > self.threshold, self.high_weight, self.low_weight)
        
        # Calculate the weighted MAE loss
        weighted_mae = torch.sum(weight * abs(output_image - high_pass_reference)) / (torch.sum(weight) + self.epsilon)

        # Calculate the SSIM Loss 
        ssim_loss = 1 - ms_ssim(reconstruction, target_image, data_range = 1, size_average=True)
        # Total Loss 
        total_loss = self.alpha * weighted_mae  + (1 - self.alpha) * (ssim_loss)
        
        return total_loss
    
    

class ReconstructionLoss_WeightedMAE_PatchSSIM(nn.Module):
    def __init__(self, high_weight=1.0, low_weight=0.0, threshold=0.0, alpha=0.5, patch_size=64, epsilon = 1e-8):
        super(ReconstructionLoss_WeightedMAE_PatchSSIM, self).__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
        self.alpha = alpha
        self.patch_size = patch_size
        self.epsilon = epsilon
        
    def divide_into_patches(self, image, patch_size):
        """Divide an image into non-overlapping patches."""
        B, C, H, W = image.size()
        # Ensure dimensions are divisible by patch_size
        assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size."
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return patches  # Shape: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    
    def compute_patch_ssim(self, recon_patches, target_patches):
        """Compute SSIM for corresponding patches and return the mean SSIM."""
        B, C, num_patches_h, num_patches_w, patch_size, _ = recon_patches.shape
        ssim_values = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                recon_patch = recon_patches[:, :, i, j].view(B, C, patch_size, patch_size)
                target_patch = target_patches[:, :, i, j].view(B, C, patch_size, patch_size)
                ssim_values.append(ssim(recon_patch, target_patch, data_range=1.0, size_average = True))  # Assuming normalized to [0, 1]
        return torch.mean(torch.stack(ssim_values))  # Mean SSIM across all patches

    def forward(self, input_image, output_image, target_image):
        # High-Pass Reference and Reconstruction
        high_pass_reference = target_image - input_image
        reconstruction = input_image + output_image

        # Create a weight matrix based on the user-defined threshold
        weight = torch.where(torch.abs(high_pass_reference) > self.threshold, self.high_weight, self.low_weight)

        # Calculate the weighted MAE loss
        weighted_mae = torch.sum(weight * torch.abs(output_image - high_pass_reference)) / (torch.sum(weight) + self.epsilon)

        # Divide images into patches
        recon_patches = self.divide_into_patches(reconstruction, self.patch_size)
        target_patches = self.divide_into_patches(target_image, self.patch_size)

        # Calculate the mean SSIM loss from patches
        ssim_loss = 1 - self.compute_patch_ssim(recon_patches, target_patches)

        # Total Loss
        total_loss = self.alpha * weighted_mae + (1 - self.alpha) * ssim_loss

        return total_loss
    
    
class ReconstructionLoss_WeightedMSE_PatchSSIM(nn.Module):
    def __init__(self, high_weight=1.0, low_weight=0.0, threshold=0.0, alpha=0.5, patch_size=64, epsilon = 1e-8):
        super(ReconstructionLoss_WeightedMSE_PatchSSIM, self).__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
        self.alpha = alpha
        self.patch_size = patch_size
        self.epsilon = epsilon
        
    def divide_into_patches(self, image, patch_size):
        """Divide an image into non-overlapping patches."""
        B, C, H, W = image.size()
        # Ensure dimensions are divisible by patch_size
        assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size."
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return patches  # Shape: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    
    def compute_patch_ssim(self, recon_patches, target_patches):
        """Compute SSIM for corresponding patches and return the mean SSIM."""
        B, C, num_patches_h, num_patches_w, patch_size, _ = recon_patches.shape
        ssim_values = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                recon_patch = recon_patches[:, :, i, j].view(B, C, patch_size, patch_size)
                target_patch = target_patches[:, :, i, j].view(B, C, patch_size, patch_size)
                ssim_values.append(ssim(recon_patch, target_patch, data_range=1.0, size_average = True))  # Assuming normalized to [0, 1]
        return torch.mean(torch.stack(ssim_values))  # Mean SSIM across all patches

    def forward(self, input_image, output_image, target_image):
        # High-Pass Reference and Reconstruction
        high_pass_reference = target_image - input_image
        reconstruction = input_image + output_image

        # Create a weight matrix based on the user-defined threshold
        weight = torch.where(torch.abs(high_pass_reference) > self.threshold, self.high_weight, self.low_weight)

        # Calculate the weighted MAE loss
        weighted_mse = torch.sum(weight * (output_image - high_pass_reference)**2) / (torch.sum(weight)+self.epsilon)

        # Divide images into patches
        recon_patches = self.divide_into_patches(reconstruction, self.patch_size)
        target_patches = self.divide_into_patches(target_image, self.patch_size)

        # Calculate the mean SSIM loss from patches
        ssim_loss = 1 - self.compute_patch_ssim(recon_patches, target_patches)

        # Total Loss
        total_loss = self.alpha * weighted_mse + (1 - self.alpha) * ssim_loss

        return total_loss
