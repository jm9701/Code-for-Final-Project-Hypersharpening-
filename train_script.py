#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:53:30 2024

@author: josemacalintal

model 4 - history changes
1. changed all activation functions to be PReLu 
2. added a new loss function (weighted)
    a. Modified the Loss Functions 
3. changed the upsampling feature to bicubic interpolation
"""

import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from pytorch_msssim import ssim
import torch.optim as optim
import sys 
import os 

#%% 
if '.' not in sys.path: 
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import callable_autoencoder_models
import callable_loss_functions

#%% Hyperparameter setting 
model_number = "Test"
# model_number = 7
batch_size = 8 # Default is 64
num_epochs = 500
# alpha_loss = 0.75 # For Mixed MSE/MAE_SSIM Loss Function if used 
alpha_loss = "NA"  
lr = 0.0005 # Learning Rate for Adam Optimizer
mask_threshold= 0.001 #Less than 1 percent reflectance Default 0.005
patch_ssim = "NA"




gpu_device = 1
loss_name = "MSE"
# weight_decay_param = 1e-5

model_name = f"model{model_number}_{loss_name}_alpha{alpha_loss}_lr{lr}_BatchSize{batch_size}_maskthr{mask_threshold}_PatchSSIM{patch_ssim}_TotalEpoch{num_epochs}"
model_type = f"model{model_number}"

# dataset = "patch_data"
# dataset = "whole_data"
dataset = "whole_image_data"

# source = '/Volumes/Dissertation Data/'
source = '/home/jm9701/'

# DPrint Descriptions
print("Using Dataset: ", dataset)
print("Source: ", source)
print("Batch Size: ", batch_size)
print("Model Number: ", model_number)
print("Loss Used: ", loss_name)
print("Alpha (if used): ", alpha_loss)
print("Mask Threshold: ", mask_threshold)
print("Learning Rate: ", lr)
print("GPU Device: ", gpu_device)
print("Executing Model: ", model_name)

#%% Initializing the Model 
print("Model and Loss Initialization...")

# Change What Model you are running 
if model_number == 4:
    autoencoder = callable_autoencoder_models.ConvAutoencoder_Model4()
    model_type = "model4"
    
elif model_number == 5:
    autoencoder = callable_autoencoder_models.ConvAutoencoder_Model5()
    model_type = "model5"
    
elif model_number == 6: 
    autoencoder = callable_autoencoder_models.ConvAutoencoder_Model6()
    model_type = "model4"

elif model_number == 7: 
    autoencoder = callable_autoencoder_models.ConvAutoencoder_Model7()
    
elif model_number == "Test": 
    autoencoder = callable_autoencoder_models.ConvAutoencoder_TestModel()
    
    
# Optimizer
optimizer  = optim.Adam(autoencoder.parameters(), lr = lr) #, weight_decay=weight_decay_param)



# Change what Loss Function you are using 
if loss_name == "WeightedMSE": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.Reconstruction_WeightedMSELoss(threshold = mask_threshold)
    
elif loss_name == "WeightedMSE_SSIM": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_WeightedMSE_SSIM(threshold=mask_threshold, alpha=alpha_loss)
    
elif loss_name == "WeightedMAE": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_WeightedMAE(threshold = mask_threshold)
    
elif loss_name == "WeightedMAE_SSIM": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_WeightedMAE_SSIM(threshold=mask_threshold, alpha=alpha_loss)
    
elif loss_name == "MSE": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_MSE()
    
elif loss_name == "MAE": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_MAE()
    
elif loss_name == "MSE_SSIM": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_MSE_SSIM(alpha = alpha_loss)    
    
elif loss_name == "MAE_SSIM": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_MAE_SSIM(alpha = alpha_loss)    
    
elif loss_name == "WeightedMAE_MSSIM": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_WeightedMAE_MSSIM(threshold=mask_threshold, alpha = alpha_loss)
    
elif loss_name == "WeightedMSE_MSSIM": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_WeightedMSE_MSSIM(threshold=mask_threshold, alpha = alpha_loss)
    
elif loss_name == "WeightedMAE_PatchSSIM": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_WeightedMAE_PatchSSIM(threshold=mask_threshold, alpha = alpha_loss, patch_size=patch_ssim)
    
elif loss_name == "WeightedMSE_PatchSSIM": 
    print(f"Using {loss_name}")
    criterion = callable_loss_functions.ReconstructionLoss_WeightedMSE_PatchSSIM(threshold=mask_threshold, alpha = alpha_loss, patch_size=patch_ssim)


else: 
    raise KeyboardInterrupt("Loss Function Not Available. Review Loss")


# criterion = ReconstructionLoss_MAE_SSIM(alpha = alpha_loss)
# criterion = ReconstructionLoss_MSE_SSIM(alpha = alpha_loss)
# criterion = ReconstructionLoss_MSE()



#%% Choose Dataset

if dataset == "patch_data": 
    # VNIR Original - Train
    path = source + "Datasets/Cultural Heritage/Neural Network Projects/Symeon/VNIR/Train/VNIR_patches_3D_array_patchsize128.npz"
    patches_train = np.load(path)['train_data']
    
    
    # VNIR Degraded - Train
    path = source + "Datasets/Cultural Heritage/Neural Network Projects/Symeon/VNIR/Train/VNIR_deg_patches_3D_array_patchsize128.npz"
    deg_patches_train = np.load(path)['train_data']
    
    
    # VNIR Original - Validation 
    path = source + "Datasets/Cultural Heritage/Neural Network Projects/Symeon/VNIR/Validation/VNIR_patches_3D_array_patchsize128.npz"
    patches_val = np.load(path)['val_data']
    
    # VNIR Degaraded - Validation 
    path = source + "Datasets/Cultural Heritage/Neural Network Projects/Symeon/VNIR/Validation/VNIR_deg_patches_3D_array_patchsize128.npz"
    deg_patches_val = np.load(path)['val_data']
    
if dataset == "whole_data": 
    # VNIR - Mid Resolution
    path = source + "Datasets/Cultural Heritage/Neural Network Projects/Symeon/VNIR/VNIR_degraded_patches_3D_array_patchsize128.npz"
    patches_train = np.load(path)['VNIR_patches']
    
    # VNIR - Low Resolution Degraded
    path = source + "Datasets/Cultural Heritage/Neural Network Projects/Symeon/VNIR/VNIR_low_degraded_patches_3D_array_patchsize128.npz"
    deg_patches_train = np.load(path)['VNIR_patches']
    
    # SWIR - Mid Resolution Original 
    path = source + "Datasets/Cultural Heritage/Neural Network Projects/Symeon/SWIR/SWIR_patches_3D_array_patchsize128.npz"
    patches_val = np.load(path)['SWIR_patches']
    
    # SWIR - Low Resolution - Adjusted 
    path = source + "Datasets/Cultural Heritage/Neural Network Projects/Symeon/SWIR/SWIR_degraded_patches_3D_array_patchsize_adjusted128.npz" 
    deg_patches_val = np.load(path)['SWIR_patches']
    
if dataset == "whole_image_data": #No patches, uses the entire image per band 
    path = source + "Datasets/Cultural Heritage/Symeon/Degraded Products/VNIR/midres_vnir_band_images.npz"
    patches_train = np.load(path)['vnir_imgs']
    
    # VNIR - Low Resolution Degraded
    path = source + "Datasets/Cultural Heritage/Symeon/Degraded Products/VNIR/lowres_vnir_band_images.npz"
    deg_patches_train = np.load(path)['vnir_imgs']
    
    # SWIR - Mid Resolution Original 
    path = source + "Datasets/Cultural Heritage/Symeon/Degraded Products/SWIR/original_swir_band_images.npz"
    patches_val = np.load(path)['swir_imgs']
    
    # SWIR - Low Resolution - Adjusted 
    path = source + "Datasets/Cultural Heritage/Symeon/Degraded Products/SWIR/SWIR_lowres_swir_band_images.npz" 
    deg_patches_val = np.load(path)['swir_imgs']

#%% Converting Dataset into proper tensors 
print("Dataset and Dataloader Preparation...")

# Train Tensors
ref_train_tensor = torch.tensor(patches_train, dtype = torch.float32).unsqueeze(1)
deg_train_tensor = torch.tensor(deg_patches_train, dtype = torch.float32).unsqueeze(1)

# Validation Tensors
ref_val_tensor = torch.tensor(patches_val, dtype = torch.float32).unsqueeze(1)
deg_val_tensor = torch.tensor(deg_patches_val, dtype = torch.float32).unsqueeze(1)

# Dataset Generation 
train_dataset = TensorDataset(deg_train_tensor, ref_train_tensor)
val_dataset = TensorDataset(deg_val_tensor, ref_val_tensor)

# Dataloaders 
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)



#%% Preparing to Board onto GPU
print("Boarding to GPU... ")


# Boarding to GPU if avaialble
# Syntax: "cuda:1" for selecting a GPU zero-index
torch.cuda.set_device(gpu_device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)

# Loss and Validation Curve Saving 
train_record, val_record = [], [] 

# To save the best model based on validation loss
best_val_loss = float('inf')

print("Model Summary: ", summary(autoencoder, input_size = (1, 768, 768)))

#%% Training Begin
# DPrint Descriptions

print("== Model Parameters == ")
print("Using Dataset: ", dataset)
print("Source: ", source)
print("Batch Size: ", batch_size)
print("Model Number: ", model_number)
print("Loss Used: ", loss_name)
print("Alpha (if used): ", alpha_loss)
print("Mask Threshold: ", mask_threshold)
print("Learning Rate: ", lr)
print("Executing Model: ", model_name)

print("Training Begin...")

# Training loop
for epoch in range(num_epochs):
    autoencoder.train()
    train_loss = 0.0
    
    for input_image, target_image in train_loader:
        # Move data to the correct device
        input_image, target_image = input_image.to(device), target_image.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output_image = autoencoder(input_image)
        
        # Compute Loss
        loss = criterion(input_image, output_image, target_image)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate the training loss
        train_loss += loss.item() * input_image.size(0)
    
    # Average training loss for the epoch
    train_loss /= len(train_loader.dataset)
    
    # Validation loop
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_image, target_image in val_loader:
            input_image, target_image = input_image.to(device), target_image.to(device)
            
            # Forward pass
            output_image = autoencoder(input_image)
            
            # Compute loss
            loss = criterion(input_image, output_image, target_image)
            
            # Accumulate the validation loss
            val_loss += loss.item() * input_image.size(0)
    
    # Average validation loss for the epoch
    val_loss /= len(val_loader.dataset)
    
    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = source + f'Datasets/Cultural Heritage/Neural Network Projects/Code/Saved Weights/{model_type}/best_weights_{model_name}.pth'
        torch.save(autoencoder.state_dict(), save_path)
        print(f"New best model found at epoch {epoch+1}. Saving model with validation loss: {val_loss:.8f}")
    
    # Record training and validation losses
    train_record.append(train_loss)
    val_record.append(val_loss)
    
    # Print the training and validation loss for the current epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Curr Best Val Loss {best_val_loss:.8f}")

print("Training complete.")
print(f"Best model saved with validation loss: {best_val_loss:.8f}")
#%% Saving the model weights at the last epoch 
print(f"Saving the last weights of {model_name}. ")
save_path = source + f'Datasets/Cultural Heritage/Neural Network Projects/Code/Saved Weights/{model_type}/last_weights_{model_name}.pth'
torch.save(autoencoder.state_dict(), save_path)


#%% Saving the validation and train loss 
print("Saving the train and validation loss")
#File paths to save the records
train_record_path = source + f'Datasets/Cultural Heritage/Neural Network Projects/Code/Train_Val_Records/{model_type}/{model_name}_train_record.txt'
val_record_path = source + f'Datasets/Cultural Heritage/Neural Network Projects/Code/Train_Val_Records/{model_type}/{model_name}_val_record.txt'

# Save train and validation loss records to text files
with open(train_record_path, 'w') as f:
    for loss in train_record:
        f.write(f"{loss}\n")

with open(val_record_path, 'w') as f:
    for loss in val_record:
        f.write(f"{loss}\n")

print(f"Training loss record saved to {train_record_path}")
print(f"Validation loss record saved to {val_record_path}")


