#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:26:04 2024

@author: josemacalintal

Callable Class Autoencoder Models 

"""


#%% Packages 

import torch.nn as nn 
import torch

#%% Model 1 - First Model 

class ConvAutoencoder_Model1(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_Model1, self).__init__()
        
        # Encoder: 5 blocks with Conv + ReLU + MaxPool layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Block 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Block 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Block 3
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Block 4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Block 5
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
        )

        # Decoder: 5 blocks with Conv + ReLU + Upsample layers
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Block 1
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4x4 -> 8x8

            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Block 2
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8x8 -> 16x16

            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Block 3
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32

            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Block 4
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64

            nn.Conv2d(32, 1, kernel_size=3, padding=1),# Block 5 (Output Layer)
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
            
            nn.Conv2d(1, 1, kernel_size=3, padding = 1), #Output Block
            nn.Tanh()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#%% Model 2 

class ConvAutoencoder_Model2(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_Model2, self).__init__()
        
        # Encoder: 5 blocks with Conv + ReLU + MaxPool layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Block 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Block 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Block 3
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Block 4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Block 5
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
        )

        # Decoder: 5 blocks with Conv + ReLU + Upsample layers
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Block 1
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4x4 -> 8x8

            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Block 2
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8x8 -> 16x16

            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Block 3
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32

            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Block 4
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64

            nn.Conv2d(32, 1, kernel_size=3, padding=1),# Block 5 (Output Layer)
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
            
            nn.Conv2d(1, 1, kernel_size=3, padding = 1), #Output Block
            nn.Tanh()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#%% Model 3 

class ConvAutoencoder_Model3(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_Model3, self).__init__()
        
        # Encoder: 5 blocks with Conv + BatchNorm + ReLU + MaxPool layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Block 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Block 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Block 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Block 4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Block 5
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
        )

        # Decoder: 5 blocks with Conv + BatchNorm + PReLU + Upsample layers
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Block 1
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4x4 -> 8x8

            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Block 2
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8x8 -> 16x16

            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Block 3
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32

            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Block 4
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64

            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Block 5 (Output Layer)
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
            
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Output Block
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#%% Model 4 

class ConvAutoencoder_Model4(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_Model4, self).__init__()
        
        # Encoder: 5 blocks with Conv + BatchNorm + ReLU + MaxPool layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Block 1
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Block 2
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Block 3
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Block 4
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Block 5
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
        )

        # Decoder: 5 blocks with Conv + BatchNorm + PReLU + Upsample layers
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Block 1
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),  # 4x4 -> 8x8

            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Block 2
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),  # 8x8 -> 16x16

            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Block 3
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),  # 16x16 -> 32x32

            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Block 4
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),  # 32x32 -> 64x64

            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Block 5 (Output Layer)
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic'),  # 64x64 -> 128x128
            
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Output Block
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 


#%% Model 5 

class ConvAutoencoder_Model5(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_Model5, self).__init__()
        
        # Encoder: 5 blocks with Conv + BatchNorm + PReLU + MaxPool layers
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Block 1
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 128x128 -> 64x64
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Block 2
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Block 3
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )
        
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Block 4
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )
        
        self.encoder_block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Block 5
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )
        
        # Decoder: 5 blocks with Conv + BatchNorm + PReLU + Upsample layers
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Block 1 (512 + 256 -> 768 channels)
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic')  # 4x4 -> 8x8
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),  # Block 2 (256 + 256-> 512 channels)
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic')  # 8x8 -> 16x16
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # Block 3 (128 + 128 ->  256 channels)
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic')  # 16x16 -> 32x32
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),  # Block 4 (64 + 64 -> 128 channels)
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic')  # 32x32 -> 64x64
        )
        
        self.decoder_block5 = nn.Sequential(
            nn.Conv2d(32 + 32, 1, kernel_size=3, padding=1),  # Block 5 (32 + 32 channels)
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic')  # 64x64 -> 128x128
        )
        
        # Final Output Layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder_block1(x)  # 128x128 -> 64x64
        enc2 = self.encoder_block2(enc1)  # 64x64 -> 32x32
        enc3 = self.encoder_block3(enc2)  # 32x32 -> 16x16
        enc4 = self.encoder_block4(enc3)  # 16x16 -> 8x8
        enc5 = self.encoder_block5(enc4)  # 8x8 -> 4x4
        
        # Decoder path with skip connections
        dec1 = self.decoder_block1(enc5)  # 4x4 -> 8x8 
        
        dec2 = self.decoder_block2(torch.cat([dec1, enc4], dim=1))  # 8x8 -> 16x16
        dec3 = self.decoder_block3(torch.cat([dec2, enc3], dim=1))  # 16x16 -> 32x32
        dec4 = self.decoder_block4(torch.cat([dec3, enc2], dim=1))  # 32x32 -> 64x64
        dec5 = self.decoder_block5(torch.cat([dec4, enc1], dim=1))  # 64x64 -> 128x128
        
        
        # Output Layer
        output = self.output_conv(dec5) #Tanh output with 1 channel.
        return output

#%% Model 6 
'''
Trained on the entire image. and treats each photo as a band. 

'''
class ConvAutoencoder_Model6(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_Model6, self).__init__()
        
        # Encoder: 5 blocks with Conv + BatchNorm + PReLU + MaxPool layers
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Block 1 - 64 Channels 
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 768x768x1 -> 384x384x64
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Block 2 - 128 out channels
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 384 x 384->192x192x128
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Block 3 - 256 out channels
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 192 x 192 -> 96 x 96 x 256 
        )
        
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Block 4 - 512 out channels
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 96 x 96-> 48x48x512
        )
        
        self.encoder_block5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # Block 5 - 1024 out channels
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 48x48 -> 24 x 24 x 1024
        )
        
        # Decoder: 5 blocks with Conv + BatchNorm + PReLU + Upsample layers
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # Block 1 (514 out channels)
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 24 x 24 into 48x48x512
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),  # Block 2 (concatenate 512enc + 512dec = 1024 out)
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bicubic')  # 48x48 -> 96 x 96 x 256
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),  # Block 3 (256 + 256 ->  128 channels)
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 96x96-> 192x192x128
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # Block 4 (257enc +256enc -> 512 channels)
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 32x32 -> 64x64
        )
        
        self.decoder_block5 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),  # Block 5 (32 + 32 channels)
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 64x64 -> 128x128
        )
        
        # Final Output Layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder_block1(x)  # 768 spatial -> 384 soatial
        enc2 = self.encoder_block2(enc1)  # 384 -> 192 
        enc3 = self.encoder_block3(enc2)  # 192 -> 96
        enc4 = self.encoder_block4(enc3)  # 96 -> 48
        enc5 = self.encoder_block5(enc4)  # 48x48 -> 24x24
        
        # Decoder path with skip connections
        dec1 = self.decoder_block1(enc5)  # 24 x 24 x 1024->48x48x512 
        
        dec2 = self.decoder_block2(torch.cat([dec1, enc4], dim=1))  # 48x48x(512 + 512) -> 96x96x(256)
        dec3 = self.decoder_block3(torch.cat([dec2, enc3], dim=1))  # 96x96x(256 +256)-> 192 x 192 x (128)
        dec4 = self.decoder_block4(torch.cat([dec3, enc2], dim=1))  # 192x192x(128+128) -> 384x384x64
        dec5 = self.decoder_block5(torch.cat([dec4, enc1], dim=1))  # 384x384x(64 + 64) -> 768x768x64
        
        
        # Output Layer
        output = self.output_conv(dec5) #Tanh output with 1 channel. 64 channels to 1 channel
        return output

#%% Model 7 
'''
Removed the batch normalization 

'''
class ConvAutoencoder_Model7(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_Model7, self).__init__()
        
        # Encoder: 5 blocks with Conv+ PReLU + MaxPool layers
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Block 1 - 64 Channels,
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 768x768x1 -> 384x384x64
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Block 2 - 128 out channels,
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 384 x 384->192x192x128
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Block 3 - 256 out channels,
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 192 x 192 -> 96 x 96 x 256 
        )
        
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Block 4 - 512 out channels,
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 96 x 96-> 48x48x512
        )
        
        self.encoder_block5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # Block 5 - 1024 out channels),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)  # 48x48 -> 24 x 24 x 1024
        )
        
        # Decoder: 5 blocks with Conv + BatchNorm + PReLU + Upsample layers
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # Block 1 (514 out channels),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 24 x 24 into 48x48x512
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),  # Block 2 (concatenate 512enc + 512dec = 1024 out),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 48x48 -> 96 x 96 x 256
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),  # Block 3 (256 + 256 ->  128 channels),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 96x96-> 192x192x128
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),  # Block 4 (257enc +256enc -> 512 channels),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 32x32 -> 64x64
        )
        
        self.decoder_block5 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),  # Block 5 (32 + 32 channels),
            nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 64x64 -> 128x128
        )
        
        # Final Output Layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        # Downward path
        enc1 = self.encoder_block1(x)  # 768 spatial -> 384 soatial
        enc2 = self.encoder_block2(enc1)  # 384 -> 192 
        enc3 = self.encoder_block3(enc2)  # 192 -> 96
        enc4 = self.encoder_block4(enc3)  # 96 -> 48
        
        
        # Connecting Path
        enc5 = self.encoder_block5(enc4)  # 48x48 -> 24x24
        dec1 = self.decoder_block1(enc5)  # 24 x 24 x 1024->48x48x512 
        
        # Decoder path with skip connections
        dec2 = self.decoder_block2(torch.cat([dec1, enc4], dim=1))  # 48x48x(512 + 512) -> 96x96x(256)
        dec3 = self.decoder_block3(torch.cat([dec2, enc3], dim=1))  # 96x96x(256 +256)-> 192 x 192 x (128)
        dec4 = self.decoder_block4(torch.cat([dec3, enc2], dim=1))  # 192x192x(128+128) -> 384x384x64
        dec5 = self.decoder_block5(torch.cat([dec4, enc1], dim=1))  # 384x384x(64 + 64) -> 768x768x64
        
        
        # Output Layer
        output = self.output_conv(dec5) #Tanh output with 1 channel. 64 channels to 1 channel
        return output

#%% Testing Model
'''
Testing Model to try different changes. 
# Added Batch norm to the last layer 
# Revised both model7 and training model on bilinear upsampling
# Nov 23 2024 - returned back to original filter size 
# Nov 25 2024 - removing scaling layer, trying out the dropout layer
# Nov 27 2024 - removed batch norm layer

'''

class ConvAutoencoder_TestModel(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_TestModel, self).__init__()
        
        # Encoder: 5 blocks with Conv + BatchNorm + PReLU + Dropout + MaxPool
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.1),  # Dropout with 20% probability
            nn.MaxPool2d(2, 2)  # 768x768 -> 384x384
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.1),  # Dropout with 20% probability
            nn.MaxPool2d(2, 2)  # 384x384 -> 192x192
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.2),  # Dropout with 30% probability
            nn.MaxPool2d(2, 2)  # 192x192 -> 96x96
        )
        
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.2),  # Dropout with 30% probability
            nn.MaxPool2d(2, 2)  # 96x96 -> 48x48
        )
        
        self.encoder_block5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.4),  # Dropout with 50% probability
            nn.MaxPool2d(2, 2)  # 48x48 -> 24x24
        )
        
        # Decoder: 5 blocks with Conv + BatchNorm + PReLU + Dropout + Upsample
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.2),  # Dropout with 30% probability
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 24x24 -> 48x48
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.2),  # Dropout with 30% probability
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 48x48 -> 96x96
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.1),  # Dropout with 20% probability
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 96x96 -> 192x192
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.1),  # Dropout with 20% probability
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 192x192 -> 384x384
        )
        
        self.decoder_block5 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(p=0.1),  # Dropout with 20% probability
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 384x384 -> 768x768
        )
        
        # Final Output Layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Tanh()  # Output in the range [-1, 1]
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder_block1(x)  # 768 -> 384
        enc2 = self.encoder_block2(enc1)  # 384 -> 192
        enc3 = self.encoder_block3(enc2)  # 192 -> 96
        enc4 = self.encoder_block4(enc3)  # 96 -> 48
        enc5 = self.encoder_block5(enc4)  # 48 -> 24
        
        # Decoder path with skip connections
        dec1 = self.decoder_block1(enc5)  # 24 -> 48
        dec2 = self.decoder_block2(torch.cat([dec1, enc4], dim=1))  # 48 -> 96
        dec3 = self.decoder_block3(torch.cat([dec2, enc3], dim=1))  # 96 -> 192
        dec4 = self.decoder_block4(torch.cat([dec3, enc2], dim=1))  # 192 -> 384
        dec5 = self.decoder_block5(torch.cat([dec4, enc1], dim=1))  # 384 -> 768
        
        # Final Output Layer
        output = self.output_conv(dec5)  # 64 channels -> 1 channel
        return output







