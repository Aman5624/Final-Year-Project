"""
Script to create placeholder deep learning models for Final project
These models have the correct input/output shapes but are not trained.
You should replace them with properly trained models later.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, 
    Activation, BatchNormalization, Dropout
)
import numpy as np

# Create models directory if it doesn't exist
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"Created {MODELS_DIR} directory")

# ============================================================================
# 1. BUILDING DETECTION MODEL (RIO_building_detect_model1.keras)
# Input: (256, 256, 1) - Grayscale
# Output: Binary mask for buildings
# ============================================================================
def create_building_detection_model():
    """
    Creates a building detection model.
    Input: Grayscale image (256, 256, 1)
    Output: Binary segmentation mask (256, 256, 1)
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        UpSampling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        UpSampling2D((2, 2)),
        
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        UpSampling2D((2, 2)),
        
        Conv2D(1, (1, 1), activation='sigmoid', padding='same'),
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================================================================
# 2. WATER BODY DETECTION MODEL (water_model_heavy_rgb.h5)
# Input: (256, 256, 3) - RGB
# Output: Binary mask for water bodies
# ============================================================================
def create_water_detection_model():
    """
    Creates a water body detection model.
    Input: RGB image (256, 256, 3)
    Output: Binary segmentation mask (256, 256, 1)
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        Conv2D(1, (1, 1), activation='sigmoid', padding='same'),
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================================================================
# 3. CHANGE DETECTION MODEL (unet_building_change_detection1.h5)
# Input: (512, 512, 3) - RGB (two images or difference image)
# Output: Change detection mask (512, 512, 1)
# ============================================================================
def create_change_detection_unet():
    """
    Creates a U-Net model for change detection.
    Input: RGB image (512, 512, 3)
    Output: Binary change detection mask (512, 512, 1)
    """
    inputs = Input((512, 512, 3))
    
    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4], axis=3)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3], axis=3)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2], axis=3)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# ============================================================================
# Create and save all models
# ============================================================================
def main():
    print("Creating placeholder models for Nirmaan Netra...")
    print("=" * 60)
    
    # 1. Building Detection Model
    print("\n1. Creating Building Detection Model (RIO_building_detect_model1.keras)...")
    building_model = create_building_detection_model()
    building_model_path = os.path.join(MODELS_DIR, 'RIO_building_detect_model1.keras')
    building_model.save(building_model_path)
    print(f"   ✓ Saved to {building_model_path}")
    print(f"   Input shape: (256, 256, 1) | Output shape: (256, 256, 1)")
    
    # 2. Water Body Detection Model
    print("\n2. Creating Water Body Detection Model (water_model_heavy_rgb.h5)...")
    water_model = create_water_detection_model()
    water_model_path = os.path.join(MODELS_DIR, 'water_model_heavy_rgb.h5')
    water_model.save(water_model_path)
    print(f"   ✓ Saved to {water_model_path}")
    print(f"   Input shape: (256, 256, 3) | Output shape: (256, 256, 1)")
    
    # 3. Change Detection Model
    print("\n3. Creating Change Detection U-Net Model (unet_building_change_detection1.h5)...")
    change_model = create_change_detection_unet()
    change_model_path = os.path.join(MODELS_DIR, 'unet_building_change_detection1.h5')
    change_model.save(change_model_path)
    print(f"   ✓ Saved to {change_model_path}")
    print(f"   Input shape: (512, 512, 3) | Output shape: (512, 512, 1)")
    
    print("\n" + "=" * 60)
    print("✓ All models created successfully!")
    print("\n⚠️  IMPORTANT NOTES:")
    print("  • These are UNTRAINED placeholder models for testing only")
    print("  • They have the correct input/output shapes for the app")
    print("  • Replace with properly trained models for production use")
    print("  • Train on your own labeled dataset for real results")
    print("=" * 60)

if __name__ == '__main__':
    main()
