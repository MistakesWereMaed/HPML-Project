import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_currents(u, v, lat, lon, arrow_step=100, arrow_scale=0.1):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    lon, lat = np.meshgrid(lon, lat)
    intensity = np.sqrt(u**2 + v**2)
    im = ax.pcolormesh(lon, lat, intensity, cmap='Blues')

    u_scaled = u * arrow_scale
    v_scaled = v * arrow_scale

    skip = (slice(None, None, arrow_step), slice(None, None, arrow_step))

    ax.quiver(lon[skip], lat[skip], u_scaled[skip], v_scaled[skip], color='black', scale=2, width=0.003, headwidth=4)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='dotted')

    plt.title('Intensity and Direction of Currents')
    plt.show()

def plot_height(zos, lat, lon):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    lon, lat = np.meshgrid(lon, lat)
    im = ax.pcolormesh(lon, lat, zos, cmap='Blues')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='dotted')

    plt.title('Sea Level Height')
    plt.show()

def plot_training_time(csv_path):
    df = pd.read_csv(csv_path)
    
    if 'gpu_count' not in df.columns or 'train_time' not in df.columns:
        raise ValueError("CSV file must contain 'gpu_count' and 'train_time' columns.")
    
    gpu_counts = df['gpu_count']
    training_times = df['train_time']
    
    plt.figure(figsize=(8, 6))
    plt.bar(gpu_counts, training_times, color='skyblue')
    plt.xlabel("GPU Count")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time by GPU Count")
    plt.xticks(gpu_counts)
    plt.show()

def plot_loss_history(csv_path):
    df = pd.read_csv(csv_path)
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Train Loss
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss History')
    ax1.grid()
    
    # Plot Validation Loss
    ax2.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss History')
    ax2.grid()

    # Display the plots
    plt.tight_layout()
    plt.show()

def plot_accuracy_over_time(npz_path):
    data = np.load(npz_path)
    lead_times = data['lead_times']
    loss = data['loss']
    plt.figure(figsize=(8, 6))
    plt.plot(lead_times, loss, marker='o', linestyle='-')
    plt.xlabel('Prediction Lead Time (Days)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Prediction Lead Time')
    plt.grid()
    plt.show()