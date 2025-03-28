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

def plot_loss_and_training_time(csv_path):
    df = pd.read_csv(csv_path)

    if not {'gpu_count', 'train_time', 'avg_val_loss'}.issubset(df.columns):
        raise ValueError("CSV file must contain 'gpu_count', 'train_time', and 'avg_val_loss' columns.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Training Time by GPU Count
    gpu_counts = df['gpu_count']
    training_times = df['train_time']

    ax1.bar(gpu_counts, training_times, color='skyblue')
    ax1.set_xlabel("GPU Count")
    ax1.set_ylabel("Training Time (seconds)")
    ax1.set_title("Training Time by GPU Count")
    ax1.set_xticks(gpu_counts)

    # Plot Validation Loss by GPU Count
    val_losses = df['avg_val_loss']
    
    ax2.bar(gpu_counts, val_losses, color='orange')
    ax2.set_xlabel("GPU Count")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss by GPU Count")
    ax2.set_xticks(gpu_counts)

    plt.tight_layout()
    plt.show()

def plot_loss_history(csv_path):
    df = pd.read_csv(csv_path)
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Train Loss
    ax1.plot(df['epoch'] + 1, df['train_loss'], label='Train Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss History')
    ax1.grid()
    
    # Plot Validation Loss
    ax2.plot(df['epoch'] + 1, df['val_loss'], label='Validation Loss', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss History')
    ax2.grid()

    # Display the plots
    plt.tight_layout()
    plt.show()

def plot_accuracy_over_time(npz_path):
    # Load NPZ file
    data = np.load(npz_path)
    
    # Extract loss
    loss = data["loss"]
    
    # Generate x-axis as days (assuming each entry corresponds to a day)
    days = np.arange(1, len(loss) + 1)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(days, loss, marker='o', linestyle='-')
    plt.xlabel("Prediction Lead Time (Days)")
    plt.ylabel("Loss")
    plt.title("Model Loss Over Prediction Lead Time")
    plt.grid(True)
    plt.show()