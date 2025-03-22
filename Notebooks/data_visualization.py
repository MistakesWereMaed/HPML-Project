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
    plt.figure(figsize=(8, 6))
    plt.plot(df['gpu_count'], df['train_time'], marker='o', linestyle='-')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs. GPU Count')
    plt.grid()
    plt.show()

def plot_loss_history(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid()
    plt.show()

def plot_accuracy_over_lead_time(npz_path):
    data = np.load(npz_path)
    lead_times = data['lead_times']
    accuracies = data['accuracies']
    plt.figure(figsize=(8, 6))
    plt.plot(lead_times, accuracies, marker='o', linestyle='-')
    plt.xlabel('Prediction Lead Time (Days)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Prediction Lead Time')
    plt.grid()
    plt.show()