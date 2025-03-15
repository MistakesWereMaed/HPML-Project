import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

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