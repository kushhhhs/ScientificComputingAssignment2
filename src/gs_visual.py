"""
Gray Scott Reaction-Diffusion visualisation

This module visualises the Gray Scott class fields for U and V as 
animations, single field plots with heatmaps and multiple (to compare).
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import HTML


def create_animation(gs, frames=100, updates_per_frame=20):
    """Creates an animation of U using ArtistAnimation (precomputed 
    frames). Returns the animation which can then be saved."""

    fig, ax = plt.subplots()
    im = ax.imshow(gs.U[1:-1, 1:-1], cmap="jet", interpolation="nearest", 
        vmin=0, vmax=1)

    # Store precomputed frames
    ims = []
    for _ in range(frames):
        for _ in range(updates_per_frame):
            gs.update()
        im_ = ax.imshow(gs.U[1:-1, 1:-1], cmap="jet", animated=True, vmin=0, 
            vmax=1)
        ims.append([im_])

    # Use ArtistAnimation
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

    # Close the figure to not display in the notebook
    plt.close(fig)

    return ani


def plot_field_UV(gs, iterations=10000):
    """Plots the U and V concentration field after running for 
    'iterations' steps. Returns the generated figure."""

    for _ in range(iterations):
        gs.update()

    # Set the max and min of the colorbar to accurately compare
    cmin = min(gs.U.min(), gs.V.min())
    cmax = max(gs.U.max(), gs.V.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(gs.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, 
        vmax=cmax)
    ax1.set_title("Concentration Field of U")

    im2 = ax2.imshow(gs.V[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, 
        vmax=cmax)
    ax2.set_title("Concentration Field of V")

    # Create the colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical", shrink=0.8, 
        label="Concentration (U & V)")

    return fig


def plot_field(gs, field, iterations=10000):
    """Plots the U or V concentration field after running for 
    'iterations' steps. Returns the generated figure."""
    
    for _ in range(iterations):
        gs.update()

    fig, ax = plt.subplots(figsize=(6, 6))  

    # Select field to plot
    if field == 'U':
        im = ax.imshow(gs.U[1:-1, 1:-1], cmap="jet", origin="lower")
    elif field == 'V':
        im = ax.imshow(gs.V[1:-1, 1:-1], cmap="jet", origin="lower")
    else:
        raise ValueError('No valid field parameter, choose U or V')

    # create the colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Concentration")

    ax.set_title(f"Gray-Scott Reaction-Diffusion at t={iterations} for" 
        f"{field}")

    return fig


def plot_field_compare(gs1, gs2, gs3, gs4, iterations=5000):
    """Plots the field concentrations for U for 4 given Gray Scott 
    models."""

    for _ in range(iterations):
        gs1.update()
        gs2.update()
        gs3.update()
        gs4.update()

    # Set the max and min of the colorbar to accurately compare
    cmin = min(gs1.U.min(), gs2.U.min(), gs3.U.min(), gs4.U.min())
    cmax = max(gs1.U.max(), gs2.U.max(), gs3.U.max(), gs4.U.max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    (ax1, ax2), (ax3, ax4) = axes

    im1 = ax1.imshow(gs1.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, 
        vmax=cmax)
    ax1.set_title(f"Rings", fontweight='bold', 
        fontsize=16)
    ax1.set_xticks([])
    ax1.set_yticks([])

    im2 = ax2.imshow(gs2.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, 
        vmax=cmax)
    ax2.set_title(f"Splitting cells (mitosis)", 
        fontweight='bold', fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])

    im3 = ax3.imshow(gs3.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, 
        vmax=cmax)
    ax3.set_title(f"Maze", fontweight='bold', 
        fontsize=16)
    ax3.set_xticks([])
    ax3.set_yticks([])

    im4 = ax4.imshow(gs4.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, 
        vmax=cmax)
    ax4.set_title(f"Belousov-Zhabotinsky-like",
        fontweight='bold', fontsize=16)
    ax4.set_xticks([])
    ax4.set_yticks([])

    cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3, ax4], orientation="vertical", 
        shrink=0.8)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Concentration (U)", fontsize=18)

    return fig