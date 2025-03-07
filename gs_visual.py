import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_animation(gs, frames=100):
    """Creates an animation of U using FuncAnimation"""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Gray-Scott Model: U Concentration")

    im = ax.imshow(gs.U[1:-1, 1:-1], cmap='jet', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    def update_frame(frame):
        """Updates U and the image data"""
        for _ in range(10):
            gs.update()
        im.set_array(gs.U[1:-1, 1:-1])

        return [im]

    ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=20, blit=False, repeat_delay=1000)
    ani.save('gray_scott_func2.gif', writer='pillow', fps=50)  

    # Close the figure to not automatically display in the notebook
    plt.close(fig)

    return ani


def plot_field_UV(gs, iterations=10000):
    for _ in range(iterations):
        gs.update()

    cmin = min(gs.U.min(), gs.V.min())
    cmax = max(gs.U.max(), gs.V.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(gs.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, vmax=cmax)
    ax1.set_title("Concentration Field of U")

    im2 = ax2.imshow(gs.V[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, vmax=cmax)
    ax2.set_title("Concentration Field of V")

    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical", shrink=0.8, label="Concentration (U & V)")
    plt.savefig('gray_scott_field_UV.png', dpi=300)
    plt.show()


def plot_field(gs, field, iterations=10000):
    for _ in range(iterations):
        gs.update()

    plt.figure(figsize=(6, 6))

    if field == 'U':
        plt.imshow(gs.U[1:-1, 1:-1], cmap="jet", origin="lower")
    elif field == 'V':
        plt.imshow(gs.V[1:-1, 1:-1], cmap="jet", origin="lower")
    else:
        raise ValueError('No valid field parameter, choose U or V')
        
    plt.colorbar(label="Concentration", shrink=0.8)
    plt.title(f"Gray-Scott Reaction-Diffusion at t={iterations} for {field}")
    plt.savefig(f"gray_scott_field_{field}.png", dpi=300)
    plt.show()
