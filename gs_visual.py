import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_animation(gs, frames=100, fps=20):
    """Creates an animation of U using FuncAnimation"""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Gray-Scott Model: U Concentration")

    im = ax.imshow(gs.U[1:-1, 1:-1], cmap='jet', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    def update_frame(frame):
        """Updates U and the image data"""
        gs.update()
        im.set_array(gs.U[1:-1, 1:-1])

        return [im]

    ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=(1000/fps), blit=True, repeat_delay=False)
    ani.save('gray_scott_func2.gif', writer='pillow', fps=fps)  

    # Close the figure to not automatically display in the notebook
    #plt.close(fig)

    return ani


def plot_field_UV(gs, iterations=10000):
    for _ in range(gs, iterations):
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


def plot_field_compare(gs1, gs2, gs3, gs4, iterations=10000):

    for _ in range(iterations):
        gs1.update()
        gs2.update()
        gs3.update()
        gs4.update()

    cmin = min(gs1.U.min(), gs2.U.min(), gs3.U.min(), gs4.U.min())
    cmax = max(gs1.U.max(), gs2.U.max(), gs3.U.max(), gs4.U.max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    (ax1, ax2), (ax3, ax4) = axes


    im1 = ax1.imshow(gs1.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, vmax=cmax)
    ax1.set_title(f"f = {gs1.feed} and k = {gs1.kill}")

    im2 = ax2.imshow(gs2.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, vmax=cmax)
    ax2.set_title(f"f = {gs2.feed} and k = {gs2.kill}")

    im3 = ax3.imshow(gs3.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, vmax=cmax)
    ax3.set_title(f"f = {gs3.feed} and k = {gs3.kill}")

    im4 = ax4.imshow(gs4.U[1:-1, 1:-1], cmap="jet", origin="lower", vmin=cmin, vmax=cmax)
    ax4.set_title(f"f = {gs4.feed} and k = {gs4.kill}")

    cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3, ax4], orientation="vertical", shrink=0.8, label="Concentration (U & V)")
    plt.savefig('gray_scott_field_UV.png', dpi=300)
    plt.show()