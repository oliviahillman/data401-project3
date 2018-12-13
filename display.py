from math import pi
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib import animation

BLACK = 1
WHITE = 2

BLACK_COLOR_KEY = -10
WHITE_COLOR_KEY = 10

def flat_hex_points(center, size=1):
    i = np.arange(6)
    angle_deg = 60 * i - 30
    angle_rad = pi / 180 * angle_deg

    x = center[0] + size * np.cos(angle_rad)
    y = center[1] + size * np.sin(angle_rad)

    return np.dstack([ x, y ])

def generate_hex_patches(board_dim, size):
    from matplotlib.patches import Polygon

    grid = np.mgrid[0:board_dim, 0:board_dim]
    shift = np.rot90(np.tile(np.arange(board_dim) * size, (board_dim, 1)), k=3)
    xs = grid[1] - (shift * 0.85)
    ys = grid[0]

    centers = np.dstack([xs, ys])
    hexagons = np.apply_along_axis(flat_hex_points, -1, centers, size=size)
    
    hexagons = hexagons.reshape(-1, *hexagons.shape[-2:])
    
    patches = []
    for i in range(board_dim * board_dim):
        patches.append(Polygon(hexagons[i], closed=True, edgecolor='black'))
    
    return patches


def animate_board_choices(filename, stats, player, relative=False):
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 11)
    
    viridis = plt.cm.get_cmap('coolwarm', 256)
    viridis.set_over('white')
    viridis.set_under('black')

    patches = generate_hex_patches(13, 0.6)
    p = PatchCollection(patches, alpha=0.8, cmap=viridis, edgecolor='black')
    p.set_array(np.zeros((13, 13)).flatten())
    ax.add_collection(p)
    cb = fig.colorbar(p)
    
    p.set_clim(0, 1)
    ax.set_xlim(-6.8, 12.7)
    ax.set_ylim(-0.7, 12.7)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    np.set_printoptions(precision=1)

    def update(frame):
        preds = stats['preds'][frame]
        base = stats['base'][frame]
        
        if relative:
            if player == BLACK:
                black_board = BLACK_COLOR_KEY * base[:, :, 0]
                white_board = WHITE_COLOR_KEY * base[:, :, 1]
            else:
                black_board = BLACK_COLOR_KEY * base[:, :, 1]
                white_board = WHITE_COLOR_KEY * base[:, :, 0]
        else:
            black_board = BLACK_COLOR_KEY * base[:, :, BLACK - 1]
            white_board = WHITE_COLOR_KEY * base[:, :, WHITE - 1]
            
        if relative:
            preds_board = preds[:, :, 0]
        else:
            preds_board = preds[:, :, player - 1]

        complete_board = preds_board + black_board + white_board
        adjusted_board = np.flip(complete_board, axis=0).flatten()

        p.set_array(adjusted_board)
        return p,

    ani = FuncAnimation(fig, update, frames=stats['count'], blit=True)

    ani.save(filename, writer=writer)
    
    plt.close(fig)