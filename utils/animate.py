"""Animation helper utilities for capturing Matplotlib FigureCanvas frames and saving GIFs.
"""
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import imageio


def canvas_to_rgb(canvas: FigureCanvas) -> np.ndarray:
    """Capture the current canvas buffer and return an HxWx3 RGB uint8 array.

    This handles Agg backends and returns a copy safe for storage.
    """
    buf = canvas.tostring_argb()
    w, h = canvas.get_width_height()
    arr = np.frombuffer(buf, dtype='uint8').reshape((h, w, 4))
    return arr[:, :, 1:4].copy()


def save_gif_from_images(images, outfile: str, duration: float = 0.05) -> None:
    """Save a sequence of HxWx3 uint8 images as a GIF, creating parent dirs.
    """
    if not images:
        return
    parent = os.path.dirname(outfile) or '.'
    os.makedirs(parent, exist_ok=True)
    imageio.mimsave(outfile, images, duration=duration, loop=0)
