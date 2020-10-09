import numpy as np
# from fps_v0 import FPS      # Simple loop
from lib.fps.fps_v1 import FPS      # Utilise broadcasting


def farthest_sample(data, samples):
    fps = FPS(data, samples)
    fps.fit()
    sample = fps.get_selected_pts()
    return sample
