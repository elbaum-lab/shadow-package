# Copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3
# See the LICENSE-GPL-3 for details

import numpy as np

try:
    from png import Writer
    HAS_PNG = True
except ImportError:
    HAS_PNG = False

def save_png(array, filename):
    if not HAS_PNG:
        raise ImportWarning("pypng library is needed to save PNG files")
    
    # Histogram equalization via rank ordering
    flat = array.flatten()
    sorted_indices = np.argsort(flat)

    # Create output array where each pixel gets its rank as intensity
    equalized = np.empty_like(flat, dtype=np.float64)
    equalized[sorted_indices] = np.arange(len(flat), dtype=np.float64)

    # Scale to full 16-bit range
    equalized = (equalized / len(flat) * 65535).astype(np.uint16)
    array = equalized.reshape(array.shape)
    
    # Write PNG
    with open(filename, 'wb') as f:
        writer = Writer(width=array.shape[1], height=array.shape[0], greyscale=True, bitdepth=16, compression=9, alpha=False)
        writer.write(f, array)