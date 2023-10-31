import base64
import os
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np


class Renderer:
    color_dict = {
        "white": {"r": 1, "g": 1, "b": 1},
        "blue": {"r": 0, "g": 0, "b": 1},
        "green": {"r": 0, "g": 1, "b": 0},
        "yellow": {"r": 1, "g": 1, "b": 0},
        "red": {"r": 1, "g": 0, "b": 0},
    }

    def __init__(self, array):
        self.array = np.atleast_3d(array)
        height, width, channels = self.array.shape

        if channels == 1:
            self.is_color = False
            alpha = np.ones(self.array.shape)
            alpha[self.array == 0] = 0
            self.alpha = alpha
        else:
            self.is_color = True
            self.alpha = np.ones([height, width])

    def render(self, colormap=None):
        if self.is_color:
            return self.array

        elif colormap in self.color_dict.keys():
            r = self.array * self.color_dict[colormap]["r"]
            g = self.array * self.color_dict[colormap]["g"]
            b = self.array * self.color_dict[colormap]["b"]

        else:
            raise ValueError(f"Unsupported colormap: {colormap}")

        return np.dstack([r, g, b, self.alpha])


def get_callbacks():
    from app import callbacks

    return callbacks


def get_image_url(data: np.array, origin="upper", colormap=None):
    """
    Transform an array of data into a PNG string.
    Intended for use as encoded using base64

    Adapted from:
    https://github.com/python-visualization/folium/

    Which was inspired by
    https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image

    Parameters
    ----------
    data: numpy array or equivalent list-like object.
         Must be NxM (mono), NxMx3 (RGB) or NxMx4 (RGBA)
    origin : ['upper' | 'lower'], optional, default 'upper'
        Place the [0,0] index of the array in the upper left or lower left
        corner of the axes.
    colormap : custom colormap, one of ['red', 'green', 'blue', 'white', or 'change-detection']. Default is white.

    Returns
    -------
    PNG formatted byte string
    """
    renderer = Renderer(data)
    arr = renderer.render(colormap)
    height, width, _ = arr.shape

    # Normalize to uint8 if it isn't already.
    if arr.dtype != "uint8":
        with np.errstate(divide="ignore", invalid="ignore"):
            arr = arr * 255.0 / arr.max(axis=(0, 1)).reshape((1, 1, 4))
            arr[~np.isfinite(arr)] = 0
        arr = arr.astype("uint8")

    # Eventually flip the image.
    if origin == "lower":
        arr = arr[::-1, :, :]

    # Transform the array to bytes.
    raw_data = b"".join([b"\x00" + arr[i, :, :].tobytes() for i in range(height)])

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (
            struct.pack("!I", len(data))
            + chunk_head
            + struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head))
        )

    image = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            png_pack(b"IHDR", struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
            png_pack(b"IDAT", zlib.compress(raw_data, 9)),
            png_pack(b"IEND", b""),
        ]
    )

    b64encoded = base64.b64encode(image).decode("utf-8")
    url = "data:image/png;base64,{}".format(b64encoded)

    return url
