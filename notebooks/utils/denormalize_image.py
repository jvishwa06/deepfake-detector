""" Denormalise image """

import torch
import numpy as np

def denormalize_image(image:torch.Tensor) -> np.array:
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = image * 255.0
    image = image.astype(np.uint8)
    return image
