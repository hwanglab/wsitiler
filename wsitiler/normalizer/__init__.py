"""
__init__.py
Initialize the normalizer sub-module for WSItiler.

Author: Jean R Clemenceau
Date Created: 01/11/2021
"""

from PIL import Image
import numpy as np
import pkg_resources

def get_target_img():
    """
    Open target image for tile normalization as numpy array.
    
    Input:
        None

    Returns:
        the default target image as an RGB formatted numpy array.
    """
    img_path=pkg_resources.resource_filename("wsitiler.normalizer", "macenko_reference_img.png")
    target_img = Image.open(img_path).convert('RGB')
    target_img = np.array(target_img)
    return target_img