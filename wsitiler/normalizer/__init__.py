"""
__init__.py
Initialize the normalizer sub-module for WSItiler.

Author: Jean R Clemenceau
Date Created: 01/11/2021
"""

from PIL import Image
from pathlib import Path
import numpy as np
import pkg_resources
from wsitiler.normalizer.Normalizer import Normalizer
from wsitiler.normalizer.MacenkoNormalizer import MacenkoNormalizer

NORMALIZER_CHOICES = ["no_norm","macenko"]

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

def setup_normalizer(normalizer_choice, ref_img_path: Path=None):
    """
    Initialize a WSI normalizer object using the method of choice.

    Input:
        normalizer_choice (str): Valid choice for normalizer method. Use 'no_norm' to return a Null object.
        ref_img_path (str): Path to reference image for the normalizer.

    Output:
        An initialized normalizer object
    """

    normalizer = None

    # Import target image
    if ref_img_path is None or ref_img_path == "None":
        ref_img = get_target_img()
    else:
        ref_img_path = ref_img_path if isinstance(ref_img_path, Path) else Path(ref_img_path)
        if( ref_img_path.is_file() ):
            ref_img = np.array(Image.open(ref_img_path).convert('RGB'))
        else:
            raise ValueError("Target image does not exist")

    # Initialize normalizer & setup reference image if required
    if normalizer_choice is not None and normalizer_choice != "None" and normalizer_choice != "no_norm":
        if normalizer_choice in NORMALIZER_CHOICES:
            if normalizer_choice == "macenko":
                normalizer = MacenkoNormalizer()

            # Add more options here as "else if" blocks, like: 
            # elif normalizer_choice == "vahadane":
            #     normalizer = VahadaneNormalizer()
            
            else:
                raise ValueError("Normalizer choice not supported")

        normalizer.fit(ref_img)

    return normalizer
