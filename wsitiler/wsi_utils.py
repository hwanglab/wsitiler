"""
wsi_utils.py
Implements functions used throughout WSItiler package.

Author: Jean R Clemenceau
Date Created: 18/09/2022
"""

import openslide
import math
import pandas as pd
from pathlib import Path


def describe_wsi_levels(wsi_object: openslide.OpenSlide):
    '''
    Obtain a the parameters for each level of an OpenSlide image: 
    Level number, magnification, resolution & image dimensions.

    Input:
        wsi_object (OpenSlide): OpenSlide object to be described or path to WSI file.

    Output:
        A Pandas Dataframe describing parameters of object's pyramidal image
    '''
    #Validate input WSI
    if isinstance(wsi_object, Path):
        wsi_object = openslide.open_slide(str(wsi_object))
    elif isinstance(wsi_object, str):
        wsi_object = openslide.open_slide(wsi_object)
    elif isinstance(wsi_object, openslide.OpenSlide ):
        raise ValueError("Input is not valid OpenSlide instance")

    #Get baseline parameters for max magnification
    level_cnt = wsi_object.level_count
    max_objective = float(wsi_object.properties['openslide.objective-power'])
    max_mpp = float(wsi_object.properties['openslide.mpp-x'])

    #Calculate vales across pyramidal file
    levels = range(0,level_cnt)
    downsamples = [wsi_object.properties['openslide.level[%d].downsample' % i] for i in levels]
    factors = [math.log2( round(float(x)) ) for x in downsamples]
    factors[0]=1
    magnifications = ["%sx" % round(max_objective/x) for x in factors]
    resolutions = ["%smpp" % (max_mpp*x) for x in factors]

    #Extract image sizes across pyramidal file
    img_widths = [int(wsi_object.properties['openslide.level[%d].width' % i]) for i in levels]
    img_heights = [int(wsi_object.properties['openslide.level[%d].height' % i]) for i in levels]

    #Create dataframe
    img_desc = pd.DataFrame({'Level':levels,'Magnification':magnifications,'Resolution':resolutions,'Width':img_widths, "Height":img_heights})

    return(img_desc)

def find_wsi_level(wsi_object: openslide.OpenSlide,level_query: str="0"):
    '''
    Determine the desired level of an OpenSlide object according to a query string.

    Input:
        wsi_object (OpenSlide): OpenSlide object to be queried or path to WSI file.
        level_query (str): Level query string: Magnifications(AAx), Resolutions(AAmpp), Level(AA).

    Output:
        An integer indicating desired image level
    '''
    #Validate input WSI
    if isinstance(wsi_object, Path):
        wsi_object = openslide.open_slide(str(wsi_object))
    elif isinstance(wsi_object, str):
        wsi_object = openslide.open_slide(wsi_object)
    elif isinstance(wsi_object, openslide.OpenSlide ):
        raise ValueError("Input is not valid OpenSlide instance")

    # Determine wuery field based on query value
    if level_query.endswith('x'):
        field = "Magnification"
    elif level_query.endswith('mpp'):
        field = "Resolution"
    elif float(level_query).is_integer():
        field = "Level"
        level_query = int(level_query)
    else:
        raise ValueError("Level selection value format is not supported. Must be an integer or end with 'x' or 'mpp'")

    # Get description of object for query
    desc = describe_wsi_levels(wsi_object)

    try:
        theLevel = desc['Level'][desc[field]==level_query].values[0]
    except IndexError as e:
        raise ValueError("Level query value not found.")
    
    return(theLevel)