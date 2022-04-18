"""
tiler.py
Implement functions for dividing a whole slide image into tiles and save them as individual files.

Author: Jean R Clemenceau
Date Created: 11/11/2021
"""

import openslide
import argparse
import os
import math
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import traceback

from PIL import Image
from pathlib import Path
from math import ceil, floor
from skimage import transform
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, opening,closing, square

from openslide import OpenSlideError, OpenSlideUnsupportedFormatError
from PIL import UnidentifiedImageError
from PIL import Image

import wsitiler.normalizer as norm

# MICRONS_PER_TILE defines the tile edge length used when breaking WSIs into smaller images (m x m)
MICRONS_PER_TILE = 512
# NOISE_SIZE_MICRONS defines maximum size (in microns) for an artifact to be considered as noise in the WSI tissue mask.
NOISE_SIZE_MICRONS = 256
# FINAL_TILE_SIZE defines final pixel width and height of a processed tile. #TODO: remove
# FINAL_TILE_SIZE = 224
# MIN_FOREGROUND_THRESHOLD defines minimum tissue/background ratio to classify a tile as foreground.
MIN_FOREGROUND_THRESHOLD = 0.01
# NORMALIZER_CHOICES defines the valid choices for WSI normalization methods.
NORMALIZER_CHOICES= ["None","macenko"]
# SUPPORTED_WSI_FORMATS defines the WSI formats supported by Openslide.
SUPPORTED_WSI_FORMATS = [".svs",".ndpi",".vms",".vmu",".scn",".mrxs",".tiff",".svslide",".tif",".bif"]

# Log Levels
LOG_ERROR = 0
LOG_WARNING = 1
LOG_INFO = 2
LOG_DEBUG = 3

SET_LOG = LOG_ERROR

def PRINT_LOG(loglevel, log):
    '''
    Setup funciton for logging.
    '''
    if(SET_LOG >= loglevel) :
        print(log)

def describe_wsi_levels(wsi_object):
    '''
    Obtain a the parameters for each level of an OpenSlide image: 
    Level number, magnification, resolution & image dimensions.

    Input:
        wsi_object (OpenSlide): OpenSlide object to be described.

    Output:
        A Pandas Dataframe describing parameters of object's pyramidal image
    '''
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

def find_wsi_level(wsi_object,level_query="0"):
    '''
    Determine the desired level of an OpenSlide object according to a query string.

    Input:
        wsi_object (OpenSlide): OpenSlide object to be queried.
        level_query (str): Level query string: Magnifications(AAx), Resolutions(AAmpp), Level(AA).

    Output:
        An integer indicating desired image level
    '''
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

def setup_normalizer(normalizer_choice, ref_img_path=None):
    """
    Initialize a WSI normalizer object using the method of choice.

    Input:
        normalizer_choice (str): Valid choice for normalizer method. Use 'None' to return a Null object.
        ref_img_path (str): Path to reference image for the normalizer.

    Output:
        An initialized normalizer object
    """

    normalizer = None

    # Import target image
    if ref_img_path is None or ref_img_path == "None":
        ref_img = norm.get_target_img()
    else:
        if os.path.exists(ref_img_path):
            ref_img = np.array(Image.open(ref_img_path).convert('RGB'))
        else:
            raise ValueError("Target image does not exist")

    # Initialize normalizer & setup reference image if required
    if normalizer_choice is not None and normalizer_choice != "None":
        if normalizer_choice in NORMALIZER_CHOICES:
            if normalizer_choice == "macenko":
                normalizer = norm.MacenkoNormalizer.MacenkoNormalizer()

            # Add more options here as "else if" blocks, like: 
            # elif normalizer_choice == "vahadane":
            #     normalizer = norm.VahadaneNormalizer.VahadaneNormalizer()
            
            else:
                raise ValueError("Normalizer choice not supported")

        normalizer.fit(ref_img)

    return normalizer

def prepare_tiles(wsi, output, mpt=MICRONS_PER_TILE, wsi_level=0, get_chunk_id=False):
    """
    Import a WSI, calculate foreground/background, and calculate tile coordinates to output directory.

    Input:
        wsi (OpenSlide): OpenSlide object containing WSI to be processed.
        output (str): Path to output directory for processed tiles.
        mpt (str): Desire width and height of processed tiles in pixels or microns. Default: [%d].
        wsi_level (int): Image level to be tiled. Default: 0.
        get_chunk_id (bool): Wether or not to identify individual tissue chunks in slide (larger than a tile). Default: False

    Output:
        Funtion exports 3 files:
            1. Reference CSV file containing coordinates and tissue ratio for each tile.
            2. RGB thumbnail of the WSI as a PNG file.
            3. Binary tissue mask of the WSI as a PNG file.
        Function returns a tuple with with the following format:
            1. Pandas dataframe containing coordinates and tissue ratio for each tile.
            2. Pixels-per-tile value for the WSI's X axis
            3. Pixels-per-tile value for the WSI's Y axis
    """ % (MICRONS_PER_TILE)

    # Valiate output path
    if not os.path.isdir(output):
        os.mkdir(output)
    wsi_img_id = os.path.basename(output)

    # Get WSI details and find level
    wsi_params = describe_wsi_levels(wsi)

    wsi_width = wsi_params["Width"].iloc[wsi_level]
    wsi_height = wsi_params["Height"].iloc[wsi_level]

    # Calculate microns per pixel at desired level
    resolution = wsi_params["Resolution"].iloc[wsi_level]
    mpp = float(resolution.split("mpp")[0])

    # Calculate desired tile dimensions (pixels per tile)
    if mpt.endswith("um"):
        microns = float(mpt.split("um")[0])
        pixels = round(microns/mpp)
        # ppt_x = int(mpt / float(wsi.properties['openslide.mpp-x']))
        # ppt_y = int(mpt / float(wsi.properties['openslide.mpp-y']))
    else:
        try:
            pixels = int(mpt)
        except ValueError as e:
            raise ValueError("Tile length format is not valid. Provide integer value or microns using 'XXum'.")

    ppt_x = ppt_y = pixels

    # Get thumbnail for tissue mask
    thumbnail_og = wsi.get_thumbnail(size=(wsi.level_dimensions[-1][0], wsi.level_dimensions[-1][1]))
    thumbnail = np.array(thumbnail_og)
    thumbnail = (rgb2gray(thumbnail) * 255).astype(np.uint8)

    # calculate mask parameters
    # thumbnail_ratio = wsi.dimensions[0] / thumbnail.shape[1]  # wsi dimensions: (x,y); thumbnail dimensions: (rows,cols)
    # thumbnail_mpp = float(wsi.properties['openslide.mpp-x']) * thumbnail_ratio
    thumbnail_ratio = wsi_width / thumbnail.shape[1]  # wsi dimensions: (x,y); thumbnail dimensions: (rows,cols)
    thumbnail_mpp = mpp * thumbnail_ratio
    noise_size_pix = round(NOISE_SIZE_MICRONS / thumbnail_mpp)
    # noise_size = round(noise_size_pix / thumbnail_ratio)
    thumbnail_ppt_x = ceil(ppt_x / thumbnail_ratio)
    thumbnail_ppt_y = ceil(ppt_y / thumbnail_ratio)
    tile_area = thumbnail_ppt_x*thumbnail_ppt_y

    # Create and clean tissue mask
    tissue_mask = (thumbnail[:, :] < threshold_otsu(thumbnail))
    tissue_mask = remove_small_objects(tissue_mask, noise_size_pix)
    tissue_mask = closing(tissue_mask, square(5))
    tissue_mask = opening(tissue_mask, square(5))

    # # Remove holes in tissue smaller than a tile
    # tissue_mask = np.invert(tissue_mask)
    # tissue_mask = remove_small_objects(tissue_mask, tile_area)
    # tissue_mask = np.invert(tissue_mask)

    if get_chunk_id:
        # Get labels for all chunks
        chunk_mask = ndi.label(tissue_mask)[0]

        # Filter out chunks smaller than tile size
        (chunk_label, chunk_size) = np.unique(chunk_mask,return_counts=True)
        filtered_chunks = chunk_label[ chunk_size < tile_area ]
        for l in filtered_chunks:
            chunk_mask[chunk_mask == l] = 0

    # Calculate margin according to ppt sizes
    wsi_x_tile_excess = wsi_width % ppt_x
    wsi_y_tile_excess = wsi_height % ppt_y

    # Determine WSI tile coordinates
    wsi_tiles_x = list(range(ceil(wsi_x_tile_excess / 2), wsi_width - floor(wsi_x_tile_excess / 2), ppt_x))
    wsi_tiles_y = list(range(ceil(wsi_y_tile_excess / 2), wsi_height - floor(wsi_y_tile_excess / 2), ppt_y))

    # Approximate mask tile coordinates
    mask_tiles_x = [floor(i / thumbnail_ratio) for i in wsi_tiles_x]
    mask_tiles_y = [floor(i / thumbnail_ratio) for i in wsi_tiles_y]

    # Populate tile reference table
    rowlist = []
    for x in range(len(wsi_tiles_x)):
        for y in range(len(wsi_tiles_y)):
            # Get np.array subset of image (a tile)
            aTile = tissue_mask[mask_tiles_y[y]:mask_tiles_y[y] + thumbnail_ppt_y,
                    mask_tiles_x[x]:mask_tiles_x[x] + thumbnail_ppt_x]
            
            # Determine chunk id by most prevalent ID
            if get_chunk_id:
                chunk_tile = chunk_mask[mask_tiles_y[y]:mask_tiles_y[y] + thumbnail_ppt_y,
                    mask_tiles_x[x]:mask_tiles_x[x] + thumbnail_ppt_x]
                chunk_id = np.bincount(chunk_tile.flatten()).argmax()
                

            # Calculate tissue ratio for tile
            tissue_ratio = np.sum(aTile) / aTile.size

            slide_id = len(rowlist) + 1

            new_row = {"image_id": wsi_img_id,
                       "tile_id": slide_id,
                       "index_x": x,
                       "index_y": y,
                       "wsi_x": wsi_tiles_x[x],
                       "wsi_y": wsi_tiles_y[y],
                       "mask_x": mask_tiles_x[x],
                       "mask_y": mask_tiles_y[y],
                       "filename": wsi_img_id + "__tile-n-%d_x-%d_y-%d.png" % (slide_id, x, y),
                       "tissue_ratio": tissue_ratio
                       }

            if get_chunk_id:
                new_row['chunk_id'] = chunk_id

            rowlist.append(new_row)

    # Create reference dataframe
    colnames = ["image_id","tile_id", "index_x", "index_y", "wsi_x", "wsi_y", "mask_x", "mask_y", "filename", "tissue_ratio"]
    if get_chunk_id:
                colnames.append('chunk_id')
    ref_df = pd.DataFrame(data=rowlist, columns=colnames)

    # Remove filenames for empty tiles
    ref_df.loc[ref_df['tissue_ratio'] == 0, "filename"] = None

    output = Path(output) 

    # Export Mask image
    tissue_mask_trimmed=tissue_mask[mask_tiles_y[0]:mask_tiles_y[-1] + thumbnail_ppt_y,
                    mask_tiles_x[0]:mask_tiles_x[-1] + thumbnail_ppt_x]
    filename_tissuemask = os.path.basename(output) + "___tissue-mask_tilesize_x-%d-y-%d.png" % (
    thumbnail_ppt_x, thumbnail_ppt_y)
    plt.figure()
    plt.imshow(tissue_mask_trimmed, cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig(output / filename_tissuemask, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Export Chunk Mask image
    if get_chunk_id:
        chunk_mask_trimmed=chunk_mask[mask_tiles_y[0]:mask_tiles_y[-1] + thumbnail_ppt_y,
                    mask_tiles_x[0]:mask_tiles_x[-1] + thumbnail_ppt_x]
        filename_chunkmask = os.path.basename(output) + "___chunk-mask_tilesize_x-%d-y-%d.png" % (
        thumbnail_ppt_x, thumbnail_ppt_y)
        plt.figure()
        plt.imshow(chunk_mask_trimmed, cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.margins(0, 0)
        plt.savefig(output / filename_chunkmask, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Export Thumbnail image
    thumbnail_trimmed=np.array(thumbnail_og)[mask_tiles_y[0]:mask_tiles_y[-1] + thumbnail_ppt_y,
                    mask_tiles_x[0]:mask_tiles_x[-1] + thumbnail_ppt_x]
    filename_thumbnail = os.path.basename(output) + "___thumbnail_tilesize_x-%d-y-%d.png" % (
        thumbnail_ppt_x, thumbnail_ppt_y)
    plt.figure()
    plt.imshow(thumbnail_trimmed)
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig(output / filename_thumbnail, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Export CSV file
    filename_refdf = os.path.basename(output) + "___reference_wsi-tilesize_x-%d-y-%d_mask-tilesize_x-%d-y-%d_img-level_%d.tsv" % \
                     (ppt_x, ppt_y,thumbnail_ppt_x, thumbnail_ppt_y, wsi_level)
    ref_df.to_csv(output / filename_refdf, sep="\t", line_terminator="\n", index=False)

    return (ref_df, ppt_x, ppt_y)

def export_tiles(wsi, tile_data, tile_dims, output="./", normalizer=None, wsi_level=0):
    """
    Import a WSI, split in to tiles, normalize color if requested, and save individual tile files to output directory.

    Input:
        wsi (str): Path to WSI file to be processed.
        tile_data (Pandas dataframe): Details for tiles to be extracted, normalized and exported.
        tile_dims (dict): Python dictionary containing size of tiles in WSI. Fromat: {'x':<x-tiles>,'y':<y-tiles>}
        output (str): Path to output directory for processed tiles. Default: [./]
        normalizer (normalizer): Normalizer object that has been initialized and is ready to use. Default: [None].
        final_tile_size (int): Desire pixel width and height of final tiles. Use zero (0) for NO resizing. Default: [0].

    Output:
        Funtion exports tiles as PNG files to output directory.
    """
    # Open and prepare input
    wsi_image = openslide.open_slide(wsi)
    output = Path(output)

    # Process and export each tile sequentially
    for index, aTile in tile_data.iterrows():
        # Extract tile region
        aTile_img = wsi_image.read_region((aTile["wsi_x"], aTile["wsi_y"]), level=wsi_level,
                                size=(tile_dims['x'], tile_dims['y']))
        ##TODO: check if this causes IO on every call & if it'd be better to read 1 region per list & extract tiles using numpy

        #Convert to RGB array
        aTile_img = np.array( aTile_img.convert('RGB') )

        # Normalize if required
        if normalizer is not None:
            aTile_img = normalizer.transform(aTile_img)

        # Save tile image to file
        plt.imsave(output / aTile['filename'], aTile_img)

    wsi_image.close()
    return


if __name__ == '__main__':
    # Define command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default="./", help='Input directory or WSI image. Default: [./]')
    ap.add_argument('-o', '--output', default="./", help='Output directory Default: [./]')
    ap.add_argument('-c', '--cores', default=mp.cpu_count(), type=int, help='Numbers of processes to be spun up in parallel to process each WSI. Default: [%d]' % mp.cpu_count() )
    ap.add_argument('-d', '--tile_dimensions', default=MICRONS_PER_TILE, type=str, help="Defines the tile edge length used when breaking WSIs into smaller images. Provide integer value for pixel size, for micron size use format: XXum. Default: [%d]" % MICRONS_PER_TILE)
    ap.add_argument('-l', '--image_level', default="0", type=str, help="Defines the level of magnification to be used for image tiling. Query by the following formats: Level (AA), Magnification (AAx), Resolution (A.AAmpp). Default: [0] - Maximum magnification")
    ap.add_argument('-n', '--normalizer', default="macenko", choices=NORMALIZER_CHOICES, help="Select the method for WSI color normalization. Default: 'macenko'. Options: [%s]" % ( ", ".join(NORMALIZER_CHOICES) ))
    ap.add_argument('-f', '--foreground_threshold', default=MIN_FOREGROUND_THRESHOLD, type=float, help="Defines the minimum tissue/background ratio for a tile to be considered foreground. Default: [%d]" % MIN_FOREGROUND_THRESHOLD)
    ap.add_argument('-r', '--normalizer_reference', default="None", type=str, help='H & E image used as a reference for normalization. Default: [wsitiler/normalizer/macenko_reference_img.png]')
    ap.add_argument('-v', '--verbose', action='count', help='Print updates and reports as program executes. Provide the following number of "v" for the following settings: [%d: Error. %d: Warning, %d: Info, %d: Debug]' % (LOG_ERROR,LOG_WARNING,LOG_INFO,LOG_DEBUG) ) #TODO: setup logging appropriately
    ap.add_argument('-t', '--tissue_chunk_id', action='store_true', help='Set this flag to determine tissue chunk ids for each tile: Default: [False]')
    ap.add_argument('-p', '--image_parameters', action='store_true', help='Set this flag to Display the image parameters for the first WSI found. Default: [False]')    
    args = vars(ap.parse_args())

    # args = {'input': '/home/clemenj/Data/testWSI/SG_40.svs', 'output': '/home/clemenj/Data/testWSI/test_tiles/', 'cores': 80, 'tile_dimensions': '512', 'image_level': '20x', 'normalizer': 'None', 'foreground_threshold': 0.01, 'normalizer_reference': 'None', 'verbose': 4, 'tissue_chunk_id': True, 'image_parameters': False}
    # {'input': '/home/clemenj/Data/testWSI/SG_38.svs', 'output': '/home/clemenj/Data/testWSI/test_tiles/', 'cores': 80, 'tile_dimensions': 256, 'normalizer': 'macenko', 'final_tile_size': 224, 'foreground_threshold': 0.01, 'normalizer_reference': 'None', 'verbose': 4, 'tissue_chunk_id': True} ##TODO remove

    # Validate arguments
    if args['verbose'] is not None:
        SET_LOG = args['verbose']
    if args['foreground_threshold'] < MIN_FOREGROUND_THRESHOLD:
        print("Warning: Tiles with very little foreground may fail color normalization.")
    if args['normalizer_reference'] != "None" and not os.path.exists(args['normalizer_reference']):
        raise ValueError("Normalizer reference image provided does not exist")

    PRINT_LOG(LOG_INFO, "Starting tiling run")

    total_start_time = time.time()
    PRINT_LOG(LOG_DEBUG, "Run Arguments: %s" % args)

    # Determine input image paths
    all_wsi_paths = []
    if os.path.isdir(args["input"]):
        input_path = Path(args["input"])
        for ftype in SUPPORTED_WSI_FORMATS:
            all_wsi_paths.extend(input_path.rglob("*"+ftype))
    else:
        if args["input"].endswith(tuple(SUPPORTED_WSI_FORMATS)):
            if os.path.exists(args["input"]):
                all_wsi_paths.append(Path(args["input"]))

    # Notify and terminate if no input found
    if len(all_wsi_paths) < 1:
        PRINT_LOG(LOG_ERROR, "No WSIs were found from input parameters. Terminating.")
        quit(code=1)

    PRINT_LOG(LOG_INFO, "Found WSI images. Starting Processing")
    PRINT_LOG(LOG_DEBUG, "The following WSIs were found:")
    if(SET_LOG >= LOG_DEBUG):
        for i,aPath in enumerate([str(s) for s in all_wsi_paths]):
            PRINT_LOG(LOG_DEBUG, "%d:\t%s" % (i+1,aPath) )

    # If requested, print image parameters and quit
    if args['image_parameters']:
        print("%d WSIs were found from input parameters.\nThe following are example parameters from <<%s>>:" 
            % (len(all_wsi_paths), all_wsi_paths[0]))
        a_wsi = openslide.open_slide(str(all_wsi_paths[0]))
        print(describe_wsi_levels(a_wsi))
        quit(code=0)
            
    #Prepare output path
    if os.path.isdir(args["output"]):
        outpath = Path(args["output"])

    # Process wsi images
    # i=0; wsi=all_wsi_paths[i] # TODO: remove
    for i,wsi in enumerate(all_wsi_paths):
        PRINT_LOG(LOG_INFO, "%d - Processing %s" % (i+1,str(wsi)))
        setup_start_time = time.time()
    
        # Prepare output path for wsi's tiles
        wsi_name = wsi.name.split(".")[0]
        out_tile_path = outpath / wsi_name

        #Open WSI
        try:
            wsi_image = openslide.open_slide(str(wsi))
        except (OpenSlideError,OpenSlideUnsupportedFormatError,UnidentifiedImageError) as e:
            # INFO
            PRINT_LOG(LOG_WARNING ,"%d - WARNING: WSI could not be read. Skipping: %s\n" % (i+1,str(wsi)))
            # DEBUG: Details for WSI reading error
            PRINT_LOG(LOG_DEBUG, ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
            
            continue
        
        PRINT_LOG(LOG_INFO ,"%d - Generating tile reference and mask" % (i+1) )
        tile_ref_start_time = time.time()
        
        # Prepare tiling reference
        img_level = find_wsi_level(wsi_image,args['image_level'])
        (ref_df, ppt_x, ppt_y) = prepare_tiles(wsi=wsi_image, output=str(out_tile_path), mpt=args["tile_dimensions"], wsi_level=img_level, get_chunk_id=args['tissue_chunk_id'])

        tile_ref_end_time = time.time()
        PRINT_LOG(LOG_DEBUG, "%d - Tile Reference Time: %f" % (i+1, tile_ref_end_time-tile_ref_start_time) )

        # Split non-empty tiles evenly for multiprocessing
        tile_data_lists = np.array_split(ref_df.loc[ref_df['tissue_ratio'] > args['foreground_threshold'] ], args['cores'])

        PRINT_LOG(LOG_INFO, "%d - Initializing normalizer" % (i+1) )
        PRINT_LOG(LOG_DEBUG, "%d - Normalizer method: %s \n%d - Normalizer reference: %s" % (i+1,args['normalizer'],i+1,args['normalizer_reference']) )

        # Prepare normalizer
        normalizer = setup_normalizer(normalizer_choice=args['normalizer'], ref_img_path=args['normalizer_reference'])

        PRINT_LOG(LOG_INFO, "%d - Exporting %d tiles" % (i+1, sum([len(x) for x in tile_data_lists]) ) )
        PRINT_LOG(LOG_DEBUG, "%d - Pool size: %d cores" % (i+1,args['cores']) )
        async_start_time = time.time()
                
        # Process tiles in parallel        
        pool = mp.Pool(args['cores'])
        for a_tile_list in tile_data_lists:
            pool.apply_async(func=export_tiles,kwds={
                'tile_data':a_tile_list,
                'wsi':str(wsi),
                'normalizer':normalizer,
                'tile_dims':{'x':ppt_x,'y':ppt_y},
                'output':str(out_tile_path),
                'wsi_level':img_level
                })
        pool.close()
        pool.join()
        pool.terminate()

        PRINT_LOG(LOG_INFO, "%d - Finished Exporting tiles" % (i+1) )
        async_end_time = time.time()
        PRINT_LOG(LOG_DEBUG, "%d - WSI Setup Time: %f" % (i+1, async_start_time-setup_start_time) )
        PRINT_LOG(LOG_DEBUG, "%d - Tile Export Time: %f" % (i+1, async_end_time-async_start_time) )
        PRINT_LOG(LOG_DEBUG, "%d - Total WSI Processing Time: %f" % (i+1, async_end_time-setup_start_time) )
        
    PRINT_LOG(LOG_INFO ,"Finished Processing All WSIs" )
    total_end_time = time.time()
    PRINT_LOG(LOG_DEBUG, "Total Time: %f" % (total_end_time-total_start_time) )
