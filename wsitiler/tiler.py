"""
tiler.py
Implement functions for dividing a whole slide image into tiles and save them as individual files.

Author: Jean R Clemenceau
Date Created: 11/11/2021
"""

import openslide
import argparse
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import traceback
import warnings

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
MICRONS_PER_TILE = 256
# NOISE_SIZE_MICRONS defines maximum size (in microns) for an artifact to be considered as noise in the WSI tissue mask.
NOISE_SIZE_MICRONS = 256
# FINAL_TILE_SIZE defines final pixel width and height of a processed tile.
FINAL_TILE_SIZE = 224
# MIN_FOREGROUND_THRESHOLD defines minimum tissue/background ratio to classify a tile as foreground.
MIN_FOREGROUND_THRESHOLD = 0
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
    if(SET_LOG >= loglevel) :
        print(log)


def setup_normalizer(normalizer_choice, ref_img_path=None):
    """
    Initialize a WSI normalizer object using the method of choice.

    Input:
        normalizer_choice (str): Valid choice for normalizer method. Use 'None' to return a Null object.
        ref_img_path (str): Path to reference image for the normalizer.

    Output:
        An initialized normalizer object:
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

def prepare_tiles(wsi, output, mpt=MICRONS_PER_TILE, get_chunk_id=False):
    """
    Import a WSI, calculate foreground/background, and calculate tile coordinates to output directory.

    Input:
        wsi (str): Path to WSI file to be processed.
        output (str): Path to output directory for processed tiles.
        mpt (int): Desire width and height of processed tiles in microns. Default: [%d].
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

    # Calculate desired tile dimensions (pixels per tile)
    ppt_x = int(mpt / float(wsi.properties['openslide.mpp-x']))
    ppt_y = int(mpt / float(wsi.properties['openslide.mpp-y']))

    # Get thumbnail for tissue mask
    thumbnail_og = wsi.get_thumbnail(size=(wsi.level_dimensions[-1][0], wsi.level_dimensions[-1][1]))
    thumbnail = np.array(thumbnail_og)
    thumbnail = (rgb2gray(thumbnail) * 255).astype(np.uint8)

    # calculate mask parameters
    thumbnail_ratio = wsi.dimensions[0] / thumbnail.shape[1]  # wsi dimensions: (x,y); thumbnail dimensions: (rows,cols)
    thumbnail_mpp = float(wsi.properties['openslide.mpp-x']) * thumbnail_ratio
    noise_size_pix = round(NOISE_SIZE_MICRONS / thumbnail_mpp)
    noise_size = round(noise_size_pix / thumbnail_ratio)
    thumbnail_ppt_x = ceil(ppt_x / thumbnail_ratio)
    thumbnail_ppt_y = ceil(ppt_y / thumbnail_ratio)
    tile_area = thumbnail_ppt_x*thumbnail_ppt_y

    # Create and clean tissue mask
    tissue_mask = (thumbnail[:, :] < threshold_otsu(thumbnail))
    tissue_mask = closing(tissue_mask, square(5))
    tissue_mask = opening(tissue_mask, square(5))
    tissue_mask = remove_small_objects(tissue_mask, noise_size)
    tissue_mask = ndi.binary_fill_holes(tissue_mask)

    if get_chunk_id:
        # Get labels for all chunks
        chunk_mask = ndi.label(tissue_mask)[0]

        # Filter out chunks smaller than tile size
        (chunk_label, chunk_size) = np.unique(chunk_mask,return_counts=True)
        filtered_chunks = chunk_label[ chunk_size < tile_area ]
        for l in filtered_chunks:
            chunk_mask[chunk_mask == l] = 0

    # Calculate margin according to ppt sizes
    wsi_x_tile_excess = wsi.dimensions[0] % ppt_x
    wsi_y_tile_excess = wsi.dimensions[1] % ppt_y

    # Determine WSI tile coordinates
    wsi_tiles_x = list(range(ceil(wsi_x_tile_excess / 2), wsi.dimensions[0] - floor(wsi_x_tile_excess / 2), ppt_x))
    wsi_tiles_y = list(range(ceil(wsi_y_tile_excess / 2), wsi.dimensions[1] - floor(wsi_y_tile_excess / 2), ppt_y))

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

            new_row = {"tile_id": slide_id,
                       "index_x": x,
                       "index_y": y,
                       "wsi_x": wsi_tiles_x[x],
                       "wsi_y": wsi_tiles_y[y],
                       "mask_x": mask_tiles_x[x],
                       "mask_y": mask_tiles_y[y],
                       "filename": os.path.basename(output) + "__tile-n-%d_x-%d_y-%d.png" % (slide_id, x, y),
                       "tissue_ratio": tissue_ratio
                       }

            if get_chunk_id:
                new_row['chunk_id'] = chunk_id

            rowlist.append(new_row)

    # Create reference dataframe
    colnames = ["tile_id", "index_x", "index_y", "wsi_x", "wsi_y", "mask_x", "mask_y", "filename", "tissue_ratio"]
    if get_chunk_id:
                colnames.append('chunk_id')
    ref_df = pd.DataFrame(data=rowlist, columns=colnames)

    # Remove filenames for empty tiles
    ref_df.loc[ref_df['tissue_ratio'] == 0, "filename"] = None

    output = Path(output) 

    # Export Mask image
    filename_tissuemask = os.path.basename(output) + "__tissue-mask_tilesize_x-%d-y-%d.png" % (
    thumbnail_ppt_x, thumbnail_ppt_y)
    plt.figure()
    plt.imshow(tissue_mask, cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig(output / filename_tissuemask, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Export Chunk Mask image
    if get_chunk_id:
        filename_tissuemask = os.path.basename(output) + "__chunk-mask_tilesize_x-%d-y-%d.png" % (
        thumbnail_ppt_x, thumbnail_ppt_y)
        plt.figure()
        plt.imshow(chunk_mask, cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.margins(0, 0)
        plt.savefig(output / filename_tissuemask, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Export Thumbnail image
    filename_thumbnail = os.path.basename(output) + "__thumbnail_tilesize_x-%d-y-%d.png" % (
        thumbnail_ppt_x, thumbnail_ppt_y)
    plt.figure()
    plt.imshow(thumbnail_og)
    plt.axis('off')
    plt.margins(0, 0)
    plt.savefig(output / filename_thumbnail, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Export CSV file
    filename_refdf = os.path.basename(output) + "__reference_wsi-tilesize_x-%d-y-%d_mask-tilesize_x-%d-y-%d.tsv" % \
                     (ppt_x, ppt_y,thumbnail_ppt_x, thumbnail_ppt_y)
    ref_df.to_csv(output / filename_refdf, sep="\t", line_terminator="\n", index=False)

    return (ref_df, ppt_x, ppt_y)

def export_tiles(wsi, tile_data, tile_dims, output="./", normalizer=None, final_tile_size=0):
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
        aTile_img = wsi_image.read_region((aTile["wsi_x"], aTile["wsi_y"]), level=0,
                                size=(tile_dims['x'], tile_dims['y']))
        ##TODO: check if this causes IO on every call & if it'd be better to read 1 region per list & extract tiles using numpy

        #Convert to RGB array
        aTile_img = np.array( aTile_img.convert('RGB') )

        # Normalize if required
        if normalizer is not None:
            aTile_img = normalizer.transform(aTile_img)

            # with warnings.catch_warnings():
            #     warnings.filterwarnings('error')
            #     try:
            #         aTile_img = normalizer.transform(aTile_img)
            #     except Warning as e:
            #         print('**Normalizer Warning Found! Tile: [x=%s,y=%s] File: %s Error:\n' % (aTile["wsi_x"], aTile["wsi_y"], wsi),
            #             traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__) )

        # Resize tile to final size
        if final_tile_size != 0:
            aTile_img = transform.resize(aTile_img, (final_tile_size,final_tile_size,3), order=1)  # 0:nearest neighbor

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
    ap.add_argument('-m', '--microns_per_tile', default=MICRONS_PER_TILE, type=int, help="Defines the tile edge length used when breaking WSIs into smaller images. Default: [%d]" % MICRONS_PER_TILE)
    ap.add_argument('-n', '--normalizer', default="macenko", choices=NORMALIZER_CHOICES, help="Select the method for WSI color normalization. Default: 'macenko'. Options: [%s]" % ( ", ".join(NORMALIZER_CHOICES) ))
    ap.add_argument('-z', '--final_tile_size', default=FINAL_TILE_SIZE, type=int, help="Defines the final tile size in pixels (N x N), give zero (0) for no resizing. If processed tile isn't this size, it will be interpolated to fit. Default: [%d]" % FINAL_TILE_SIZE)
    ap.add_argument('-f', '--foreground_threshold', default=MIN_FOREGROUND_THRESHOLD, type=int, help="Defines the minimum tissue/background ratio for a tile to be considered foreground. Default: [%d]" % MIN_FOREGROUND_THRESHOLD)
    ap.add_argument('-r', '--normalizer_reference', default="None", type=str, help='H & E image used as a reference for normalization. Default: [wsitiler/normalizer/macenko_reference_img.png]')
    ap.add_argument('-v', '--verbose', action='count', help='Print updates and reports as program executes. Provide the following number of "v" for the following settings: [%d: Error. %d: Warning, %d: Info, %d: Debug]' % (LOG_ERROR,LOG_WARNING,LOG_INFO,LOG_DEBUG) ) #TODO: setup logging appropriately
    ap.add_argument('-t', '--tissue_chunk_id', action='store_true', help='Set this flag to determine tissue chunk ids for each tile: Default: [False]')    
    args = vars(ap.parse_args())

    # args = {'input': '/home/clemenj/Data/testWSI/SG_38.svs', 'output': '/home/clemenj/Data/testWSI/test_tiles/', 'cores': 80, 'microns_per_tile': 256, 'normalizer': 'macenko', 'final_tile_size': 224, 'foreground_threshold': 0, 'normalizer_reference': 'None', 'verbose': 4, 'tissue_chunk_id': True} ##TODO remove

    # Validate arguments
    if args['verbose'] is not None:
        SET_LOG = args['verbose']
    #TODO: check normalizer reference
    #TODO: check input exists

    PRINT_LOG(LOG_INFO, "Starting tiling run")

    import time
    import tracemalloc
    total_start_time = time.time()
    PRINT_LOG(LOG_DEBUG, "Run Arguments: %s" % args)

    #Prepare output path
    if os.path.isdir(args["output"]):
        outpath = Path(args["output"])

    # Determine input image paths
    all_wsi_paths = []
    if os.path.isdir(args["input"]):
        input_path = Path(args["input"])
        for ftype in SUPPORTED_WSI_FORMATS:
            all_wsi_paths.extend(input_path.rglob("*"+ftype))
    else:
        if args["input"].endswith(tuple(SUPPORTED_WSI_FORMATS)):
            all_wsi_paths.append(Path(args["input"]))

    PRINT_LOG(LOG_INFO, "Found WSI images. Starting Processing")
    PRINT_LOG(LOG_DEBUG, "The following WSIs were found:")
    if(SET_LOG >= LOG_DEBUG):
        for i,aPath in enumerate([str(s) for s in all_wsi_paths]):
            PRINT_LOG(LOG_DEBUG, "%d:\t%s" % (i+1,aPath) )

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
        (ref_df, ppt_x, ppt_y) = prepare_tiles(wsi=wsi_image, output=str(out_tile_path), mpt=args["microns_per_tile"], get_chunk_id=args['tissue_chunk_id'])

        tile_ref_end_time = time.time()
        PRINT_LOG(LOG_DEBUG, "%d - Tile Reference Time: %f" % (i+1, tile_ref_end_time-tile_ref_start_time) )

        # Split non-empty tiles evenly for multiprocessing
        tile_data_lists = np.array_split(ref_df.loc[ref_df['tissue_ratio'] > args['foreground_threshold'] ], args['cores'])

        PRINT_LOG(LOG_INFO, "%d - Initializing normalizer" % (i+1) )
        PRINT_LOG(LOG_DEBUG, "%d - Normalizer method: %s \n%d - Normalizer reference: %s" % (i+1,args['normalizer'],i+1,args['normalizer_reference']) )

        # Prepare normalizer
        normalizer = setup_normalizer(normalizer_choice=args['normalizer'], ref_img_path=args['normalizer_reference'])

        PRINT_LOG(LOG_INFO, "%d - Exporting tiles" % (i+1) )
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
                'final_tile_size':args['final_tile_size']
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
