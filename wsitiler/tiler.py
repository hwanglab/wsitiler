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

from pathlib import Path
from math import ceil, floor
from skimage import transform
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, opening,closing, square
from scipy.ndimage import binary_fill_holes

import wsitiler.normalizer as norm

# MICRONS_PER_TILE defines the tile edge length used when breaking WSIs into smaller images (m x m)
MICRONS_PER_TILE = 256
# NOISE_SIZE_MICRONS defines maximum size (in microns) for an artifact to be considered as noise in the WSI tissue mask.
NOISE_SIZE_MICRONS = 256
# FINAL_TILE_SIZE defines final pixel width and height of a processed tile.
FINAL_TILE_SIZE = 224
# MIN_FOREGROUND_THRESHOLD defines minimum tissue/background ratio to classify a tile as foreground.
MIN_FOREGROUND_THRESHOLD = 0
# HE_REF_IMG defines the path to the default reference image for WSI color normalization.
# HE_REF_IMG = str(Path(__file__).absolute() / "normalizer/macenko_reference_img.png") # TODO: restore
HE_REF_IMG = str(Path().absolute() / "wsitiler/normalizer/macenko_reference_img.png") #TODO: delete
# NORMALIZER_CHOICES defines the valid choices for WSI normalization methods.
NORMALIZER_CHOICES= ["None","macenko"]
# SUPPORTED_WSI_FORMATS defines the WSI formats supported by Openslide.
SUPPORTED_WSI_FORMATS = [".svs",".ndpi",".vms",".vmu",".scn",".mrxs",".tiff",".svslide",".tif",".bif"]

def setup_normalizer(normalizer_choice, ref_img_path):
    """
    Initialize a WSI normalizer object using the method of choice.

    Input:
        normalizer_choice (str): Valid choice for normalizer method. Use 'None' to return a Null object.
        ref_img_path (str): Path to reference image for the normalizer.

    Output:
        An initialized normalizer object:
    """

    normalizer = None
    ref_img = plt.imread(str(ref_img_path))

    # Initialize normalizer & setup reference image if required
    if normalizer_choice is not None and normalizer_choice != "None":
        if normalizer_choice in NORMALIZER_CHOICES:
            if normalizer_choice == "macenko":
                normalizer = norm.MacenkoNormalizer.MacenkoNormalizer()
                ref_img = np.array(ref_img)  
            
            # Add more options here as "else if" blocks      

        normalizer.fit(ref_img)

    return normalizer

def prepare_tiles(wsi, output, mpt=MICRONS_PER_TILE):
    """
    Import a WSI, calculate foreground/background, and calculate tile coordinates to output directory.

    Input:
        wsi (str): Path to WSI file to be processed.
        output (str): Path to output directory for processed tiles.
        mpt (int): Desire width and height of processed tiles in microns. Default: [%d].

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

    # Create and clean tissue mask
    tissue_mask = (thumbnail[:, :] < threshold_otsu(thumbnail))
    tissue_mask = closing(tissue_mask, square(5))
    tissue_mask = opening(tissue_mask, square(5))
    tissue_mask = remove_small_objects(tissue_mask, noise_size)
    tissue_mask = binary_fill_holes(tissue_mask)

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

            # Calculate tissue ratio for tile
            tissue_ratio = np.sum(aTile) / aTile.size

            slide_id = len(rowlist) + 1

            new_row = {"tile_id": slide_id,
                       "wsi_x": wsi_tiles_x[x],
                       "wsi_y": wsi_tiles_y[y],
                       "mask_x": mask_tiles_x[x],
                       "mask_y": mask_tiles_y[y],
                       "filename": os.path.basename(output) + "__tile-n-%d_x-%d_y-%d.png" % (slide_id, x, y),
                       "tissue_ratio": tissue_ratio
                       }

            rowlist.append(new_row)

    # Create reference dataframe
    colnames = ["tile_id", "wsi_x", "wsi_y", "mask_x", "mask_y", "filename", "tissue_ratio"]
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
        aTile_img = np.array(aTile_img)[:,:,0:3]

        # Normalize if required
        if normalizer is not None:
            aTile_img = normalizer.transform(aTile_img)

        # Resize tile to final size
        if final_tile_size != 0:
            aTile_img = transform.resize(aTile_img, (final_tile_size,final_tile_size,3), order=1)  # 0:nearest neighbor

        # Save tile image to file
        plt.imshow(aTile_img);plt.axis('off');plt.margins(0, 0);plt.savefig(output / aTile['filename'], bbox_inches='tight', pad_inches=0);plt.close()

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
    ap.add_argument('-r', '--normalizer_reference', default=HE_REF_IMG, type=str, help='H & E image used as a reference for normalization. Default: [%s]' % HE_REF_IMG )
    ap.add_argument('-v', '--verbose', action='count', help='Print updates and reports as program executes.') #TODO: setup logging appropriately
    args = vars(ap.parse_args())
    # args = vars(ap.parse_args(["-i","C:/Users/clemenj/Documents/Data_local/testWSI/","-o","C:/Users/clemenj/Documents/Data_local/testWSI/test_tiles/"]))# TODO: remove
    # args = vars(ap.parse_args(["-i","/home/clemenj/Data/testWSI/","-o","/home/clemenj/Data/testWSI/test_tiles/","-c","16","-n", "macenko"]))# TODO: remove

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

    # Process wsi images
    # wsi=all_wsi_paths[0] # TODO: remove
    for wsi in all_wsi_paths: #TODO: restore
        # setup_start_time = time.time() # TODO: remove
        # Prepare output path for wsi's tiles
        wsi_name = wsi.name.split(".")[0]
        out_tile_path = outpath / wsi_name

        #Open WSI
        wsi_image = openslide.open_slide(str(wsi))
        # TODO: catch errors opening, report and skip

        # Execute tiling
        (ref_df, ppt_x, ppt_y) = prepare_tiles(wsi=wsi_image, output=str(out_tile_path), mpt=args["microns_per_tile"])

        # Split non-empty tiles evenly for multiprocessing
        tile_data_lists = np.array_split(ref_df.loc[ref_df['tissue_ratio'] > args['foreground_threshold'] ], args['cores'])

        # Prepare normalizer
        normalizer = setup_normalizer(normalizer_choice=args['normalizer'], ref_img_path=args['normalizer_reference'])

        # Process tiles in parallel        
        pool = mp.Pool(args['cores'])
        # async_start_time = time.time() # TODO: remove
        for a_tile_list in tile_data_lists:
            pool.apply_async(func=export_tiles,kwds={'tile_data':a_tile_list,'wsi':str(wsi),'normalizer':normalizer,
                                                        'tile_dims':{'x':ppt_x,'y':ppt_y},'output':str(out_tile_path),
                                                        'final_tile_size':args['final_tile_size']})
        pool.close()
        pool.join()
        pool.terminate()
        # async_end_time = time.time() # TODO: remove

        # print("%s - Setup time: %f" % (wsi_name, async_start_time-setup_start_time) )# TODO: remove
        # print("%s - Tiling time: %f" % (wsi_name, async_end_time-async_start_time) )# TODO: remove
        # print("%s - Total time: %f" % (wsi_name, async_end_time-setup_start_time) )# TODO: remove




########################Debbuging lines #TODO: remove
# ap.add_argument('-e', '--export_figures', action='store_true', help='Flag for saving all heatmaps as PNG images.')

# import time
# plt.imshow(tissue_mask);plt.show();plt.close()
# plt.imshow(aTile_img);plt.show();plt.close()
# export_tiles(tile_data=tile_data_lists[0].iloc[0:10],wsi=str(wsi),normalizer=normalizer,tile_dims={'x':ppt_x,'y':ppt_y},output=str(out_tile_path),final_tile_size=args['final_tile_size'])