"""
annotate.py
Implementation of module for adding annotations to a WsiManager object.
Commandline interface implemented for module as well.

Author: Jean R Clemenceau
Date Created: 12/07/2022
"""

import argparse
import openslide
import numpy as np
import pandas as pd
import logging as log
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import transform
from matplotlib import colors
from skimage.filters import threshold_otsu
from skimage.color import rgba2rgb, rgb2gray

# import wsitiler.normalizer as norm
from wsitiler.WsiManager import WsiManager as wm


def annotate_from_mask(theWM: wm,label: str, maskPath: Path, label_color: str="red", value_threshold: float=None, tile_threshold: float=0, dry_run=False, outdir: Path=None):
    '''
    Annotates a WsiManager object's tiles according to a mask image of a given feature.

        Input:
            theWM (WsiManager): A WsiManager object to be annotated by a binary mask. Required.
            label (string): Name of the feature of interest to be annotated.
            maskPath (Path): maskPath (Path): File path to an image for a binary feature mask.
            label_color (str): Color name or hex code <#FFFFFF> used in input mask. (Optional: used if >1 mask present in image). Default: red
            value_threshold (float): Threshold value to dichotomize mask if original image is grayscale.(Optional)
            tile_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
            dry_run (bool): Whether or not to execute as dry-run (export image of resulting mask but not the annotated object)
            outdir (Path): File path to directory to export annotations to. Optional if WsiManager outdir field has been set.
        Output:
            Annotates WsiManager object by saving a mask and assigning tiles to object.
    '''
    log.info("Annotating from mask image -- ID: %s, Label: %s" % (theWM.wsi_id, label))

    if maskPath is None:
        raise ValueError("No mask or path to mask were given.")
    else:
        if type(maskPath) == str:
            maskPath = Path(maskPath)
        elif not issubclass( type(maskPath), Path):
            raise ValueError("Mask path value is not a file path.")

        if not maskPath.is_file():
            raise FileNotFoundError("Mask image file was not found.")
        else:
            # Import image
            mask = np.array(plt.imread(maskPath))

    # Validate dimensions for mask ndarray
    if len(mask.shape) < 2 or len(mask.shape) > 3:
        raise ValueError("Mask image format has too many dimensions.")
    
    #if image is RGB, make binary mask
    if len(mask.shape) == 3:

        if mask.shape[2]==2 or mask.shape[2] > 4:
            raise ValueError("Mask image has incompatible number of channels (must be grayscale or RGB).")
        
        #if mask image is formatted as RGBA, remove alpha channel
        if mask.shape[2]== 4:
            mask = np.array(rgba2rgb(mask))

        #if mask image is formatted as RGB, dichotomize to single channel
        if mask.shape[2] == 3:

            #if original image is grayscale, apply threshold
            if value_threshold is not None:
                log.debug("Dichotomizing mask from rgb with threshold: value > %f" % value_threshold)
                mask = rgb2gray(mask)
                mask = (mask > value_threshold)

            #if image is rgb color, dichotomize
            else:
                if label_color.startswith("#"):
                    posLabel = colors.hex2color(label_color)
                else:
                    posLabel = colors.to_rgb(label_color)

                mask = np.all(posLabel == mask, axis=2)

    # Ensure that mask is binary
    if len(np.unique(mask)) > 2:
        log.debug("Dichotomizing mask from single channel with threshold: value > %f" % value_threshold)
        mask = (mask*1 > value_threshold)
    else:
        # If mask is binary, make sure it is boolean
        mask = (mask*1 > 0)

    # Check aspect ratios
    thumbnail_ar = theWM.thumbnail.shape[0]/theWM.thumbnail.shape[1]
    mask_ar = mask.shape[0]/mask.shape[1]
    ar_err = abs(1-(mask_ar/thumbnail_ar))
    log.debug("Aspect Ratios -- Thumbnail: %f, Mask: %f, Percent diff: %f" % (thumbnail_ar, mask_ar, ar_err))

    if ar_err > 0.01:
        raise ValueError("Mask & slide proportions are too different. Consider image co-registration first.")
        #TODO: Implement co-registration

    #Find and validate output directory if given
    if outdir is None and theWM.outdir is None:
        raise ValueError("No output directory has been supplied")
    elif outdir is not None:
        if isinstance(outdir,str):
            outdir = Path(outdir)
        elif not isinstance(outdir,Path):
            raise ValueError("Output path is not a Path or a string.")

        #if output directory has correct format, use it directly.
        if outdir.suffix == "":
            theWM.outdir = outdir
        else:
            raise ValueError("Output path does not have directory format.")

    #Apply annotation
    theWM.annotate_from_binmask(label=label,mask=mask,label_color=label_color,threshold=tile_threshold, export_mask=dry_run)

    # if not dry_run:
    #     log.debug("Exporting annotated WsiManager object -- ID: %s, label: %s, output: %s" % (theWM.wsi_id, label, theWM.outdir) )
    #     theWM.export_info()

    return

def annotate_from_thumbnail(theWM: wm,label: str, maskPath: Path, label_color: str="red", tile_threshold: float=0, dry_run=False, outdir: Path=None):
    '''
    Annotates a WsiManager object's tiles according to an annotated image of the WSI's thumbnail image of a given feature.

         Input:
            theWM (WsiManager): A WsiManager object to be annotated by a binary mask. Required.
            label (string): Name of the feature of interest to be annotated.
            maskPath (Path): maskPath (Path): File path to an image for a binary feature mask.
            label_color (str): Color name or hex code <#FFFFFF> used in input mask. (Optional: used if >1 mask present in image). Default: red
            tile_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
            dry_run (bool): Whether or not to execute as dry-run (export image of resulting mask but not the annotated object)
            outdir (Path): File path to directory to export annotations to. Optional if WsiManager outdir field has been set.
        Output:
            Annotates WsiManager object by saving a mask and assigning tiles to object.
    '''
    log.info("Annotating from thumbnail image -- ID: %s, Label: %s, color: %s" % (theWM.wsi_id, label, label_color))

    if maskPath is None:
        raise ValueError("No mask or path to mask were given.")
    else:
        if type(maskPath) == str:
            maskPath = Path(maskPath)
        elif not issubclass( type(maskPath), Path):
            raise ValueError("Mask path value is not a file path.")

        if not maskPath.is_file():
            raise FileNotFoundError("Mask image file was not found.")
        else:
            # Import image
            mask = np.array(plt.imread(maskPath))

    # Validate dimensions for mask ndarray
    if len(mask.shape) < 2 or len(mask.shape) > 3:
        raise ValueError("Mask image format has wrong number of dimensions.")
    
    #if image is RGB, make binary mask
    if len(mask.shape) == 3:

        if mask.shape[2]<2 or mask.shape[2] > 4:
            raise ValueError("Mask image has incompatible number of channels (must be RGB).")
        
        # If mask image is formatted as RGBA, remove alpha channel
        if mask.shape[2]== 4:
            mask = np.array(rgba2rgb(mask))

        # Dichotomize rgb image using requested label color
        if mask.shape[2] == 3:

            # Determine positive lable according to given format
            if label_color.startswith("#"):
                posLabel = colors.hex2color(label_color)
            else:
                posLabel = colors.to_rgb(label_color)

            # Ensure binary mask is boolean
            mask = np.all(posLabel == mask, axis=2)

    # Check aspect ratios
    thumbnail_ar = theWM.thumbnail.shape[0]/theWM.thumbnail.shape[1]
    mask_ar = mask.shape[0]/mask.shape[1]
    ar_err = abs(1-(mask_ar/thumbnail_ar))
    log.debug("Aspect Ratios -- Thumbnail: %f, Mask: %f, Percent diff: %f" % (thumbnail_ar, mask_ar, ar_err))

    if ar_err > 0.01:
        raise ValueError("Mask & slide proportions are too different. Consider image co-registration first.")
        #TODO: Implement co-registration

    #Find and validate output directory if given
    if outdir is None and theWM.outdir is None:
        raise ValueError("No output directory has been supplied")
    elif outdir is not None:
        if isinstance(outdir,str):
            outdir = Path(outdir)
        elif not isinstance(outdir,Path):
            raise ValueError("Output path is not a Path or a string.")

        #if output directory has correct format, use it directly.
        if outdir.suffix == "":
            theWM.outdir = outdir
        else:
            raise ValueError("Output path does not have directory format.")

    #Apply annotation
    theWM.annotate_from_binmask(label=label,mask=mask,label_color=label_color,threshold=tile_threshold, export_mask=dry_run)

    return


# def annotate_from_ihc(theWM: wm,label: str, maskPath: Path):
#     '''
#     Annotates a WsiManager object's tiles by generating binary mask of an IHC slide's DAP channel.

#         Input:
#             theWM (WsiManager): A WsiManager object to be annotated by a binary mask. Required.
#             label (string): Name of the feature of interest to be annotated.
#             maskPath (Path): maskPath (Path): File path to an image for a binary feature mask.
#             label_color (str): Color name or hex code <#FFFFFF> used in input mask.
#             value_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
#             tile_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
#         Output:
#             Annotates WsiManager object by saving a mask and assigning tiles to object.
#     '''

# def annotate_from_multiplex_ihc(theWM: wm,label: str, maskPath: Path):
#     '''
#     Annotates a WsiManager object's tiles according to a binary mask image of a given feature.

#         Input:
#             theWM (WsiManager): A WsiManager object to be annotated by a binary mask. Required.
#             label (string): Name of the feature of interest to be annotated.
#             maskPath (Path): maskPath (Path): File path to an image for a binary feature mask.
#             label_color (str): Color name or hex code <#FFFFFF> used in input mask.
#             value_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
#             tile_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
#         Output:
#             Annotates WsiManager object by saving a mask and assigning tiles to object.
#     '''

# def annotate_from_binmask(theWM: wm, label: str, mask: np.ndarray=None, maskPath: Path=None, label_color: str="red", threshold: float=0):
#     """
#     Exports metadata from a about tissue chunks in a given WsiMaanager object.

#     Input:
#         theWM (WsiManager): A WsiManager object to be annotated by a binary mask. Required.
#         mask (Path): File path to directory containing a WsiManager's files.
#         maskPath (Path): File path to directory containing a WsiManager's files.
#         label_color (str): Coloe name or hex code <#FFFFFF> used in input mask.
#         threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
#     Output:
#         Annotates WsiManager object by saving a mask and assigning tiles.
#     """

#     log.info("Annotating from binary mask -- ID: %s, Label: %s" % (theWM.wsi_id, label))

#     # Validate mask input
#     if mask is None:
#         if maskPath is None:
#             raise ValueError("No mask or path to mask were given.")
#         else:
#             if type(maskPath) == str:
#                 maskPath = Path(maskPath)
#             elif not issubclass( type(maskPath), Path):
#                 raise ValueError("Mask path value is not a file path.")

#             if not maskPath.is_file():
#                 raise FileNotFoundError("Mask image file was not found.")
#             else:
#                 # Import image
#                 mask = np.array(plt.imread(maskPath))

#     # Validate dimensions for mask ndarray
#     if len(mask.shape) < 2 or len(mask.shape) > 4:
#         raise ValueError("Mask image format has too many dimensions.")

#     #check if mask image is formatted as RGBA
#     if len(mask.shape) == 4:
#         mask = np.array(rgba2rgb(mask))
    
#     #if image is RGB, make binary mask
#     if len(mask.shape) == 3:

#         if mask.shape[2]==2 or mask.shape[2] > 3:
#             raise ValueError("Mask image has incompatible number of channels (must be grayscale or RGB).")

#         elif mask.shape[2] == 3:
#             #Process positive label color
#             if label_color.startswith("#"):
#                 posLabel = colors.hex2color(label_color)
#             else:
#                 posLabel = colors.to_rgb(label_color)

#             mask = np.all(posLabel == mask, axis=2)

#     # Validate that mask is binary
#     if len(np.unique(mask)) > 2:
#         raise ValueError("The provided mask is not binary.")
#     else:
#         # If mask is binary, make sure it is boolean
#         mask = (mask*1 > 0)

#     # Check aspect ratios
#     thumbnail_ar = theWM.thumbnail.shape[0]/theWM.thumbnail.shape[1]
#     mask_ar = mask.shape[0]/mask.shape[1]
#     ar_err = abs(1-(mask_ar/thumbnail_ar))
#     log.debug("Aspect Ratio too different -- WSI: %f, Mask: %f, Percent diff: %f" % (thumbnail_ar, mask_ar, ar_err))

#     if ar_err > 0.01:
#         raise ValueError("Mask & slide proportions are too different. Consider image co-registration first.")
#         #TODO: Implement co-registration
#     else:
#         # Fit mask to thumbnail if within margin of error (<=1%)
#         mask = transform.resize(mask,theWM.tissue_mask.shape,order=0)

#     # Save mask to object
#     maskName = label+"_mask"
#     setattr(theWM,maskName,mask)
#     log.debug("Saved mask to object as: %s" % (maskName))

#     # Prepare functions to check which tiles pass mask threshold
#     mask_tile = lambda x,y: mask[y:y+theWM.thumbnail_ppt_y, x:x+theWM.thumbnail_ppt_x]
#     tile_check = lambda aTile: (np.sum(mask_tile(aTile['mask_x'],aTile['mask_y'])) / mask_tile(aTile['mask_x'],aTile['mask_y']).size) > threshold

#     # Create new column for new labels
#     theWM.tile_data[label] = False

#     # Identify tissue foreground for computational efficiency
#     tissue_tiles = theWM.tile_data[ ~pd.isnull(theWM.tile_data.tilename) ].index

#     # Annotate tiles where mask presence is above threshold
#     log.debug("Labeling %d foreground tiles as positive if mask proportion > %f" % (len(tissue_tiles), threshold))
#     theWM.tile_data.loc[tissue_tiles,label] = theWM.tile_data.iloc[tissue_tiles].apply(tile_check, axis=1)

#     return(theWM)


####################

# Log Levels
VERBOSE_LEVEL = [log.ERROR, log.WARNING, log.INFO, log.DEBUG]
MASK_TYPES = {"binary":"annotate_from_mask","continuous":"annotate_from_mask","thumbnail":"annotate_from_thumbnail",}
MASK_TYPE_KEYS = list(MASK_TYPES.keys())

if __name__ == '__main__':
    # Define command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default="./", help='WsiManagerData.json file paths or Comma-separated list of input directories. Default: [./]')
    ap.add_argument('-o', '--output', default=None, help="Output directory Default: Input object's directory")
    ap.add_argument('-m', '--mask', default=None, help="Filaname or comma-separated list of directories containing annotations masks. Filenames must match objects' wsi_id. Required")
    ap.add_argument('-t', '--mask_type', choices=MASK_TYPE_KEYS, default=MASK_TYPE_KEYS[0], help="Type of annotation masks.Default: [%s]" % MASK_TYPE_KEYS[0])
    ap.add_argument('-l', '--label', default=None, help='Filaname or comma-separated list of directories containing annotations masks. Optional if providing settings.csv file')
    ap.add_argument('-s', '--settings', default=None, help='Numeric threshold value for non-binary masks, color name or hex value for binary mask, or path to settings.csv file. Required')
    ap.add_argument('-c', '--cores', default=mp.cpu_count(), type=int, help='Numbers of processes to be spun up in parallel to process each WSI. Default: [%d]' % mp.cpu_count() )
    ap.add_argument('-v', '--verbose', action='count', help='Print updates and reports as program executes. Provide the following number of "v" for the following settings: [Default: Error, v: Warning, vv: Info, vvv: Debug]')
    ap.add_argument('-y', '--dry_run', action='store_true', help='Set this flag to output annotation maps without saving annotated object (Good for testing parameters). Default: [False]')  
    # args = vars(ap.parse_args())
    
    args = vars(ap.parse_args(['-i','/home/clemenj/Data/brain/Mouse_PDX/tiled_wsis/SG_01/','-m','/home/clemenj/Data/BLCA_TRRC2819/CCF_Batch2_outputs/pred_tils/SG_01_color.png','-l','predicted_TIL','-s','red','-c','20','-vvvvv'])) #TODO: rm
    

    # Determine Verbosity
    if args['verbose'] is not None:
        if args['verbose'] > len(VERBOSE_LEVEL)-1:
            log.getLogger().setLevel(VERBOSE_LEVEL[-1])
        else:
            log.getLogger().setLevel(VERBOSE_LEVEL[args['verbose']])
    else:
        log.getLogger().setLevel(log.ERROR)

    #Validate output path
    if args['output'] is not None:
        output = Path(args['output'])
        if not output.is_dir():
            log.warning("Output directory not found, creating directory.")
            output.mkdir(parents=True)

    # Determine input files
    all_wm_json_paths = [Path(i) for i in args['input'].split(',') if i.endswith("___WsiManagerData.json") and Path(i).is_file()]

    # Determine input files if directories were given
    if len(all_wm_json_paths) < 1:
        # Determine input paths
        all_input_dirs = [Path(i) for i in args['input'].split(',') if Path(i).is_dir()]
        
        #Find all exported wm json files
        all_wm_json_paths = []
        for aPath in all_input_dirs:
            all_wm_json_paths += list(aPath.glob("**/*___WsiManagerData.json"))

    #List the WM object's directories based on json file location
    all_wm_paths = [aPath.parent.parent for aPath in all_wm_json_paths]
    
    # Notify and terminate if no input found
    if len(all_wm_paths) < 1:
        log.error( "No directories were found from input parameters. Terminating.")
        quit(code=1)
    else:
        log.info("%d WsiManager paths found" % len(all_wm_paths))
        log.debug("WsiManager paths to be processed:\n%s" % "\n".join([str(i) for i in all_wm_paths]) )

    # Determine label masks
    if args['mask'] is None:
        log.error( "No masks were found from input parameters. Terminating.")
        quit(code=1)

    # Find masks files
    all_mask_files = [Path(i) for i in args['mask'].split(',') if i.endswith('.jpg') or i.endswith('.png') or Path(i).is_file()]
    all_mask_dirs = [Path(i) for i in args['mask'].split(',') if Path(i).is_dir()]

    #TODO: check settings according to mask type

    #Determine core count
    core_cnt = min(len(all_wm_paths), args['cores'])
    log.debug("Processing WM instances using %d cores" % core_cnt)

    #TODO: execute annotation (check for dry-run)

    #TODO: If not dry run, reexport wm info

    


    