"""
annotate.py
Implementation of module for adding annotations to a WsiManager object.
Commandline interface implemented for module as well.

Author: Jean R Clemenceau
Date Created: 12/07/2022
"""

import argparse
import numpy as np
import pandas as pd
import logging as log
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path
from skimage import transform
from matplotlib import colors
from skimage.filters import threshold_otsu, threshold_yen
from skimage.color import rgba2rgb, rgb2gray, rgb2hed
from skimage.morphology import opening, closing, square, remove_small_objects
# import wsitiler.normalizer as norm
from wsitiler.WsiManager import WsiManager as wm
from wsitiler.wsi_utils import read_fluorescent_ome_tiff

## Utility functions ##

def isfloat(num: str):
    '''
    Determines if a string holds a float.
        Input:
            num (str): A string that will be checked for a float. Required.
        Output:
            a float value or None if not compatible.
    '''
    try:
        f = float(num)
        return f
    except ValueError:
        return None
    except TypeError:
        return None

def iscolor(col: str):
    '''
    Determines if a string holds a color.
        Input:
            col (str): A string that will be checked for a color. Required.
        Output:
            a hex color value or None if not compatible.
    '''
    try:
        if col.startswith("#"):
            c = colors.hex2color(col)
        else:
            c = colors.to_rgb(col)
        return colors.to_hex(c)
    except ValueError:
        return None
    except AttributeError:
        return None

def exportWM(theWM: wm):
    '''
    Exports info for a WsiManager object (for multiprocesising).
        Input:
            theWM (WsiManager): WsiManager object to export info.
        Output:
            None
    '''
    theWM.export_info()
    return

## Annotation functions##

def annotate_from_mask(theWMpath: Path,label: str, maskPath: Path, label_color: str="red", value_threshold: float=None, tile_threshold: float=0, dry_run=False, outdir: Path=None):
    '''
    Annotates a WsiManager object's tiles according to a mask image of a given feature.

        Input:
            theWMpath (Path): A Path to the directory of the WsiManager to be annotated by a binary mask. Required.
            label (string): Name of the feature of interest to be annotated.
            maskPath (Path): maskPath (Path): File path to an image for a binary feature mask.
            label_color (str): Color name or hex code <#FFFFFF> used in input mask. (Optional: used if >1 mask present in image). Default: red
            value_threshold (float): Threshold value to dichotomize mask if original image is grayscale.(Optional)
            tile_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
            dry_run (bool): Whether or not to execute as dry-run (export image of resulting mask but not the annotated object)
            outdir (Path): File path to directory to export annotations to. Optional if WsiManager outdir field has been set.
        Output:
            Annotated WsiManager object by saving a mask and assigning tiles to object.
    '''

    #Validate and find WM object
    try:
        theWM = wm.fromdir(theWMpath)
        log.info("Annotating from mask image -- ID: %s, Label: %s" % (theWM.wsi_id, label))
    except FileNotFoundError:
        raise FileNotFoundError("No WsiManager object was found in input path: %s" % str(theWMpath))

    # Validate and find mask input
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
            log.debug("Output directory reset-- Wsi_id: %s, Outdir: %s" % (theWM.wsi_id, theWM.outdir))

        else:
            raise ValueError("Output path does not have directory format.")

    #Apply annotation
    theWM.annotate_from_binmask(label=label,mask=mask,label_color=label_color,threshold=tile_threshold, export_mask=dry_run)

    # if not dry_run:
    #     log.debug("Exporting annotated WsiManager object -- ID: %s, label: %s, output: %s" % (theWM.wsi_id, label, theWM.outdir) )
    #     theWM.export_info()

    return theWM

def annotate_from_thumbnail(theWMpath: Path,label: str, maskPath: Path, label_color: str="red", tile_threshold: float=0, dry_run=False, outdir: Path=None, clean_edges: bool=False):
    '''
    Annotates a WsiManager object's tiles according to an annotated image of the WSI's thumbnail image of a given feature.

         Input:
            theWMpath (Path): A Path to the directory of the WsiManager to be annotated by a binary mask. Required.
            label (string): Name of the feature of interest to be annotated.
            maskPath (Path): maskPath (Path): File path to an image for a binary feature mask.
            label_color (str): Color name or hex code <#FFFFFF> used in input mask. (Optional: used if >1 mask present in image). Default: red
            tile_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
            dry_run (bool): Whether or not to execute as dry-run (export image of resulting mask but not the annotated object)
            outdir (Path): File path to directory to export annotations to. Optional if WsiManager outdir field has been set.
            clean_edges (bool): Whether or not to clean up edges of mask image.
        Output:
            Annotated WsiManager object by saving a mask and assigning tiles to object.
    '''
    
    #Validate and find WM object
    try:
        theWM = wm.fromdir(theWMpath)
        log.info("Annotating from mask image -- ID: %s, Label: %s" % (theWM.wsi_id, label))
    except FileNotFoundError:
        raise FileNotFoundError("No WsiManager object was found in input path: %s" % str(theWMpath))

    # Validate and find mask input
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

    # Cleanup mask edges
    if(clean_edges):
        mask = closing(mask, square(5))
        mask = opening(mask, square(5))

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

    return theWM


def annotate_from_ihc(theWMpath: Path,label: str, maskPath: Path, value_threshold: float=None, tile_threshold: float=0, dry_run=False, outdir: Path=None):
    '''
    Annotates a WsiManager object's tiles by generating binary mask of an IHC slide's DAP channel.

         Input:
            theWMpath (Path): A Path to the directory of the WsiManager to be annotated by a binary mask. Required.
            label (string): Name of the feature of interest to be annotated.
            maskPath (Path): maskPath (Path): File path to a chromogenic IHC slide image used for a binary feature mask.
            value_threshold (float): Threshold value to dichotomize IHC's DAB channel.(Optional)
            tile_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
            dry_run (bool): Whether or not to execute as dry-run (export image of resulting mask but not the annotated object)
            outdir (Path): File path to directory to export annotations to. Optional if WsiManager outdir field has been set.
        Output:
            Annotated WsiManager object by saving a mask and assigning tiles to object.
    '''
    
    #Validate and find WM object
    try:
        theWM = wm.fromdir(theWMpath)
        log.info("Annotating from mask image -- ID: %s, Label: %s" % (theWM.wsi_id, label))
    except FileNotFoundError:
        raise FileNotFoundError("No WsiManager object was found in input path: %s" % str(theWMpath))

    # Validate and find mask input
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
    if len(mask.shape) != 3:
        raise ValueError("IHC Mask image format has wrong number of dimensions.")
    # Verify image is RGB & remove Alpha channel
    else:
        if mask.shape[2]<2 or mask.shape[2] > 4:
            raise ValueError("Mask image has incompatible number of channels (must be RGB).")
        # If mask image is formatted as RGBA, remove alpha channel
        if mask.shape[2]== 4:
            mask = np.array(rgba2rgb(mask))

    # find  and clean IHC tissue mask
    mask_gray = (rgb2gray(mask) * 255)
    ihc_tissue_mask = (mask_gray[:, :] < threshold_otsu(mask_gray))
    ihc_tissue_mask = remove_small_objects(ihc_tissue_mask, 5)
    ihc_tissue_mask = closing(ihc_tissue_mask, square(5))
    ihc_tissue_mask = opening(ihc_tissue_mask, square(5))

    # Check aspect ratios 
    thumbnail_ar = theWM.thumbnail.shape[0]/theWM.thumbnail.shape[1]
    mask_ar = mask.shape[0]/mask.shape[1]
    ar_err = abs(1-(mask_ar/thumbnail_ar))
    log.debug("Aspect Ratios -- Thumbnail: %f, Mask: %f, Percent diff: %f" % (thumbnail_ar, mask_ar, ar_err))

    if ar_err > 0.01:
        raise ValueError("Mask & slide proportions are too different. Consider image co-registration first.")
        #TODO: Implement co-registration

    # Identify isolate DAB channel
    mask_hed = rgb2hed(mask)
    label_img = mask_hed[:,:,2]

    # Determine threshold for DAB channel
    if value_threshold is not None:
        log.debug("Dichotomizing IHC mask from DAB channel with supplied threshold: value > %f" % value_threshold)
    else:
        value_threshold = threshold_yen(label_img)
        log.debug("Dichotomizing IHC mask from DAB channel with Yen's threshold: value > %f" % value_threshold)

    # Apply threshold and generate DAB mask for labeling
    label_mask = (label_img > value_threshold)
    label_mask = closing(label_mask, square(5))
    label_mask = opening(label_mask, square(5))
    label_mask = remove_small_objects(label_mask, 5)

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
            log.debug("Output directory reset-- Wsi_id: %s, Outdir: %s" % (theWM.wsi_id, theWM.outdir))

        else:
            raise ValueError("Output path does not have directory format.")

    #Apply annotation
    theWM.annotate_from_binmask(label=label,mask=label_mask,threshold=tile_threshold, export_mask=dry_run)
    #TODO: if dry run, export mask with H&E as background

    return theWM

    
def annotate_from_multiplex_ihc(theWMpath: Path,label: str, maskPath: Path, label_channel: str="0", value_threshold: float=None, tile_threshold: float=0, dry_run=False, outdir: Path=None, clean_edges: bool=False):
    '''
    Annotates a WsiManager object's tiles by generating binary mask of a fluorescent IHC slide's selected channel.

         Input:
            theWMpath (Path): A Path to the directory of the WsiManager to be annotated by a binary mask. Required.
            label (string): Name of the feature of interest to be annotated.
            maskPath (Path): maskPath (Path): File path to an OME-TIFF fluorescent IHC image used for a binary feature mask.
            label_channel (str): Channel index or name (protein) used as input mask. Default: 0
            value_threshold (float): Threshold value to dichotomize IHC's DAB channel.(Optional)
            tile_threshold (float): Threshold value for proportion of mask in a tile necessary to annotate a tile.
            dry_run (bool): Whether or not to execute as dry-run (export image of resulting mask but not the annotated object)
            outdir (Path): File path to directory to export annotations to. Optional if WsiManager outdir field has been set.
            clean_edges (bool): Whether or not to clean up edges of mask image.

        Output:
            Annotated WsiManager object by saving a mask and assigning tiles to object.
    '''

    #Validate and find WM object
    try:
        theWM = wm.fromdir(theWMpath)
        log.info("Annotating from mask image -- ID: %s, Label: %s" % (theWM.wsi_id, label))
    except FileNotFoundError:
        raise FileNotFoundError("No WsiManager object was found in input path: %s" % str(theWMpath))

    # Validate and find mask input
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
            log.debug("Importing fIHC TIF image from: %s" % str(maskPath))
            (mask_meta, mask_ihc) = read_fluorescent_ome_tiff(maskPath)

    # Validate channel selection
    if isinstance(label_channel, str):
        if label_channel.isnumeric():
            label_channel = int(label_channel)
        elif label_channel in mask_meta['channels']:
            label_idx = mask_meta['channels'].index(label_channel)
        else:
            ValueError("IHC channel name is not available. Available options are: %s" % (", ".join(mask_meta['channels'])))

    if isinstance(label_channel, int):
        if label_channel in range(0,len(mask_meta['channels'])):
            label_idx = label_channel
        else:
            raise ValueError("IHC channel index is out of bounds. Value must be within [%d,%d]" % (0,len(mask_meta['channels']-1)))
    
    #Get tissue mask (based on sum of all channels)
    mask_sum = mask_ihc.sum(axis=2)
    mask_sum = transform.rescale(mask_sum,(1/theWM.thumbnail_ratio), anti_aliasing=False)
    mask_sum = (mask_sum > threshold_otsu(mask_sum))
    mask_sum = closing(mask_sum, square(5))
    mask_sum = opening(mask_sum, square(5))

    noise_size = (theWM.NOISE_SIZE_MICRONS * mask_meta['mpp']) / theWM.thumbnail_ratio
    mask_sum = remove_small_objects(mask_sum, noise_size)

    # TODO Coregister images #####################################################################################################################
        
    #Get mask for label
    label_channel = mask_meta['channels'][label_idx]
    mask = mask_ihc[:,:,label_idx]

    # Determine threshold for IHC channel
    if value_threshold is not None:
        log.debug("Dichotomizing IHC mask from [%d]%s channel with supplied threshold: value > %f" % (label_idx,label_channel,value_threshold) )
    else:
        value_threshold = threshold_otsu(mask)
        log.debug("Dichotomizing IHC mask from [%d]%s channel with Otsu's threshold: value > %f" % (label_idx,label_channel,value_threshold) )

    # Apply threshold and generate channel mask for labeling
    label_mask = (mask > value_threshold)
    label_mask = transform.rescale(label_mask,(1/theWM.thumbnail_ratio), anti_aliasing=False, order=0) > 0
    if clean_edges:
        label_mask = closing(label_mask, square(5))
        label_mask = opening(label_mask, square(5))
        label_mask = remove_small_objects(label_mask, noise_size)
    
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
            log.debug("Output directory reset-- Wsi_id: %s, Outdir: %s" % (theWM.wsi_id, theWM.outdir))

        else:
            raise ValueError("Output path does not have directory format.")

    #Apply annotation
    theWM.annotate_from_binmask(label=label,mask=label_mask,threshold=tile_threshold, export_mask=dry_run)
    #TODO: if dry run, export mask with H&E as background

    return theWM

####################

# Log Levels
VERBOSE_LEVEL = [log.ERROR, log.WARNING, log.INFO, log.DEBUG]
ANNOT_TYPES = {"binary":"annotate_from_mask","continuous":"annotate_from_mask","thumbnail":"annotate_from_thumbnail","ihc":"annotate_from_ihc"}#,"fihc":"annotate_from_multiplex_ihc"
ANNOT_TYPE_KEYS = list(ANNOT_TYPES.keys())

if __name__ == '__main__':
    # Define command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default="./", help='WsiManagerData.json file paths or Comma-separated list of input directories. Default: [./]')
    ap.add_argument('-o', '--output', default=None, help="Output directory Default: Input object's directory")
    ap.add_argument('-m', '--mask', default=None, help="Filaname or comma-separated list of directories containing annotations masks. Filenames must match objects' wsi_id. Required")
    ap.add_argument('-a', '--annotation_type', choices=ANNOT_TYPE_KEYS, default=ANNOT_TYPE_KEYS[0], help="Type of annotation masks.Default: [%s]" % ANNOT_TYPE_KEYS[0])
    ap.add_argument('-l', '--label', default=None, help='Name of the feature being annotated. Optional if providing settings.csv file')
    ap.add_argument('-s', '--settings', default=None, help='Numeric threshold value for non-binary masks, color name or hex value for binary mask, or path to settings.csv file. Required')
    ap.add_argument('-e', '--tidy_mask_edges', action='store_true', help='Set this flag clean & smooth edges when using an annotated thumbnail mask. Default: [False]')  
    ap.add_argument('-t', '--tile_threshold', default=0.0, type=float, help='Minimum proportion of positive pixels to annotate a tile as positive. Default: [0.0]' )
    ap.add_argument('-c', '--cores', default=mp.cpu_count(), type=int, help='Numbers of processes to be spun up in parallel to process each WSI. Default: [%d]' % mp.cpu_count() )
    ap.add_argument('-v', '--verbose', action='count', help='Print updates and reports as program executes. Provide the following number of "v" for the following settings: [Default: Error, v: Warning, vv: Info, vvv: Debug]')
    ap.add_argument('-y', '--dry_run', action='store_true', help='Set this flag to output annotation maps without saving annotated object (Good for testing parameters). Default: [False]')  
    args = vars(ap.parse_args())
    
    # args = vars(ap.parse_args(['-i','/home/clemenj/Data/brain/Mouse_PDX/tiled_wsis/SG_01/','-m','/home/clemenj/Data/BLCA_TRRC2819/CCF_Batch2_outputs/pred_tils/SG_01_color.png','-l','predicted_TIL','-s','red','-c','20','-vvvvv'])) #TODO: rm
    # args = vars(ap.parse_args(['-i','/home/clemenj/Data/BLCA_TRRC2819/wsi_sample_tiled/','-m','/home/clemenj/Data/BLCA_TRRC2819/wsi_sample_test_annots','-a','thumbnail','-l','artifacts','-s','black','-c','20','-vvvvv',"-y"])) #TODO: rm
    # args = vars(ap.parse_args(['-i','/home/clemenj/Data/BLCA_TRRC2819/wsi_sample_tiled/','-m','/home/clemenj/Data/BLCA_TRRC2819/wsi_sample_test_annots','-l','tumor','-s','#00ff00','-c','20','-vvvvv'])) #TODO: rm
    # args = vars(ap.parse_args(['-i','/home/clemenj/Data/brain/Mouse_PDX/tiled_wsis/181_he+#3','-m','/home/clemenj/Data/brain/Mouse_PDX/tiled_wsis/181_he+#3/info/181_he+#3___tumor_annot.png','-a','ihc','-l','tumor','-c','20','-vvvvv',"-y"])) #TODO: rm
    

    # Determine Verbosity
    if args['verbose'] is not None:
        if args['verbose'] > len(VERBOSE_LEVEL)-1:
            log.getLogger().setLevel(VERBOSE_LEVEL[-1])
        else:
            log.getLogger().setLevel(VERBOSE_LEVEL[args['verbose']])
    else:
        log.getLogger().setLevel(log.ERROR)

    #Validate output path
    output = None
    if args['output'] is not None:
        output = Path(args['output'])
        if not output.is_dir():
            log.warning("Output directory not found, creating directory.")
            output.mkdir(parents=True)

    # Validate label name input
    if args['label'] is None or args['label'] =="":
        log.error( "No label name was given. Terminating.")
        quit(code=1)

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
    wsi_ids = [aPath.name for aPath in all_wm_paths]
    
    # Notify and terminate if no input found
    if len(all_wm_paths) < 1:
        log.error( "No directories were found from input parameters. Terminating.")
        quit(code=1)
    else:
        log.info("%d WsiManager paths found" % len(all_wm_paths))
        log.debug("WsiManager paths to be processed:\n%s" % "\n".join([str(i) for i in all_wm_paths]) )
  
    # Determine value threshold for dichotomizing mask (if needed)
    val_threshold = isfloat(args['settings'])
    if args['annotation_type'] == "continuous":
        if val_threshold is None:
            log.error("Continuous mask mode was selected, but settings did not supply a float value threshold.")
            quit(code=1)

    # Determine color mask value for binary mask
    lab_color = iscolor(args['settings'])
    if args['annotation_type'] in ["binary","thumbnail"]:
        if lab_color is None:
            log.error("%s mask was selected, but settings did not supply a valid color value." % args["annotation_type"])
            quit(code=1)

    #TODO: determine settings for mIHC using CSV

    # Determine label masks
    if args['mask'] is None:
        log.error( "No masks were found from input parameters. Terminating.")
        quit(code=1)

    # Find masks files
    all_mask_files = [Path(i) for i in args['mask'].split(',') if i.endswith('.jpg') or i.endswith('.png') or Path(i).is_file()]
    all_mask_dirs = [Path(i) for i in args['mask'].split(',') if Path(i).is_dir()]

    # Set up run parameters
    params = pd.DataFrame(list(zip(wsi_ids,all_wm_paths)), columns=['wsi_id','wm_dir'])
    params["label"] = args['label']
    params["label_color"] = lab_color
    params["value_threshold"] = val_threshold
    params['tidy_mask_edges'] = args['tidy_mask_edges']
    params["tile_threshold"] = args['tile_threshold']
    params["dry_run"] = args['dry_run']
    params["outdir"] = output
    params['annotation_function'] = ANNOT_TYPES[args['annotation_type']]
    params['mask_path'] = None

    #Determine masks for each WM to be annotated
    for i,annotRun in params.iterrows():

        #Find first mask file present that matches wsi_id
        matched_masks = [aPath for aPath in all_mask_files if aPath.name.find(annotRun.wsi_id)>=0]
        if len(matched_masks) > 0:
            params.loc[i,'mask_path'] = matched_masks[0]
        else:
            #Find first mask in any of the listed directories
            allPaths = [list(aPathList.glob("*%s*.png" % annotRun.wsi_id)) for aPathList in all_mask_dirs]
            allPaths = allPaths+[list(aPathList.glob("*%s*.jpg" % annotRun.wsi_id)) for aPathList in all_mask_dirs]
            firstPaths = [pathList[0] for pathList in allPaths if len(pathList) > 0]
            if len(firstPaths) > 0:
                params.loc[i,'mask_path'] = firstPaths[0]

    # Preserve only runs with valid masks & validate
    params = params[ ~pd.isnull(params.mask_path) ]
    if len(params) < 1:
        log.error( "No viable annotations were possible with given mask parameters. Terminating.")
        quit(code=1) 

    #Determine core count
    core_cnt = min(len(all_wm_paths), args['cores'])
    log.debug("Processing WM instances using %d cores" % core_cnt)

    # Process annotations in parallel        
    pool = mp.Pool(core_cnt)
    resWM = []
    for i,annotRun in params.iterrows():
        runArgs = {
            "theWMpath": annotRun.wm_dir,
            "label": annotRun.label,
            'maskPath': annotRun.mask_path,
            "label_color": annotRun.label_color,
            "value_threshold": annotRun.value_threshold,
            "clean_edges": annotRun.tidy_mask_edges,
            "tile_threshold": annotRun.tile_threshold,
            "dry_run": annotRun.dry_run,
            "outdir": annotRun.outdir,
            }
        if annotRun.annotation_function == "annotate_from_thumbnail":
            del runArgs['value_threshold']
        if annotRun.annotation_function in ["annotate_from_thumbnail","annotate_from_ihc"]:
            del runArgs['clean_edges']
        aRes = pool.apply_async(func=eval(annotRun.annotation_function),kwds=runArgs)
        resWM.append(aRes)
    pool.close()
    pool.join()
    pool.terminate()

    # If not dry-run export annotated data
    if not args['dry_run']:
        #Get all results
        resWM = [aRes.get() for aRes in resWM]

        # Parallelize export of annotated WM objects
        pool = mp.Pool(core_cnt)
        for i,aWM in enumerate(resWM):
            pool.apply_async(func=exportWM,kwds={"theWM":aWM})
        pool.close()
        pool.join()
        pool.terminate()

    #TODO: Fix error output given always forcing filestructure for dry run 
    #TODO: always export masks
    #TODO: implement tissue coregistration


    