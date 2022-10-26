"""
tiler.py
Commandline interface for dividing a set of whole slide image into tiles and save them as individual files.

Author: Jean R Clemenceau
Date Created: 11/11/2021
"""

import os
import re
import time
import argparse
import logging as log
import multiprocessing as mp

from pathlib import Path

import wsitiler.normalizer as norm
from wsitiler.WsiManager import WsiManager as wm
from wsitiler.wsi_utils import describe_wsi_levels, find_wsi_level

# Log Levels
VERBOSE_LEVEL = [log.ERROR, log.WARNING, log.INFO, log.DEBUG]

if __name__ == '__main__':
    # Define command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default="./", help='Input directory or WSI image. Default: [./]')
    ap.add_argument('-o', '--output', default="./", help='Output directory Default: [./]')
    ap.add_argument('-c', '--cores', default=mp.cpu_count(), type=int, help='Numbers of processes to be spun up in parallel to process each WSI. Default: [%d]' % mp.cpu_count() )
    ap.add_argument('-d', '--tile_dimensions', default=wm.PIXELS_PER_TILE, type=str, help="Defines the tile edge length used when breaking WSIs into smaller images. Provide integer value for pixel size, for micron size use format: XXum. Default: [%s]" % wm.PIXELS_PER_TILE)
    ap.add_argument('-l', '--image_level', default="0", help="Defines the level of magnification to be used for image tiling. None by the following formats: Level (AA), Magnification (AAx), Resolution (A.AAmpp). Default: [0] - Maximum magnification")
    ap.add_argument('-n', '--normalizer', default=norm.NORMALIZER_CHOICES[0], help="Comma-separated list of methods for WSI color normalization. Default: 'None'. Options: [%s]" % ( ", ".join(norm.NORMALIZER_CHOICES) ))
    ap.add_argument('-z', '--noise_size', default=wm.NOISE_SIZE_MICRONS, type=int, help="Defines the maximum size in microns of an item in the binary mask to be considered noise. Default: [%d]" % wm.NOISE_SIZE_MICRONS)
    ap.add_argument('-f', '--foreground_threshold', default=wm.MIN_FOREGROUND_THRESHOLD, type=float, help="Defines the minimum tissue/background ratio for a tile to be considered foreground. Default: [%f]" % wm.MIN_FOREGROUND_THRESHOLD)
    ap.add_argument('-e', '--tile_export_format', default=wm.SUPPORTED_TILE_EXPORT_FORMATS[0], choices=wm.SUPPORTED_TILE_EXPORT_FORMATS, help="Select the format used to export tile files. Default: '%s'. Options: [%s]" % ( wm.SUPPORTED_TILE_EXPORT_FORMATS[0],", ".join(wm.SUPPORTED_TILE_EXPORT_FORMATS) ))
    ap.add_argument('-r', '--normalizer_reference', default="None", type=str, help='H & E image used as a reference for normalization. Default: [wsitiler/normalizer/macenko_reference_img.png]')
    ap.add_argument('-v', '--verbose', action='count', help='Print updates and reports as program executes. Provide the following number of "v" for the following settings: [Default: Error, v: Warning, vv: Info, vvv: Debug]')
    ap.add_argument('-t', '--tissue_chunk_id', action='store_true', help='Set this flag to determine tissue chunk ids for each tile: Default: [False]')
    ap.add_argument('-p', '--image_parameters', action='store_true', help='Set this flag to Display the image parameters for the first WSI found. Default: [False]')  
    ap.add_argument('-y', '--dry_run', action='store_true', help='Set this flag to run tile prep and heamaps without tiling the image (Good for testing parameters). Default: [False]')  
    args = vars(ap.parse_args())

    # Validate arguments
    if args['verbose'] is not None:
        if args['verbose'] > len(VERBOSE_LEVEL)-1:
            log.getLogger().setLevel(VERBOSE_LEVEL[-1])
        else:
            log.getLogger().setLevel(VERBOSE_LEVEL[args['verbose']])
    else:
        log.getLogger().setLevel(log.ERROR)

    if args['foreground_threshold'] < wm.MIN_FOREGROUND_THRESHOLD:
        log.warning("Tiles with very little foreground may fail color normalization.")

    if args['normalizer_reference'] != "None" and not os.path.exists(args['normalizer_reference']):
        raise ValueError("Normalizer reference image provided does not exist")
    elif args['normalizer_reference'] == "None":
        args['normalizer_reference'] = None

    #Validate normalization methods
    if args['normalizer'] == "None" or args['normalizer'] == "":
        norm_methods = [norm.Normalizer().method]
    else:
        norm_methods = re.split(', ?', args['normalizer'])
        for i in norm_methods:
            if i not in norm.NORMALIZER_CHOICES and i:
                raise ValueError("%i is NOT a valid normalization method.")

    log.info("Starting tiling run")

    total_start_time = time.time()
    log.debug("Run Arguments: %s" % args)

    # Determine input image paths
    all_wsi_paths = []
    if os.path.isdir(args["input"]):
        input_path = Path(args["input"])
        for ftype in wm.SUPPORTED_WSI_FORMATS:
            all_wsi_paths.extend(input_path.rglob("*"+ftype))
    else:
        if args["input"].endswith(tuple(wm.SUPPORTED_WSI_FORMATS)):
            if os.path.exists(args["input"]):
                all_wsi_paths.append(Path(args["input"]))

    # Notify and terminate if no input found
    if len(all_wsi_paths) < 1:
        log.error( "No WSIs were found from input parameters. Terminating.")
        quit(code=1)

    log.info("Found WSI images. Starting Processing")
    log.debug("The following WSIs were found:")
    for i,aPath in enumerate([str(s) for s in all_wsi_paths]):
        log.debug("%d:\t%s" % (i+1,aPath) )

    # If requested, print image parameters and quit
    if args['image_parameters']:
        print("%d WSIs were found from input parameters.\nThe following are example parameters from <<%s>>:" 
            % (len(all_wsi_paths), all_wsi_paths[0]))
        print(describe_wsi_levels(all_wsi_paths[0]))
        quit(code=0)
            
    #Prepare output path
    outpath = Path(args["output"])
    if not outpath.is_dir():
        outpath.mkdir(parents=True)

    # Process wsi images
    for i,wsi in enumerate(all_wsi_paths):
        log.info("%d - Processing %s" % (i+1, wsi.stem ))
        wsi_start_time = time.time()

        # Prepare tiling reference
        img_level = find_wsi_level(wsi,args['image_level'])

        #Prepare tiling data
        aManager = wm(
            wsi_src = wsi,
            wsi_id = wsi.stem, 
            outdir = outpath, 
            mpt = args["tile_dimensions"], 
            wsi_level = img_level, 
            min_tissue = args['foreground_threshold'], 
            noise_size = args['noise_size'], 
            segment_tissue = args['tissue_chunk_id'], 
            normalization = norm_methods
        )

        # Export metadata
        aManager.export_info()

        if args['dry_run']:
            log.debug("%d - Dry Run Time: %f" % (i+1, time.time()-wsi_start_time) )
            continue

        # Export tiles
        aManager.export_tiles_multiprocess(
            filetype = args['tile_export_format'],
            ref_img = args['normalizer_reference'],
            cores = args['cores']
        )

        log.info("%d - Total Processing Time: %f" % (i+1, time.time()-wsi_start_time) )

    log.info("Finished Processing All WSIs" )
    log.info("Total Time: %f" % (time.time()-total_start_time) )
  