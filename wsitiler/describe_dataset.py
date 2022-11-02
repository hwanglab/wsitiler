"""
describe_dataset.py
Commandline interface for describing a dataset of exported preprocessed WSImanager objects.

Author: Jean R Clemenceau
Date Created: 10/31/2022
"""

import argparse
import numpy as np
import pandas as pd
import logging as log
import multiprocessing as mp

from pathlib import Path
from skimage import measure
from skimage.filters import threshold_otsu

# import wsitiler.normalizer as norm
from wsitiler.WsiManager import WsiManager as wm

# Log Levels
VERBOSE_LEVEL = [log.ERROR, log.WARNING, log.INFO, log.DEBUG]

def describe_a_wm(wmDir: Path=None):
    info_dir = {}

    theWM = wm.fromdir(wmDir)

    info_dir['wsi_id'] = theWM.wsi_id
    info_dir['wsi_src'] = str(theWM.wsi_src)
    info_dir['mpp'] = theWM.wsi_mpp
    info_dir['normalizations'] = ";".join(theWM.normalization)

    # Determine available annotations
    allMasks = [aMask for aMask in theWM.__dict__.keys() if aMask.endswith("_mask")]
    annots = [aMask[0:-5] for aMask in allMasks if aMask != 'tissue_mask' and aMask != 'tissue_chunk_mask']
    info_dir['annotations'] = ";".join(annots)

    info_dir['tilesize_px'] = theWM.wsi_ppt_x
    info_dir['tilesize_um'] = theWM.wsi_ppt_x * theWM.wsi_mpp

    # Calculate Foreground
    fg_px = len(theWM.tissue_mask[theWM.tissue_mask])
    info_dir['tissue_fg_px'] = fg_px #TODO: report un um2 instead
    info_dir['tissue_fg_perc'] = fg_px / theWM.tissue_mask.size
    info_dir['tissue_fg_tiles'] = len(theWM.tile_data[ ~pd.isnull(theWM.tile_data.tilename) ])

    if "tissue_chunk_mask" in allMasks:
        # Get tissue chunk properties
        chunk_table = pd.DataFrame(measure.regionprops_table(theWM.tissue_chunk_mask,
            properties=('label','area','centroid','eccentricity',
                'equivalent_diameter','major_axis_length','minor_axis_length')
        ))
        large_chunk_threshold = threshold_otsu(chunk_table.area)
        large_chunk_table = chunk_table[chunk_table.area > large_chunk_threshold]
        max_chunk = chunk_table[chunk_table.area == max(chunk_table.area)]

        info_dir['tissue_chunk_cnt'] = len(chunk_table)
        info_dir['tissue_chunk_size_mean'] = np.mean(chunk_table.area)
        info_dir['tissue_chunk_size_stdev'] = np.std(chunk_table.area)
        info_dir['large_tissue_chunk_cnt'] = len(large_chunk_table)
        info_dir['large_tissue_chunk_ids'] = ";".join([str(i) for i in large_chunk_table.label])
        info_dir['large_tissue_chunk_size_mean'] = np.mean(large_chunk_table.area)
        info_dir['large_tissue_chunk_size_stdev'] = np.std(large_chunk_table.area)
        info_dir['max_chunk_label'] = max_chunk.label[0]
        info_dir['max_chunk_area'] = max_chunk.area[0]
        info_dir['max_chunk_eccentricity'] = max_chunk.eccentricity[0]
        info_dir['max_chunk_major_axis_length'] = max_chunk.major_axis_length[0]
        info_dir['max_chunk_minor_axis_length'] = max_chunk.minor_axis_length[0]
        

    return info_dir


if __name__ == '__main__':
    # Define command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default="./", help='Comma-separated List of input directories. Default: [./]')
    ap.add_argument('-o', '--output', default="./", help='Output directory Default: [./]')
    ap.add_argument('-c', '--cores', default=mp.cpu_count(), type=int, help='Numbers of processes to be spun up in parallel to process each WSI. Default: [%d]' % mp.cpu_count() )
    ap.add_argument('-v', '--verbose', action='count', help='Print updates and reports as program executes. Provide the following number of "v" for the following settings: [Default: Error, v: Warning, vv: Info, vvv: Debug]')
    # ap.add_argument('-y', '--dry_run', action='store_true', help='Set this flag to run tile prep and heamaps without tiling the image (Good for testing parameters). Default: [False]')  
    # ap.add_argument('-p', '--plots', action='store_true', help='Set this flag to export plots describing the data. Default: [False]')  #TODO: implement
    args = vars(ap.parse_args())

    # Determine Verbosity
    if args['verbose'] is not None:
        if args['verbose'] > len(VERBOSE_LEVEL)-1:
            log.getLogger().setLevel(VERBOSE_LEVEL[-1])
        else:
            log.getLogger().setLevel(VERBOSE_LEVEL[args['verbose']])
    else:
        log.getLogger().setLevel(log.ERROR)

    #Validate output path
    output = Path(args['output'])
    if not output.is_dir():
        log.warning("Output directory not found, creating directory.")
        output.mkdir(parents=True)

    # Determine input paths
    all_input_paths = [Path(i) for i in args['input'].split(',') if Path(i).is_dir()]
    
    #Find all exported wm json files
    all_wm_json_paths = []
    for aPath in all_input_paths:
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

    #Determine core count
    core_cnt = min(len(all_wm_paths), args['cores'])
    log.debug("Processing WM instances using %d cores" % core_cnt)

    # Process tiles in parallel        
    pool = mp.Pool(core_cnt)
    res = []
    for aWMPath in all_wm_paths:
        aRes = pool.apply_async(func=describe_a_wm,kwds={'wmDir':aWMPath })
        res.append(aRes)
    pool.close()
    pool.join()
    pool.terminate()

    #Get all results
    res = [aRes.get() for aRes in res]

    #Make dataframe
    wm_description = pd.DataFrame(data=res, columns=['wsi_id',
        'wsi_src','mpp','normalizations','annotations',
        'tilesize_px','tilesize_um',
        'tissue_fg_px','tissue_fg_perc','tissue_fg_tiles',
        'tissue_chunk_cnt',
        'tissue_chunk_size_mean','tissue_chunk_size_stdev',
        'large_tissue_chunk_cnt','large_tissue_chunk_ids',
        'large_tissue_chunk_size_mean','large_tissue_chunk_size_stdev',
        'max_chunk_label','max_chunk_area','max_chunk_eccentricity',
        'max_chunk_major_axis_length','max_chunk_minor_axis_length']
    )

    #Export tsv fule
    filename = output / "wm_data_description.tsv"
    wm_description.to_csv(filename, sep="\t", line_terminator="\n", index=False)
    log.debug("Exported data description to: %s" % filename)