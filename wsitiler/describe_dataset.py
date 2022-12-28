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
    """
    Exports metadata from a about tissue chunks in a given WsiMaanager object.

    Input:
        wmDir (Path): File path to directory containing a WsiManager's files.
    Output:
        Creates output pandas dataframe with description.
    """
    info_dir = {}

    theWM = wm.fromdir(wmDir)

    info_dir['wsi_id'] = theWM.wsi_id
    info_dir['parent_id'] = theWM.parent_id if hasattr(theWM,"parent_id") else ""
    info_dir['wsi_src'] = str(theWM.wsi_src)
    info_dir['mpp'] = theWM.wsi_mpp
    info_dir['normalizations'] = ";".join(theWM.normalization)

    # Determine available annotations
    allMasks = [aMask for aMask in theWM.__dict__.keys() if aMask.endswith("_mask")]
    annots = [aMask[0:-5] for aMask in allMasks if aMask != 'tissue_mask' and aMask != 'tissue_chunk_mask']
    info_dir['annotations'] = ";".join(annots)

    info_dir['tilesize_px'] = theWM.wsi_ppt_x
    info_dir['tilesize_um'] = theWM.wsi_ppt_x * theWM.wsi_mpp

    # Determine tile export formats available
    tile_export_format = []
    if len(list((theWM.outdir / theWM.wsi_id / "tiles").glob("*/*.png"))) > 0:
        tile_export_format.append("png")
    if len(list((theWM.outdir / theWM.wsi_id / "tiles").glob("*/*.npy"))) > 0:
        tile_export_format.append("npy")
    info_dir['tile_export_format'] = ",".join(tile_export_format)

    #TODO: export size in um^2

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

        # Find large and largest chunks
        large_chunk_threshold = threshold_otsu(chunk_table.area)
        large_chunk_table = chunk_table[chunk_table.area > large_chunk_threshold]
        large_chunk_table = large_chunk_table.sort_values(by=['area'],ascending=False)
        max_chunk = large_chunk_table.iloc[0]

        # Save relevant values
        info_dir['tissue_chunk_cnt'] = len(chunk_table)
        info_dir['tissue_chunk_size_mean'] = np.mean(chunk_table.area)
        info_dir['tissue_chunk_size_stdev'] = np.std(chunk_table.area)
        info_dir['large_tissue_chunk_cnt'] = len(large_chunk_table)
        info_dir['large_tissue_chunk_ids'] = ";".join([str(i) for i in large_chunk_table.label])
        info_dir['large_tissue_chunk_size_mean'] = np.mean(large_chunk_table.area)
        info_dir['large_tissue_chunk_size_stdev'] = np.std(large_chunk_table.area)
        info_dir['max_chunk_label'] = max_chunk['label']
        info_dir['max_chunk_area'] = max_chunk['area']
        info_dir['max_chunk_eccentricity'] = max_chunk['eccentricity']
        info_dir['max_chunk_major_axis_length'] = max_chunk['major_axis_length']
        info_dir['max_chunk_minor_axis_length'] = max_chunk['minor_axis_length']
        

    return info_dir


if __name__ == '__main__':
    # Define command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default="./", help='Comma-separated list of input directories. Default: [./]')
    ap.add_argument('-o', '--output', default="./", help='Output directory Default: [./]')
    ap.add_argument('-c', '--cores', default=mp.cpu_count(), type=int, help='Numbers of processes to be spun up in parallel to process each WSI. Default: [%d]' % mp.cpu_count() )
    ap.add_argument('-v', '--verbose', action='count', help='Print updates and reports as program executes. Provide the following number of "v" for the following settings: [Default: Error, v: Warning, vv: Info, vvv: Debug]')
    # ap.add_argument('-s', '--split', default="no", choices=["no", "large", "largest"], help='Split tissue chunks into its own WM object depending on their size. Options: [no, large, largest]. Default: [no]')
    # ap.add_argument('-y', '--dry_run', action='store_true', help='Set this flag to run tile prep and heamaps without tiling the image (Good for testing parameters). Default: [False]')  
    args = vars(ap.parse_args())
    
    # args = vars(ap.parse_args(['-i','/home/clemenj/Data/brain/Mouse_PDX/tiled_wsis/','-o','/home/clemenj/Data/brain/Mouse_PDX/tiled_wsis/','-c','70','-vvvvv'])) #TODO: rm
    

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
    wm_description = pd.DataFrame(data=res, columns=['wsi_id',"parent_id",
        'wsi_src','mpp','normalizations','annotations','tile_export_format',
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

#################

    # # Prepare to split WM by tissue
    # if args['split'] == "largest":
    #     all_splits = pd.DataFrame({"wmPath": all_wm_paths, "labels": wm_description.max_chunk_label, "tile_export_format":wm_description.tile_export_format})
    # elif args['split'] == "large":
    #     all_splits = pd.DataFrame({"wmPath": all_wm_paths, "labels": [i.split(';') for i in wm_description.large_tissue_chunk_ids], "tile_export_format":wm_description.tile_export_format })
    #     all_splits = all_splits.explode("labels")
    # else:
    #     all_splits = pd.DataFrame()

    # if len(all_splits) > 0:
    #     #Determine core count
    #     core_cnt = min(len(all_splits), args['cores'])
    #     cores_per_split = int( args['cores'] / core_cnt)

    #     # Process splits in parallel        
    #     pool = mp.Pool(core_cnt)
    #     for index, aSplit in all_splits.iterrows():
    #         pool.apply_async(func=process_split,kwds={
    #             'wmDir': aSplit['WMPath'],
    #             'label': aSplit['labels'],
    #             'tile_formats': aSplit['tile_export_format'],
    #             'cores': cores_per_split
    #             })
    #     pool.close()
    #     pool.join()
    #     pool.terminate()
