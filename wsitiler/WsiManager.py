"""
wsiManager.py
Implement WsiManager, the core class for the wsitiler framework. 
Contains functions for dividing a whole slide image into tiles and
save them as individual files.

Author: Jean R Clemenceau
Date Created: 18/09/2022
"""

import openslide
import argparse
import os
import math
import time
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import traceback

from PIL import Image
from pathlib import Path
from math import ceil, floor
from skimage import transform
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, opening,closing, square

from wsitiler.wsi_utils import describe_wsi_levels
# import wsitiler.normalizer as norm

# PIXELS_PER_TILE defines the default tile edge length m used when breaking WSIs into smaller images (m x m)
PIXELS_PER_TILE = "512px"
# NOISE_SIZE_MICRONS defines default maximum size (in microns) for an artifact to be considered as noise in the WSI tissue mask.
NOISE_SIZE_MICRONS = 256
# MIN_FOREGROUND_THRESHOLD defines default minimum tissue/background ratio to classify a tile as foreground.
MIN_FOREGROUND_THRESHOLD = 0.10

class WsiManager:
    ''

    # SUPPORTED_WSI_FORMATS defines the WSI formats supported by Openslide.
    SUPPORTED_WSI_FORMATS = [".svs",".ndpi",".vms",".vmu",".scn",".mrxs",".tiff",".svslide",".tif",".bif"]

    #Constructor with Openslide object
    def __init__(self,wsi_src: Path=None, wsi: openslide.OpenSlide=None, wsi_id: str=None, outdir: Path=None, mpt: str=PIXELS_PER_TILE, wsi_level: int=0, min_tissue: float=MIN_FOREGROUND_THRESHOLD, noise_size: int=NOISE_SIZE_MICRONS, segment_tissue: bool=False):
        """
        WsiManager constructor based on an OpenSlide object and additional parameters.

        Input:
            wsi_src (Path): File path to location of original WSI file. Optional if OpenSlide object is supplied.
            wsi (OpenSlide): OpenSlide object containing WSI to be processed. Optional if wsi_src is supplied.
            wsi_id (str): unique name assigned to the WSI. Optional if wsi_src is supplied.
            outdir (Path): File path to directory to contain collection of WsiManager output files. Default: None
            mpt (str): Desired width and height of processed tiles in pixels (use 'px' as suffix) or microns (use 'um' as suffix). Default: [%s].
            wsi_level (int): Image level to be tiled. Default: 0.
            min_tissue (float): Minimum foreground tissue ratio for marking and saving a tile. Default: [%d]
            noise_size (int): Maximum size (in microns) of forground regions to be considered noise & be removed. Default: [%d]
            segment_tissue (bool): Generate tissue chunk segmentation mask and assign labels to corresponding tiles. Default: [False]

        Output:
            new WSIManager object
        """ % (PIXELS_PER_TILE,MIN_FOREGROUND_THRESHOLD,NOISE_SIZE_MICRONS)

        # Validate and set metadata parameters
        self.img_lvl = wsi_level

        #Validate OpenSlide WSI object
        if wsi is not None and not isinstance(wsi,openslide.OpenSlide):
            raise ValueError("WSI object suplied is not an OpenSlide object")
        
        #Validate WSI source
        if wsi_src is None and wsi is None:
            raise ValueError("No WSI source object or filepath were given.")
        elif wsi_src is None:
            self.wsi_src = None
        else:
            if isinstance(wsi_src,str):
                wsi_src = Path(wsi_src)
            elif not isinstance(wsi_src,Path):
                raise ValueError("WSI source filepath is not a Path or a string.")
        
            if wsi_src.is_file():
                if wsi_src.suffix in WsiManager.SUPPORTED_WSI_FORMATS:
                    self.wsi_src = wsi_src
                    if wsi is None:
                        wsi = openslide.OpenSlide(str(wsi_src))
                else:
                    raise ValueError("File type of given WSI source is not supported.")
            else:
                raise ValueError("WSI source filepath is not a file.")

        #Validate WSI ID
        if wsi_id is None and wsi_src is None:
            ValueError("WSI ID not supplied.")
        elif wsi_id is None:
            self.wsi_id = wsi_src.stem.replace(" ","_")
        else:
            self.wsi_id = str(wsi_id)

        #Validate output directory
        if outdir is not None:
            if isinstance(outdir,str):
                outdir = Path(outdir)
            elif not isinstance(outdir,Path):
                raise ValueError("Output path is not a Path or a string.")

            #if output directory has correct format, save it.
            if outdir.suffix == "":
                self.outdir = outdir
            else:
                raise ValueError("Output path does not have directory format.")
        else:
            self.outdir = None

        # Get WSI details and find level
        wsi_params = describe_wsi_levels(wsi)

        wsi_width = wsi_params["Width"].iloc[self.img_lvl]
        wsi_height = wsi_params["Height"].iloc[self.img_lvl]

        # Calculate microns per pixel at desired level
        resolution = wsi_params["Resolution"].iloc[self.img_lvl]
        self.wsi_mpp = float(resolution.split("mpp")[0])

        # Calculate desired tile dimensions (pixels per tile)
        if mpt.endswith("um"):
            microns = float(mpt.split("um")[0])
            pixels = round(microns/self.wsi_mpp)
        else:
            try:
                pixels = int(mpt.split("px")[0])
            except ValueError as e:
                raise ValueError("Tile mpt length format is not valid. Provide pixel count using 'XXpx' or microns using 'XXum'")

        self.wsi_ppt_x = self.wsi_ppt_y = pixels

        # Get thumbnail for tissue mask
        thumbnail_og = wsi.get_thumbnail(size=(wsi.level_dimensions[-1][0], wsi.level_dimensions[-1][1]))
        self.thumbnail = np.array(thumbnail_og)
        thumbnail_gray = (rgb2gray(self.thumbnail) * 255).astype(np.uint8)

        #Save micros per pixel in thumbnail:
        self.thumbnail_mpp = float(wsi_params["Resolution"].iloc[-1].split("mpp")[0])

        # calculate mask parameters
        self.thumbnail_ratio = wsi_width / thumbnail_gray.shape[1]  # wsi dimensions: (x,y); thumbnail dimensions: (rows,cols)
        thumbnail_mpp = self.wsi_mpp * self.thumbnail_ratio
        self.thumbnail_ppt_x = ceil(self.wsi_ppt_x / self.thumbnail_ratio)
        self.thumbnail_ppt_y = ceil(self.wsi_ppt_y / self.thumbnail_ratio)

        noise_size_pix = round(noise_size / thumbnail_mpp)

        # Create and clean tissue mask
        self.tissue_mask = (thumbnail_gray[:, :] < threshold_otsu(thumbnail_gray))
        self.tissue_mask = remove_small_objects(self.tissue_mask, noise_size_pix)
        self.tissue_mask = closing(self.tissue_mask, square(5))
        self.tissue_mask = opening(self.tissue_mask, square(5))

        # # Remove holes in tissue smaller than a tile
        # tissue_mask = np.invert(tissue_mask)
        # tissue_mask = remove_small_objects(tissue_mask, tile_area)
        # tissue_mask = np.invert(tissue_mask)

        # Calculate margin according to ppt sizes
        wsi_x_tile_excess = wsi_width % self.wsi_ppt_x
        wsi_y_tile_excess = wsi_height % self.wsi_ppt_y

        # Determine WSI tile coordinates
        wsi_tiles_x = list(range(ceil(wsi_x_tile_excess / 2), wsi_width - floor(wsi_x_tile_excess / 2), self.wsi_ppt_x))
        wsi_tiles_y = list(range(ceil(wsi_y_tile_excess / 2), wsi_height - floor(wsi_y_tile_excess / 2), self.wsi_ppt_y))

        # Approximate mask tile coordinates
        mask_tiles_x = [floor(i / self.thumbnail_ratio) for i in wsi_tiles_x]
        mask_tiles_y = [floor(i / self.thumbnail_ratio) for i in wsi_tiles_y]

        #Trim thumbnail & mask
        self.thumbnail = self.thumbnail[mask_tiles_y[0]:mask_tiles_y[-1] + self.thumbnail_ppt_y,
                        mask_tiles_x[0]:mask_tiles_x[-1] + self.thumbnail_ppt_x]
        self.tissue_mask = self.tissue_mask[mask_tiles_y[0]:mask_tiles_y[-1] + self.thumbnail_ppt_y,
                        mask_tiles_x[0]:mask_tiles_x[-1] + self.thumbnail_ppt_x]

        # Generatetissue segmentation mask if required
        if segment_tissue:
            # Get labels for all chunks
            self.tissue_chunk_mask = ndi.label(self.tissue_mask)[0]

            # Filter out chunks smaller than tile area
            tile_area = self.thumbnail_ppt_x*self.thumbnail_ppt_y
            (chunk_label, chunk_size) = np.unique(self.tissue_chunk_mask,return_counts=True)
            filtered_chunks = chunk_label[ chunk_size < tile_area ]
            bg_label = np.unique(self.tissue_chunk_mask[self.tissue_mask == 0])[0]
            for l in filtered_chunks:
                self.tissue_chunk_mask[self.tissue_chunk_mask == l] = bg_label

        # Populate tile reference table###########
        rowlist = []
        for x in range(len(wsi_tiles_x)):
            for y in range(len(wsi_tiles_y)):
                # Get np.array subset of image (a tile)
                aTile = self.tissue_mask[mask_tiles_y[y]:mask_tiles_y[y] + self.thumbnail_ppt_y,
                        mask_tiles_x[x]:mask_tiles_x[x] + self.thumbnail_ppt_x]

                # Calculate tissue ratio for tile
                tissue_ratio = np.sum(aTile) / aTile.size

                aTile_id = len(rowlist)

                new_row = {"image_id": self.wsi_id,
                        "tile_id": aTile_id,
                        "index_x": x,
                        "index_y": y,
                        "wsi_x": wsi_tiles_x[x],
                        "wsi_y": wsi_tiles_y[y],
                        "mask_x": mask_tiles_x[x],
                        "mask_y": mask_tiles_y[y],
                        "tilename": self.wsi_id + "__tile-n-%d_x-%d_y-%d" % (aTile_id, x, y),
                        "tissue_ratio": tissue_ratio
                        }
                
                # # Determine chunk id by most prevalent ID
                if segment_tissue:
                    # Get chunk labels for pixels in tile
                    chunk_tile = self.tissue_chunk_mask[
                        mask_tiles_y[y]:mask_tiles_y[y] + self.thumbnail_ppt_y,
                        mask_tiles_x[x]:mask_tiles_x[x] + self.thumbnail_ppt_x]
                    
                    # Determine chunk id by most prevalent non-background ID
                    flat_chunk_tile = chunk_tile.flatten()
                    flat_chunk_tile = flat_chunk_tile[flat_chunk_tile != bg_label]
                    if(len(flat_chunk_tile) > 1):
                        chunk_id = np.bincount(flat_chunk_tile).argmax()
                    else:
                        chunk_id = bg_label
                    
                    new_row['chunk_id'] = chunk_id

                rowlist.append(new_row)

        # Create reference dataframe
        colnames = ["image_id","tile_id", "index_x", "index_y", "wsi_x", "wsi_y", "mask_x", "mask_y", "tilename", "tissue_ratio"]
        if segment_tissue:
                    colnames.append('chunk_id')
        self.tile_data = pd.DataFrame(data=rowlist, columns=colnames)

        # Remove filenames for empty tiles
        self.tile_data.loc[self.tile_data['tissue_ratio'] < min_tissue, "tilename"] = None

        return(None)

    def segment_tissue(self):
        """
        Segments tissue mask from the given WsiManager object to find individual tissue regions. Assigns labels to tiles and generates labeled mask.
        """

        # Get labels for all chunks
        self.tissue_chunk_mask = ndi.label(self.tissue_mask)[0]

        # Filter out chunks smaller than tile area
        tile_area = self.thumbnail_ppt_x*self.thumbnail_ppt_y
        (chunk_label, chunk_size) = np.unique(self.tissue_chunk_mask,return_counts=True)
        filtered_chunks = chunk_label[ chunk_size < tile_area ]
        bg_label = np.unique(self.tissue_chunk_mask[self.tissue_mask == 0])[0]
        new_bg_label = -1
        self.tissue_chunk_mask[self.tissue_chunk_mask==bg_label] = new_bg_label

        for l in filtered_chunks:
            self.tissue_chunk_mask[self.tissue_chunk_mask == l] = new_bg_label

        # Save chunk_id data to tiles
        self.tile_data['chunk_id'] = new_bg_label

        #Iterate over non-background tiles
        for i,row in self.tile_data[ ~pd.isnull(self.tile_data.tilename) ].iterrows():
            
            # Get chunk labels for pixels in tile
            chunk_tile = self.tissue_chunk_mask[row['mask_y']:row['mask_y'] + self.thumbnail_ppt_y,
                row['mask_x']:row['mask_x'] + self.thumbnail_ppt_x]
            
            # Determine chunk id by most prevalent non-background ID
            flat_chunk_tile = chunk_tile.flatten()
            flat_chunk_tile = flat_chunk_tile[flat_chunk_tile >= 0]
            if(len(flat_chunk_tile) > 1):
                chunk_id = np.bincount(flat_chunk_tile).argmax()
            else:
                chunk_id = new_bg_label

            # save value
            self.tile_data.at[i,'chunk_id'] = chunk_id

        return(None)


    def export_info(self, outdir: Path=None):
        """
        Exports metadata from a WsiManager object to a given directory.

        Input:
            outdir (Path): File path to directory to contain collection of WsiManager output directories. Optional if WsiManager outdir field has been set.
        Output:
            Creates output directory '<outdir>/<wsi_id>/info/'. Generates JSON file for metadata, TSV file for tile data, and exports thumbnail and masks as PNGs.
        """

        #Validate output directory
        if outdir is None and self.outdir is None:
            raise ValueError("Output path has not been given.")
        
        #Validate and set output to given value
        if outdir is not None:
            if isinstance(outdir,str):
                outdir = Path(outdir)
            elif not isinstance(outdir,Path):
                raise ValueError("Output path is not a Path or a string.")

            #if output directory has correct format, save it.
            if outdir.suffix == "":
                final_outdir = outdir / self.wsi_id / "info"
                if self.outdir is None:
                    self.outdir = outdir
            else:
                raise ValueError("Output path does not have directory format.")
        else:
            final_outdir = self.outdir / self.wsi_id / "info"

        # Ensure output directory exists
        if not final_outdir.is_dir():
            final_outdir.mkdir(parents=True)

        # Save tile data as tsv file
        filename_tiledf = self.wsi_id + "___reference_wsi-tilesize_x-%d-y-%d_mask-tilesize_x-%d-y-%d_img-level_%d.tsv" % \
                        (self.wsi_ppt_x, self.wsi_ppt_y,self.thumbnail_ppt_x, self.thumbnail_ppt_y, self.img_lvl)
        self.tile_data.to_csv(final_outdir / filename_tiledf, sep="\t", line_terminator="\n", index=False)

        # Export thumbnail masks/images #TODO finish 
        thumbnail_path = self.export_thumbnail(outdir= final_outdir, export=True)
        tissue_mask_path = self.export_bin_mask(outdir= final_outdir, export=True)
        tissue_chunk_mask_path = self.export_chunk_mask(outdir= final_outdir, export=True)
        thumbnail_tiles_path = self.export_thumbnail(outdir= final_outdir, export=True, showTiles=True)

        # Save object parameters as yml
        instance_dict = self.__dict__.copy()
        if 'wsi_src' in instance_dict.keys():
            instance_dict['wsi_src'] = str(self.wsi_src)
        if 'outdir' in instance_dict.keys():
            instance_dict['outdir'] = str(self.outdir)

        #todo: fix serialization
        if 'thumbnail' in instance_dict.keys():
                    del(instance_dict['outdir'])
                    # instance_dict['outdir'] = self.thumbnail.tolist()
        if 'tissue_mask' in instance_dict.keys():
                    del(instance_dict['tissue_mask'])
                    # instance_dict['tissue_mask'] = self.tissue_mask.tolist()
        if 'tissue_chunk_mask' in instance_dict.keys():
                    del(instance_dict['tissue_chunk_mask'])
                    # instance_dict['tissue_chunk_mask'] = self.tissue_chunk_mask.tolist()

        instance_dict['thumbnail_path'] = str(thumbnail_path)
        instance_dict['tissue_mask_path'] = str(tissue_mask_path)
        instance_dict['tissue_chunk_mask_path'] = str(tissue_chunk_mask_path)
        instance_dict['tile_data_path'] = str(final_outdir / filename_tiledf)
        
        json_path = final_outdir / ("%s___WsiManagerData.json" % self.wsi_id)
        with open(json_path, "w") as outfile:
            json.dump(instance_dict, outfile)
        
        return(None)

    def export_chunk_mask(self, outdir: Path=None, show: bool=False, export: bool=True):
        """
        Exports and/or displays the tissue_chunk_mask as an image from a WsiManager object, if available.

        Input:
            outdir (Path): File path to directory to contain collection of WsiManager output directories. Optional if WsiManager outdir field has been set.
            show (bool): Wether or not to display the mask as a plot. Default: [False]
            export (bool): Wether or not to export the mask as a PNG file. Default: [True]
        Output:
            If 'export' is enabled, returns filepath to image, 'None' otherwise.
            Note: Places image in directory: '<outdir>/<wsi_id>/info/' unless a 'outdir' is given.
        """
        finalPath = None
        if hasattr(self,"tissue_chunk_mask"):

            if export:
                #Find and validate output directory
                if outdir is None and self.outdir is None:
                    raise ValueError("No output directory has been supplied")
                elif outdir is not None:
                    if isinstance(outdir,str):
                        outdir = Path(outdir)
                    elif not isinstance(outdir,Path):
                        raise ValueError("Output path is not a Path or a string.")

                    #if output directory has correct format, use it directly.
                    if outdir.suffix == "":
                        final_outdir = outdir
                    else:
                        raise ValueError("Output path does not have directory format.")
                else:
                    final_outdir = self.outdir / self.wsi_id / "info"

                filename_chunkmask = self.wsi_id + "___chunk-mask_tilesize_x-%d-y-%d.png" % (
                    self.thumbnail_ppt_x, self.thumbnail_ppt_y)
                
                finalPath = final_outdir / filename_chunkmask

            #Setup colormap for black background
            a_cmap = cm.get_cmap("viridis").copy()
            a_cmap.set_under(color='black')

            #TODO: Add labels option

            # Show and/or export plot
            plt.figure()
            plt.axis('off')
            plt.margins(0, 0)
            if show:
                plt.imshow(self.tissue_chunk_mask, cmap=a_cmap, interpolation='nearest', vmin=0)
            if export:
                plt.imsave(finalPath, self.tissue_chunk_mask, cmap=a_cmap, vmin=0)

        return(finalPath)

    def export_thumbnail(self, outdir: Path=None, show: bool=False, export: bool=True, showTiles: bool=False):
        """
        Exports and/or displays the WSI thumbnail as an image from a WsiManager object, if available.

        Input:
            outdir (Path): File path to directory to contain collection of WsiManager output directories. Optional if WsiManager outdir field has been set.
            show (bool): Wether or not to display the thumbnail as a plot. Default: [False]
            export (bool): Wether or not to export the thumbnail as a PNG file. Default: [True]
            showTiles (bool): Wether or not to export the thumbnail with tile annotations. Default: [False]
        Output:
            If 'export' is enabled, returns filepath to image, 'None' otherwise.
            Note: Places image in directory: '<outdir>/<wsi_id>/info/' unless a 'outdir' is given.
        """
        finalPath = None
        if hasattr(self,"thumbnail"):

            if export:
                #Find and validate output directory
                if outdir is None and self.outdir is None:
                    raise ValueError("No output directory has been supplied")
                elif outdir is not None:
                    if isinstance(outdir,str):
                        outdir = Path(outdir)
                    elif not isinstance(outdir,Path):
                        raise ValueError("Output path is not a Path or a string.")

                    #if output directory has correct format, use it directly.
                    if outdir.suffix == "":
                        final_outdir = outdir
                    else:
                        raise ValueError("Output path does not have directory format.")
                else:
                    final_outdir = self.outdir / self.wsi_id / "info"

                isTiledStr = "-tiles" if showTiles else ""
                filename_thumbnail = self.wsi_id + "___thumbnail%s_tilesize_x-%d-y-%d.png" % (
                    isTiledStr, self.thumbnail_ppt_x, self.thumbnail_ppt_y)
                
                finalPath = final_outdir / filename_thumbnail

            #Setup tiled image parameters if requested
            if showTiles:
                mask_tiles_y = np.array(self.tile_data['mask_y'])
                mask_tiles_x = np.array(self.tile_data['mask_x'])
                tissue_tiles_idx = self.tile_data[ ~pd.isnull(self.tile_data.tilename) ].index
                tissue_points_y = mask_tiles_y[tissue_tiles_idx]+(self.thumbnail_ppt_y/2)
                tissue_points_x = mask_tiles_x[tissue_tiles_idx]+(self.thumbnail_ppt_x/2)

            # Show and/or export plot
            plt.figure()
            plt.axis('off')
            plt.margins(0, 0)
            plt.imshow(self.thumbnail)
            if showTiles:
                plt.hlines(y=mask_tiles_y-mask_tiles_y[0],xmin=0,xmax=self.thumbnail.shape[1], color='b', linestyle='solid', linewidth=0.2)
                plt.vlines(x=mask_tiles_x-mask_tiles_x[0],ymin=0,ymax=self.thumbnail.shape[0], color='b', linestyle='solid', linewidth=0.2)
                plt.plot(tissue_points_x, tissue_points_y, color='k', marker='2', markersize=0.2, linestyle="None")
            if export:
                plt.savefig(finalPath, bbox_inches='tight', pad_inches=0, format="png", dpi=600)
            if not show:
                plt.close()

        return(finalPath)

    def export_bin_mask(self, outdir: Path=None, show: bool=False, export: bool=True, maskName: str="tissue_mask"):
        """
        Exports and/or displays the a binary mask as an image from a WsiManager object, if available.

        Input:
            outdir (Path): File path to directory to contain collection of WsiManager output directories. Optional if WsiManager outdir field has been set.
            show (bool): Wether or not to display the thumbnail as a plot. Default: [False]
            export (bool): Wether or not to export the thumbnail as a PNG file. Default: [True]
            maskName (str): Name of binary mask to be exported. Default: [tissue_mask]
        Output:
            If 'export' is enabled, returns filepath to image, 'None' otherwise.
            Note: Places image in directory: '<outdir>/<wsi_id>/info/' unless a 'outdir' is given.
        """
        finalPath = None
        if hasattr(self,maskName):

            #Validate attribute selection
            binMask = getattr(self, maskName)
            if type(binMask) != np.ndarray and len(np.unique(binMask)) < 3:
                raise ValueError("The attribute selected is not a binary mask")

            if export:
                #Find and validate output directory
                if outdir is None and self.outdir is None:
                    raise ValueError("No output directory has been supplied")
                elif outdir is not None:
                    if isinstance(outdir,str):
                        outdir = Path(outdir)
                    elif not isinstance(outdir,Path):
                        raise ValueError("Output path is not a Path or a string.")

                    #if output directory has correct format, use it directly.
                    if outdir.suffix == "":
                        final_outdir = outdir
                    else:
                        raise ValueError("Output path does not have directory format.")
                else:
                    final_outdir = self.outdir / self.wsi_id / "info"

                filename_mask = self.wsi_id + "___%s_tilesize_x-%d-y-%d.png" % (
                    maskName, self.thumbnail_ppt_x, self.thumbnail_ppt_y)
                
                finalPath = final_outdir / filename_mask

            # Show and/or export plot
            plt.figure()
            plt.axis('off')
            plt.margins(0, 0)
            if show:
                plt.imshow(binMask, cmap="Greys_r")
            if export:
                plt.imsave(finalPath, binMask, cmap="Greys_r")
        
        else:
            raise ValueError("Selected mask is not available as an attribute.")

        return(finalPath)

    #TODO Add save masks as npz archives
    #TODO export tiles
    #TODO create object from from directory

    #### Set properties ####
    @property
    def size(self):
        '''Return total number of tiles for a given WsiManager object'''
        return( len(self.tile_data) )

    @property
    def shape(self):
        '''Return list of [tile rows, tile columns] for a given WsiManager object'''
        y_tiles = max(self.tile_data["index_y"])
        x_tiles= max(self.tile_data["index_x"])
        return( [y_tiles,x_tiles] )

    @property
    def __str__(self):
        return f'WsiManager(ID:{self.wsi_id}; Shape: {self.shape}); Tissue Tiles: n={len(self.tile_data[ ~pd.isnull(self.tile_data.tilename) ])}'


    # #TODO finsh
    # @classmethod
    # def fromdir(self,reffile=''):
    #     '''Create a new WsiManager object from reading an exported WsiManager directory'''
    #     reffile = Path(reffile)
    #     reffile_name = reffile.stem
        
    #     #Validate reffile input
    #     if reffile.is_file() and reffile.suffix == 'tsv' and reffile_name.startswith('info___reference_'):
    #         res = re.match(r'(\w+)___reference_wsi-tilesize_x-(\d+)-y-(\d+)_mask-tilesize_x-(\d+)-y-(\d+)_img-level_(\d+)',reffile_name)

    #         values = res.groups()

    #         self.wsi_id = values[1]
    #         self.wsi_tilesize_x = values[2]
    #         self.wsi_tilesize_y = values[3]
    #         self.mask_tilesize_x = values[4]
    #         self.mask_tilesize_y = values[5]
    #         self.img_lvl = values[6]
    #         self.tile_data = pd.read_csv(str(reffile),sep='\t')

    #         self.index_dim = (max(self.tile_data.index_y), max(self.tile_data.index_x))
    #         self.wsi_dim = (max(self.tile_data.wsi_y)+self.wsi_tilesize_y, max(self.tile_data.wsi_x)+self.wsi_tilesize_x)
    #         self.mask_dim = (max(self.tile_data.mask_y)+self.mask_tilesize_y, max(self.tile_data.mask_x)+self.mask_tilesize_x)
    #         self.fields = [ i for i in self.tile_data.columns if i not in STANDARD_FIELDS ]
    #     return(self)
