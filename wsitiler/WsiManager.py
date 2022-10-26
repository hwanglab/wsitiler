"""
wsiManager.py
Implement WsiManager, the core class for the wsitiler framework. 
Contains functions for dividing a whole slide image into tiles and
save them as individual files.

Author: Jean R Clemenceau
Date Created: 18/09/2022
"""

import openslide
import time
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging as log

from PIL import Image
from typing import List
from pathlib import Path
from math import ceil, floor
from skimage import measure
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, opening,closing, square

from wsitiler.wsi_utils import describe_wsi_levels
import wsitiler.normalizer as norm

from openslide import OpenSlideError, OpenSlideUnsupportedFormatError
from PIL import UnidentifiedImageError

class WsiManager:
    """
    Class implementing a handler for processing Whole-Slide Image files.
    WsiManager allows efficient tile processing and file export of WSIs.
    """

    # SUPPORTED_WSI_FORMATS defines the WSI formats supported by Openslide.
    SUPPORTED_WSI_FORMATS = [".svs",".ndpi",".vms",".vmu",".scn",".mrxs",".tiff",".svslide",".tif",".bif"]
    # SUPPORTED_TILE_EXPORT_FORMATS defines the file formats supported for exporting tiles
    SUPPORTED_TILE_EXPORT_FORMATS = ['png','npy']
    # PIXELS_PER_TILE defines the default tile edge length m used when breaking WSIs into smaller images (m x m)
    PIXELS_PER_TILE = "512px"
    # NOISE_SIZE_MICRONS defines default maximum size (in microns) for an artifact to be considered as noise in the WSI tissue mask.
    NOISE_SIZE_MICRONS = 256
    # MIN_FOREGROUND_THRESHOLD defines default minimum tissue/background ratio to classify a tile as foreground.
    MIN_FOREGROUND_THRESHOLD = 0.10
    # Background label for segmented tissue
    BACKGROUND_LABEL = -1

    #Constructor with Openslide object
    def __init__(self,wsi_src: Path=None, wsi: openslide.OpenSlide=None, wsi_id: str=None, outdir: Path=None, mpt: str=PIXELS_PER_TILE, wsi_level: int=0, min_tissue: float=MIN_FOREGROUND_THRESHOLD, noise_size: int=NOISE_SIZE_MICRONS, segment_tissue: bool=False, normalization: List[str]=['no_norm'], getEmpty: bool=False):
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
            normalization (List[str]): List of normalization names to appy to during tiling (use  'no_norm' for no normalization). Default: [no_norm]
            getEmpty (bool): Return an empty WsiManager object. Used for internal class methods. Default: [False]

        Output:
            new WSIManager object
        """ % (WsiManager.PIXELS_PER_TILE,WsiManager.MIN_FOREGROUND_THRESHOLD,WsiManager.NOISE_SIZE_MICRONS)
        log.info("Generating new WsiManager Instance -- ID: %s, WSI: %s" % (wsi_id, wsi_src))
        constructor_start_time = time.time()

        # Return empty object if requested
        if getEmpty:
            return(None)

        # Validate and set metadata parameters
        self.img_lvl = wsi_level
        if all([isinstance(i, str) for i in normalization]) and all([i in norm.NORMALIZER_CHOICES for i in normalization]):
            self.normalization = normalization
        else:
            raise ValueError('At least one requested normalization value is not a valid Normalizer option.')

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
                        #Open WSI
                        try:
                            wsi = openslide.OpenSlide(str(wsi_src))
                        except (OpenSlideError,OpenSlideUnsupportedFormatError,UnidentifiedImageError) as e:
                            log.warning("%s - WARNING: WSI could not be read. Skipping: %s\n" % (wsi_id, str(wsi_src)))
                            raise ValueError("WSI could not be opened.")
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
        wsi.close()

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
            self.tissue_chunk_mask[self.tissue_chunk_mask==bg_label] = WsiManager.BACKGROUND_LABEL

            for l in filtered_chunks:
                self.tissue_chunk_mask[self.tissue_chunk_mask == l] = WsiManager.BACKGROUND_LABEL

        # Populate tile reference table
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
                    flat_chunk_tile = flat_chunk_tile[flat_chunk_tile != WsiManager.BACKGROUND_LABEL]
                    if(len(flat_chunk_tile) > 1):
                        chunk_id = np.bincount(flat_chunk_tile).argmax()
                    else:
                        chunk_id = WsiManager.BACKGROUND_LABEL
                    
                    new_row['chunk_id'] = chunk_id

                rowlist.append(new_row)

        # Create reference dataframe
        colnames = ["image_id","tile_id", "index_x", "index_y", "wsi_x", "wsi_y", "mask_x", "mask_y", "tilename", "tissue_ratio"]
        if segment_tissue:
                    colnames.append('chunk_id')
        self.tile_data = pd.DataFrame(data=rowlist, columns=colnames)

        # Remove filenames for empty tiles
        self.tile_data.loc[self.tile_data['tissue_ratio'] < min_tissue, "tilename"] = None

        log.debug("%s - WsiManager instantiation time: %f" % (self.wsi_id, time.time() - constructor_start_time) )

        return(None)
    
    #### Implement Core Functions ####

    def segment_tissue(self):
        """
        Segments tissue mask from the given WsiManager object to find individual tissue regions. Assigns labels to tiles and generates labeled mask.
        """
        log.info("Segmenting Tissue Chunks -- ID: %s" % (self.wsi_id))

        # Get labels for all chunks
        self.tissue_chunk_mask = ndi.label(self.tissue_mask)[0]

        # Filter out chunks smaller than tile area
        tile_area = self.thumbnail_ppt_x*self.thumbnail_ppt_y
        (chunk_label, chunk_size) = np.unique(self.tissue_chunk_mask,return_counts=True)
        filtered_chunks = chunk_label[ chunk_size < tile_area ]
        bg_label = np.unique(self.tissue_chunk_mask[self.tissue_mask == 0])[0]
        self.tissue_chunk_mask[self.tissue_chunk_mask==bg_label] = WsiManager.BACKGROUND_LABEL

        for l in filtered_chunks:
            self.tissue_chunk_mask[self.tissue_chunk_mask == l] = WsiManager.BACKGROUND_LABEL

        # Save chunk_id data to tiles
        self.tile_data['chunk_id'] = WsiManager.BACKGROUND_LABEL

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
                chunk_id = WsiManager.BACKGROUND_LABEL

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
        log.info("Exporting WsiManager instance's metadata -- ID: %s, Output Directory: %s" % (self.wsi_id, outdir))
        export_start_time = time.time()

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
        tissue_chunk_mask_labels_path = self.export_chunk_mask(outdir= final_outdir, export=True, labels=True)

        # Prepare instance attributes for JSON file
        instance_dict = self.__dict__.copy()

        #save record of expirted image locations
        instance_dict['thumbnail_path'] = str(thumbnail_path)
        instance_dict['tissue_mask_path'] = str(tissue_mask_path)
        instance_dict['tissue_chunk_mask_path'] = str(tissue_chunk_mask_path)
        instance_dict['tile_data_path'] = str(final_outdir / filename_tiledf)

        # Format instance attributes for JSON file
        if 'wsi_src' in instance_dict.keys():
            instance_dict['wsi_src'] = str(self.wsi_src)
        if 'outdir' in instance_dict.keys():
            instance_dict['outdir'] = str(self.outdir)
        if 'tile_data' in instance_dict.keys():
            del(instance_dict['tile_data'])

        # Get key names for all image masks
        mask_vars = [i for i in instance_dict.keys() if i.endswith('_mask')]
        mask_vars = ['thumbnail'] + mask_vars
        
        #Get slice dictionary to get all masks & thumbnail
        mask_dir = {k:instance_dict[k] for k in mask_vars}

        #save masks and thumbnail arrays as npz archive
        arrays_path = final_outdir / ("%s___imgmask_arrays.npz" % self.wsi_id)
        instance_dict['array_data_path'] = str(arrays_path)
        np.savez_compressed( arrays_path ,**mask_dir)

        #Remove from masks from instace dictionary
        for aMask in mask_vars:
            del(instance_dict[aMask])
        
        #Save to JSON file
        json_path = final_outdir / ("%s___WsiManagerData.json" % self.wsi_id)
        instance_dict['WsiManager_data_path'] = str(json_path)
        with open(json_path, "w") as outfile:
            json.dump(instance_dict, outfile)
        
        log.debug("%s - Metadata export time: %f" % (self.wsi_id, time.time() - export_start_time) )

        return(None)

    def export_chunk_mask(self, outdir: Path=None, show: bool=False, export: bool=True, labels=False):
        """
        Exports and/or displays the tissue_chunk_mask as an image from a WsiManager object, if available.

        Input:
            outdir (Path): File path to directory to contain collection of WsiManager output directories. Optional if WsiManager outdir field has been set.
            show (bool): Wether or not to display the mask as a plot. Default: [False]
            export (bool): Wether or not to export the mask as a PNG file. Default: [True]
            labels (bool): Wether or not to display the tissue labels in the image. Default: [False]
        Output:
            If 'export' is enabled, returns filepath to image, 'None' otherwise.
            Note: Places image in directory: '<outdir>/<wsi_id>/info/' unless a 'outdir' is given.
        """
        log.info("Exporting Segmented Tissue Mask%s -- ID: %s" % (" and Labels" if labels else "", self.wsi_id))
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

                hasLabelStr = "-labeled" if labels else ""
                filename_chunkmask = self.wsi_id + "___chunk_mask%s_tilesize_x-%d-y-%d.png" % (
                    hasLabelStr,self.thumbnail_ppt_x, self.thumbnail_ppt_y)
                
                finalPath = final_outdir / filename_chunkmask

            #Setup colormap for black background
            a_cmap = cm.get_cmap("viridis").copy()
            a_cmap.set_under(color='black')

            # Show and/or export plot
            plt.figure()
            plt.axis('off')
            plt.margins(0, 0)
            plt.imshow(self.tissue_chunk_mask, cmap=a_cmap, interpolation='nearest', vmin=0)
            if labels:
                measures = measure.regionprops_table(self.tissue_chunk_mask, properties={'label','centroid'})
                measures = pd.DataFrame(measures)
                for i,props in measures.iterrows():
                    plt.text(props['centroid-1'], props['centroid-0'],int(props['label']),
                        fontweight='bold',fontsize="medium",color="gainsboro")
            if export:
                # plt.imsave(finalPath, self.tissue_chunk_mask, cmap=a_cmap, vmin=0)
                plt.savefig(finalPath, bbox_inches='tight', pad_inches=0, format="png", dpi=600)
            if not show:
                plt.close()

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
        log.info("Exporting WSI Thumbnail%s -- ID: %s" % (" and Tiles" if showTiles else "", self.wsi_id))
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
        log.info("Exporting Binary Mask -- ID: %s, Mask: %s" % (self.wsi_id, maskName))

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

    def export_tiles(self, filetype = "png", tile_idx_list: List[str]=[], outdir: Path=None, normalizer=None, ref_img: Path=None, wsi_image: openslide.OpenSlide=None):
        """
        Import a WSI, split in to tiles, normalize color if requested, and save individual tile files to output directory.

        Input:
            self (WsiManager): WsiManager instance for exporting its tiles
            filetype (str): Format for exporting tile files. Options: ['png','npy']. Default: [png].
            tile_idx_list (List[str]): List of indeces from tile_data DataFrame that will be exported. Default: export all non-background tiles.
            outdir (Path): File path to directory that contains collection of WsiManager output directories. Optional if WsiManager outdir field has been set.
            normalizer (str or Normalizer): Valid name or Normalizer object to be used for tile stain normalization. Default: [None].
            ref_img (Path): Path to reference image for stain normalization. Default: Default image reference from wsitiler.normalizer.
            wsi_image (Openslide): OpenSlide object used to extract image tiles (used for multiprocessing). Default: [None].

        Output:
            Funtion exports tiles as files to output directory.
        """
        tile_cnt=len(tile_idx_list)
        tile_cnt_str = str(tile_cnt) if tile_cnt>0 else "All",
        export_start_time = time.time()
        log.info("Exporting %s Tiles -- ID: %s" % (tile_cnt_str, self.wsi_id)) 
        
        # Validate exporting file type
        if filetype not in ['png','npy']:
            raise ValueError("Tile export format not supported.")
        log.debug("Output Format: %s" % str(filetype))

        # Validate normalizer
        if normalizer is None or isinstance(normalizer, str):
            # if normalizer not given, use object's first requested method as default
            if normalizer is None and len(self.normalization) >0:
                normalizer = norm.setup_normalizer(self.normalization[0], ref_img)
            # if normalizer method is given but not valid, raise exception
            elif normalizer is not None and normalizer not in norm.NORMALIZER_CHOICES:
                raise ValueError("Supplpied normalizer choice name is not valid.")
            # if given normalizer method is valid, generate Normalizer object
            else:
                normalizer = norm.setup_normalizer(normalizer, ref_img)
        else:
            # if normalizer is acctive Normalizer object, ensure method is fit to target
            if isinstance(normalizer, norm.Normalizer):
                if not normalizer.is_fit and normalizer != None:
                    if ref_img is None:
                        ref_img = norm.get_target_img()
                    elif not ref_img.is_file():
                        raise ValueError ("Supplied reference image path is not a file or does not exist.")
                    
                    #Fit normalizer if not fit
                    normalizer.fit(ref_img)
            else:
                raise ValueError("Supplied normalizer is not a Normalizer object")
        log.debug("Normalizer: %s" % normalizer)
        
        # Validate output directory
        if outdir is None and self.outdir is None:
            raise ValueError("Output path has not been given.")
        
        # Validate and set output to given value
        if outdir is not None:
            if isinstance(outdir,str):
                outdir = Path(outdir)
            elif not isinstance(outdir,Path):
                raise ValueError("Output path is not a Path or a string.")

            #if output directory has correct format, save it.
            if outdir.suffix == "":
                if self.outdir is None:
                    self.outdir = outdir
            else:
                raise ValueError("Output path does not have directory format.")
        
        # Make final tile output directory path
        final_outdir = self.outdir / self.wsi_id / "tiles" / normalizer.method
        log.debug("Final output directory: %s" % str(final_outdir))

        # Ensure output directory exists
        if not final_outdir.is_dir():
            final_outdir.mkdir(parents=True)

        # Validate list of tile indices
        if len(tile_idx_list) < 1:
            # If no subset given, export all foreground tiles
            exported_tiles = self.tile_data[ ~pd.isnull(self.tile_data.tilename) ]
        else:
            exported_tiles = self.tile_data.iloc[tile_idx_list]
        log.debug("Tiles for exporting: %d" % len(exported_tiles))
        
        # Open and prepare input WSI if not given
        wsi_given = True
        if wsi_image is None:
            wsi_given = False
            wsi_image = openslide.open_slide(str(self.wsi_src))
            log.debug("Default WSI object: %s" % str(wsi_image))
        elif isinstance(wsi_image, Path):
            log.debug("Supplied WSI path: %s" % str(wsi_image))
            wsi_image = openslide.open_slide(str(wsi_image))
        else:
            log.debug("Supplied WSI object: %s" % str(wsi_image))

        # Process and export each tile sequentially
        tiling_start_time = time.time()
        for index, aTile in exported_tiles.iterrows():

            # Extract tile region
            aTile_img = wsi_image.read_region((aTile["wsi_x"], aTile["wsi_y"]), level=0,
                                    size=(self.wsi_ppt_x, self.wsi_ppt_y))

            #Convert to RGB array
            aTile_img = np.array( aTile_img.convert('RGB') )

            # Normalize if required
            if normalizer != None:
                aTile_img = normalizer.transform(aTile_img)

            # Save tile image to file
            if aTile['tilename'] is not np.NaN and aTile['tilename'] is not None:
                filename = (aTile['tilename']+"."+filetype)
                if filetype == 'png':
                    plt.imsave(final_outdir / filename, aTile_img)
                elif filetype == 'npy':
                    np.save(file=final_outdir / filename, arr=aTile_img)
                else:
                    raise ValueError("Tile export format not supported. Tile not Exported: %s" % aTile['filename'])
                log.debug("Exporting tile #%d: %s" % (index,filename) )
            else:
                log.debug("Tile filename unavailable (#%d)" % index )
            tiling_end_time = time.time()
            log.debug("Tile Export Time: %f" % (tiling_end_time-tiling_start_time))
            

        if not wsi_given:
            wsi_image.close()

        log.debug("%s - %s tile export time: %f" % (self.wsi_id, tile_cnt_str, time.time() - export_start_time) )

        return
    
    def export_tiles_multiprocess(self, filetype = "png", tile_idx_list: List[str]=[], outdir: Path=None, normalizers: List[str]=None, ref_img: Path=None,  cores: int=1):
        """
        Import a WSI, split in to tiles, normalize color if requested, and save individual tile files to output directory.

        Input:
            self (WsiManager): WsiManager instance that will be exporting tiles
            filetype (str): Format for exporting tile files. Options: ['png','npy']. Default: [png].
            tile_idx_list (List[str]): List of indeces from tile_data DataFrame that will be exported. Default: export all non-background tiles.
            outdir (Path): File path to directory that contains collection of WsiManager output directories. Optional if WsiManager outdir field has been set.
            normalizers (list of str): List of valid Normalizer method names to be used for tile stain normalization. Default: [instance's peselected normalizers].
            ref_img (Path): Path to reference image for stain normalization. Default: Default image reference from wsitiler.normalizer.
            cores (int): number of cores used in multiprocessing. Default: [1].

        Output:
            Funtion exports tiles as files to output directory.
        """
        tile_cnt=len(tile_idx_list)
        tile_cnt_str = str(tile_cnt) if tile_cnt>0 else "All",
        export_start_time = time.time()
        log.info("Exporting %s Tiles (multiprocess) -- ID: %s, cores: %d" % (self.wsi_id, cores)) 
         
         # Validate exporting file type
        if filetype not in ['png','npy']:
            raise ValueError("Tile export format not supported.")
        log.debug("Output Format: %s" % str(filetype))
        
        # Validate cores
        if cores < 1:
            raise ValueError("Requested core count is too low. Must request at least 1 core.")
        elif cores > mp.cpu_count():
            raise ValueError("Requested core count is too High. You only  have %s cores available." % mp.cpu_count())

        # Validate list of tile indices
        if len(tile_idx_list) < 1:
            # If no subset given, export all foreground tiles
            exported_tiles = self.tile_data[ ~pd.isnull(self.tile_data.tilename) ].copy()
        else:
            exported_tiles = self.tile_data.iloc[tile_idx_list].copy()
        
        # Validate normalizer
        if normalizers is None:
            normalizers = self.normalization
        elif not isinstance(normalizers, list):
            raise ValueError("normalizers is not a list")
        elif len(normalizers) < 1:
            raise ValueError("Empty list of normalizers was supplied")
        log.debug("Requested normalizers: %s" % ", ".join(normalizers) )

        # Validate and set output to given value
        if outdir is not None:
            if isinstance(outdir,str):
                outdir = Path(outdir)
            elif not isinstance(outdir,Path):
                raise ValueError("Output path is not a Path or a string.")
        elif self.outdir is not None:
            outdir = self.outdir
        else:
            raise ValueError("Output path has not been given.")
        log.debug("Main output directory: %s" % str(outdir))

        # Split list of tiles for multiprocessing
        if cores == 1:
            core_group = 0
        else:
            core_group = np.repeat( range(0,cores-1), np.floor(len(exported_tiles) / (cores-1)) )
            core_group = np.append(core_group, np.repeat(cores, (len(exported_tiles)-len(core_group))) )
        
        exported_tiles['core_group'] = core_group
        log.debug("Tiles for exporting: %d" % len(exported_tiles))
        
        # Open object's WSI as OpenSlide object
        the_wsi = openslide.open_slide(str(self.wsi_src))

        #Start exporting iterating over normalization methods
        for aNormMethod in normalizers:

            #Prepare normalizer object
            if aNormMethod not in norm.NORMALIZER_CHOICES:
                log.warn("Invalid normalizer option: %s. Skipping tile export." % aNormMethod) 
                continue
            aNormalizer = norm.setup_normalizer(aNormMethod,ref_img)
            log.debug("Processing tiles with '%s' normalization" % aNormalizer.method )

            # Process tiles in parallel        
            pool = mp.Pool(cores)
            async_start_time = time.time()

            # Export tiles in parallel
            for aCoreGroup in np.unique(core_group):
                aTileList = list(exported_tiles.index[exported_tiles.core_group == aCoreGroup])
                log.debug("Exporting %d tiles (%s) from group #%d" % (len(aTileList),aNormalizer.method,aCoreGroup) )

                pool.apply_async(func=call_export_tiles,kwds={
                    'obj': self,
                    'filetype': filetype,
                    'tile_idx_list': aTileList,
                    'outdir': outdir,
                    'normalizer': aNormalizer
                })
        
            pool.close()
            pool.join()
            pool.terminate()
            async_end_time = time.time()
            log.debug("%s Tile Export Time: %f" % (aNormalizer.method, async_end_time-async_start_time))
        
        the_wsi.close()
        log.debug("%s - %s total multiprocess tile export time: %f" % (self.wsi_id, tile_cnt_str, time.time() - export_start_time) )

        return

    # Override string representation function
    def __str__(self):
        return f'WsiManager(ID:{self.wsi_id}; Shape: {self.shape}; Tissue Tiles: n={len(self.tile_data[ ~pd.isnull(self.tile_data.tilename) ])})'

    # Override class representation function
    def __repr__(self):
        repr_str= f"WsiManager '{self.wsi_id}': {len(self.tile_data[ ~pd.isnull(self.tile_data.tilename) ])}{self.shape}"
        if len(repr_str) > 38:
            repr_str= f"WsiManager '{self.wsi_id}'"
        if len(repr_str) > 38:
            repr_str= repr_str[:-4]+"...'"
        repr_str = "<"+repr_str+">"

        return(repr_str)

    #### Set Class Methods ####
    @classmethod
    def fromdir(cls,indir: Path=None):
        '''Create a new WsiManager object from reading an exported WsiManager directory

        Input:
            indir (Path): File path to directory containing WsiManager data.
        Output:
            New WsiManager object based on data contained in given directory.
        '''
        import_start_time = time.time()
        log.info("Importing WsiManager from Path -- Input Directory: %s" % (indir)) 
        #Find and validate output directory
        if indir is None:
            raise ValueError("No input directory has been supplied")
        else:
            if isinstance(indir,str):
                indir = Path(indir)
            elif not isinstance(indir,Path):
                raise ValueError("Input path is not a Path or a string.")
            
        # format input file
        indir = indir if indir.stem == "info" else indir / "info"

        # Import data from JSON
        jsonFileList = list(indir.glob("*___WsiManagerData.json"))

        if len(jsonFileList) > 0:
            with open(jsonFileList[0]) as json_file:
                # Get data and prepare object
                jsonData = json.load(json_file)
                newObj = WsiManager(getEmpty=True)

                #Save object metadata
                newObj.wsi_id = jsonData['wsi_id']
                newObj.wsi_src = Path(jsonData['wsi_src'])
                newObj.outdir = Path(jsonData['outdir'])
                newObj.normalization = jsonData['normalization']

                # Save tile metaparameters
                newObj.wsi_mpp = jsonData['wsi_mpp']
                newObj.wsi_ppt_x = jsonData['wsi_ppt_x']
                newObj.wsi_ppt_y = jsonData['wsi_ppt_y']
                newObj.thumbnail_mpp = jsonData['thumbnail_mpp']
                newObj.thumbnail_ratio = jsonData['thumbnail_ratio']
                newObj.thumbnail_ppt_x = jsonData['thumbnail_ppt_x']
                newObj.thumbnail_ppt_y = jsonData['thumbnail_ppt_y']

                # Save tile data
                newObj.tile_data = pd.read_csv(jsonData['tile_data_path'],sep='\t')

                # Save thumbnail and maks
                mask_npz = np.load(Path(jsonData['array_data_path']))
                for key,ar in mask_npz.items():
                    setattr(newObj,key,ar)

        else:
            raise ValueError("Input path is not a Path or a string.")
        
        log.debug("%s - Total WsiManager import time: %f" % (newObj.wsi_id, time.time() - import_start_time) )

        return(newObj)


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

###Unbound callable functions for multiprocessing
def call_export_tiles(obj, filetype = "png", tile_idx_list: List[str]=[], outdir: Path=None, normalizer=None, ref_img: Path=None, wsi_image: openslide.OpenSlide=None):
    obj.export_tiles(filetype, tile_idx_list, outdir, normalizer, ref_img, wsi_image)
    return
