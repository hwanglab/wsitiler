# WSItiler
Tools for dividing pathology whole slide images into tiles and save them as individual files.

### *Warning:* This project is still under development.

## About

WSItiler tries to optimize the pre-processing of Pathology Whole Slide Images (WSI), which tend to be very large files of tiled images of different magnifications. We do this by identifying regions of interest in a downscaled copy of the WSI and only processing the tiles that correspond to said areas (the foreground) in a parallelized manner using multiprocessing.

## Installation

1. Install [OpenSlide](https://openslide.org). WSItiler uses OpenSlide to interact with WSI files. It is a C library that must be installed separately before installing WSItiler. Here are some examples for some major operating systems:

    * **Debian (Linux):**
        ```bash
        apt-get install python3-openslide
        ```
    * **Windows:** Download Windows binaries [here](https://openslide.org/download/), and copy the binary files (i.e. the contents of ```openslide-win64-20171122\bin\```) to the same directory where your Python interpreter of choice is (```python.exe```) located. This may be in your virtual/conda environment of choice.

1. Use the package manager pip to install WSItiler like below. Rerun this command to check for and install updates.

    ```bash
    pip install git+https://github.com/hwanglab/wsitiler
    ```

## Usage

1. You can ```import wsitiler``` into your own script to set up a custom pipeline or use the supplied color normalization methods under ```wsitiler.normalizer```.
1. You can run WSItiler from the command line as an executable script by running the following command within the appropriate Python environment:

    ```bash
        python -m wsitiler.tiler <options>
    ```
    For more information about how to run WSItiler as an independent command, use:
    ```bash
        python -m wsitiler.tiler -h
    ```


## Contributing
Pull requests are welcome. Ideally, please open an issue first to discuss what you would like to change. Please follow these steps:

1. Create an issue detailing the required/desired changes.
    * If reporting a bug, please include the run settings that caused the error and the error output in the description.
1. Create a new branch from ```main``` and implement your changes there.
1. Commit your changes referencing your issue number.
1. Once you make sure your changes do not break the main use, submit a pull request and notify @jeanrclemenceau

@jeanrclemenceau will review any changes and merge them to the main.

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)