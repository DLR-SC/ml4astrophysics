"""
Helper routines to load the images and masks from SpaceNet7, some routines have been found on www.kaggle.com
"""
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pathlib
import geopandas as gpd
import rasterio as rio
from rasterio import features
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.rcParams['figure.dpi'] = 300  # increase plot resolution

class SpaceNet7Data(Dataset):
    def __init__(self, images, masks):
        images = images / 255.
        # mask values are zero and one
        self.images, self.masks = images, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        x, y = self.images[ix], self.masks[ix]
        return x, y


class ChipCreator:
    def __init__(self, chip_dimension=256, raster=False):
        self.chip_dimension = chip_dimension
        self.raster = raster

    def create_chips(self, image):
        np_array = self.__read_image(image)
        # get number of chips per colomn
        n_rows = (np_array.shape[0] - 1) // self.chip_dimension + 1
        # get number of chips per row
        n_cols = (np_array.shape[1] - 1) // self.chip_dimension + 1
        # segment image into chips and return list of chips
        l = []

        for r in range(n_rows):
            for c in range(n_cols):
                start_r_idx = r * self.chip_dimension
                end_r_idx = start_r_idx + self.chip_dimension

                start_c_idx = c * self.chip_dimension
                end_c_idx = start_c_idx + self.chip_dimension
                chip = np_array[start_r_idx:end_r_idx, start_c_idx:end_c_idx]
                if self.raster:
                    chip = np.moveaxis(chip, -1, 0)

                # Image needs to be of shade (dired_image_size, desired_image_size, 3)
                if chip.shape[0] != self.chip_dimension:
                    diff = self.chip_dimension - chip.shape[0]
                    # Add row of zeros, such that the image has the desired dimension
                    chip = np.vstack((chip, np.zeros((diff, chip.shape[1], 4))))
                if chip.shape[1] != self.chip_dimension:
                    diff = self.chip_dimension - chip.shape[1]
                    # Add column of zeros, such that the image has the desired dimension
                    chip = np.hstack((chip, np.zeros((chip.shape[0], diff, 4))))

                l.append(chip)

        return np.array(l)

    @staticmethod
    def __read_image(image):
        # check whether image is a path or array
        if isinstance(image, (pathlib.PurePath, str)):
            with Image.open(image) as img:
                # convert image into np array
                np_array = np.array(img)
            return np_array

        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError(f"Expected Path or Numpy array received: {type(image)}")


def rasterize_geojson(geojson_path, reference_raster_path):
    gdf = gpd.read_file(geojson_path)
    with rio.open(reference_raster_path) as raster:
        r = raster.read(1)

        mask = image = features.rasterize(((polygon, 255) for polygon in gdf['geometry']), out_shape=r.shape)

    return mask


def multiplot_images(list_of_images, title, ncols=4, dpi=300, raster=False, save_fig=False, exercise=2):
    mpl.rcParams['figure.dpi'] = dpi
    nrows = (len(list_of_images) - 1) // ncols + 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))

    fig.tight_layout()
    st = fig.suptitle(f'{title}', fontsize="x-large")

    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    for r, ax in enumerate(axs):
        for c, row in enumerate(ax):
            # get the current index of the image
            i = r * ncols + c
            ax[c].set_title(i)
            image = list_of_images[i]
            # if the image is presented in raster format then move the channel axis
            if raster:
                image = np.moveaxis(image, 0, -1)

            ax[c].imshow(image)

    plt.show()

    if save_fig:
        results_dir = f'../results/Sheet_{exercise}/figs/'

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        fig.savefig(f'{results_dir}/{title}.png', dpi=400)

