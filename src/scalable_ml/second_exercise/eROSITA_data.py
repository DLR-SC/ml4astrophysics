import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
import pathlib


class eROSITA(Dataset):
    def __init__(self, path,
                 patch_size=256,
                 normalize_by_exposure=True,
                 channels=[1, 2, 3, 4, 5, 6, 7]
                 ):
        """
        This dataset reads the skytile images from `path` and stores sub-images of size
         (len(channels), patch_size, patch_size) as torch.tensor's.
        For an explanation of what the skytile images represent, visit
            https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_ProductsDescription/file_naming_scheme_dr1.html

        :param path: Directory (as a string) of a folder with subfolders, each containing the following subfolders:
                        - `DET_010': Folder containing the Exposure Maps of each channel-
                        - 'EXP_010': Folder containing the Images of each channel
        :param patch_size: Integer height and width of sub-patches of the skytile images
        :param normalize_by_exposure: Boolean determining whether the Images get normalized by their corresponding exposure map.
        :param channels: The Energy bands to be included in the image
        """
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.skytiles = []
        self.skytile_imgs = []
        self.skytile_patches = None
        self.__load_and_normalize_images__(path, normalize=normalize_by_exposure)
        self.__create_patches__(self.skytile_imgs, patch_size)

    def __load_and_normalize_images__(self, path, normalize, eps=1e-6):
        for folder in os.listdir(path):         # iterate over skytiles
            if folder == '138':
                break
            folder_path = os.path.join(path, folder)
            folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])     # select first and only subfolder
            images = []
            exposure_maps = []
            for file in os.listdir(os.path.join(folder_path, 'EXP_010')):           # get all images
                skytile_index = file.split('_')[1]
                if 'Image' not in file.split('_'):
                    continue
                file_path = os.path.join(folder_path, 'EXP_010', file)
                with fits.open(os.path.join(file_path)) as hdul:
                    images.append(np.array(hdul[0].data))

            if normalize:
                for file in os.listdir(os.path.join(folder_path, 'DET_010')):       # get all exposure maps
                    if 'ExposureMap' not in file.split('_'):
                        continue
                    file_path = os.path.join(folder_path, 'DET_010', file)
                    with fits.open(os.path.join(file_path)) as hdul:
                        exposure_maps.append(np.array(hdul[0].data))

            image = np.stack([images[i-1] for i in self.channels])
            if normalize:
                exposure_map = np.stack([exposure_maps[i-1] for i in self.channels])
                image = np.divide(image, exposure_map + eps)                    # add small epsilon to avoid div by 0.

            self.skytiles.append(skytile_index)
            self.skytile_imgs.append(torch.tensor(image))

    def __create_patches__(self, images, patch_size):
        chip_creator = ChipCreator(chip_dimension=self.patch_size)
        patches = []
        for image in images:
            patches.append(chip_creator.create_chips(image))
        self.skytile_patches = torch.tensor(np.stack(patches)).view(-1, self.patch_size, self.patch_size, len(self.channels))
        self.skytile_patches = self.skytile_patches.permute(0, 3, 1, 2)

    def __len__(self):
        return self.skytile_patches.shape[0]

    def __getitem__(self, item):
        # TODO: Include masks in Dataset !!!
        return self.skytile_patches[item]


class ChipCreator:
    def __init__(self, chip_dimension=256, raster=False):
        self.chip_dimension = chip_dimension
        self.raster = raster

    def create_chips(self, image):
        np_array = image.permute(1, 2, 0).detach().numpy()
        # get number of chips per colomn
        n_rows = (np_array.shape[0] - 1) // self.chip_dimension + 1
        # get number of chips per row
        n_cols = (np_array.shape[1] - 1) // self.chip_dimension + 1
        # segment image into chips and return list of chips
        lst = []

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
                    chip = np.vstack((chip, np.zeros((diff, chip.shape[1], np_array.shape[2]))))
                if chip.shape[1] != self.chip_dimension:
                    diff = self.chip_dimension - chip.shape[1]
                    # Add column of zeros, such that the image has the desired dimension
                    chip = np.hstack((chip, np.zeros((chip.shape[0], diff, np_array.shape[2]))))

                lst.append(chip)

        return np.array(lst)

    """
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
    """


if __name__ == "__main__":
    data = eROSITA('data',
                   patch_size=256,
                   normalize_by_exposure=True,
                   channels=[1, 2, 3, 4, 5, 6, 7])
    dataloader = DataLoader(data, batch_size=16, shuffle=True)
    for img in dataloader:
        print(img.shape)
        fig, ax = plt.subplots(ncols=7)
        for c in range(7):
            ax[c].imshow(img[0, c].detach().numpy())
            ax[c].axis("off")

        plt.show()
