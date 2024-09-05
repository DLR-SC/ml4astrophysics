import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
from create_masks import read_skytile_table, find_clusters_in_skytile


class eROSITA(Dataset):
    """
    This dataset reads the skytile images from `path` and stores sub-images of size
         (len(channels), patch_size, patch_size) as torch.tensor's.
    For an explanation of what the skytile images represent, visit
        https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_ProductsDescription/file_naming_scheme_dr1.html
    """
    def __init__(self,
                 path,
                 patch_size=256,
                 normalize_by_exposure=True,
                 channels=[1, 2, 3, 4, 5, 6, 7]
                 ):
        """
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
        self.clusters_df = pd.read_csv(os.path.join(path, "erass1cl_cleaned.csv"))
        self.skytile_definitions = read_skytile_table(os.path.join(path, "SKYMAPS_052022_MPE.fits"))
        self.skytiles = []
        self.skytile_imgs = []
        self.skytile_masks = []
        self.skytile_img_patches = None
        self.skytile_mask_patches = None
        self.__load_and_normalize_images__(path, normalize=normalize_by_exposure)
        self.__create_patches__()

    def __load_and_normalize_images__(self, path, normalize, eps=1e-6):
        for folder in os.listdir(path):         # iterate over skytiles
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):  # skip files that are not folders, e.g. 'erass1cl_cleaned.csv'
                continue

            folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])     # select first and only subfolder
            images = []
            catalogs = []
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


            clusters_in_skytile = find_clusters_in_skytile(self.clusters_df, self.skytile_definitions, skytile_index)
            with fits.open(os.path.join(file_path)) as hdul:
                mask = create_mask(hdul, clusters_in_skytile).byteswap().newbyteorder()
                self.skytile_masks.append(torch.tensor(mask, dtype=torch.uint8))


    def __create_patches__(self):
        chip_creator = ChipCreator(chip_dimension=self.patch_size)
        img_patches = []
        mask_patches = []
        for image in self.skytile_imgs:
            img_patches.append(chip_creator.create_chips(image))
        for mask in self.skytile_masks:
            mask_patches.append(chip_creator.create_chips(mask.unsqueeze(0)))

        self.skytile_img_patches = torch.tensor(np.stack(img_patches)).view(-1, self.patch_size, self.patch_size, len(self.channels))
        del self.skytile_imgs
        gc.collect()

        self.skytile_mask_patches = torch.tensor(np.stack(mask_patches)).view(-1, self.patch_size, self.patch_size, 1)
        del self.skytile_masks
        gc.collect()

        self.skytile_img_patches = self.skytile_img_patches.permute(0, 3, 1, 2)
        self.skytile_mask_patches = self.skytile_mask_patches.permute(0, 3, 1, 2)


    def __len__(self):
        return self.skytile_img_patches.shape[0]

    def __getitem__(self, item):
        return self.skytile_img_patches[item], self.skytile_mask_patches[item]


class eROSITAwithNH(Dataset):
    """
    This dataset reads the skytile images of a specified energy band, the corresponding exposure map
     and the corresponding NH Map from `path` and stores sub-images of size
         (3, patch_size, patch_size) as torch.tensor's.
    For an explanation of what the skytile images represent, visit
        https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_ProductsDescription/file_naming_scheme_dr1.html
    """
    def __init__(self,
                 path,
                 patch_size=256,
                 channel=4):
        """
        :param path:
        :param patch_size:
        :param channel:
        """
        super().__init__()

    def __len__(self):
        ...

    def __getitem__(self, item):
        ...


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
                # NOTE: We need to reduce precision to save memory,
                # as I frequently got a numpy.core._exceptions._ArrayMemoryError for float64 arrays
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


def create_mask(file, clusters_in_skytile, frac_R500_to_mask=1.0):
    # read the skytile image
    hdu = file[0].data

    # make whole image black
    hdu[:,:] = 0

    # create a mask for each cluster
    center_ra = clusters_in_skytile['RA'].values
    center_dec = clusters_in_skytile['DEC'].values
    radii_arcmin = clusters_in_skytile['R500_arcmin'].values*frac_R500_to_mask

    # convert ra and dec to pixel coordinates
    w = WCS(file[0].header)
    center_xc, center_yc = w.all_world2pix(center_ra, center_dec, 0)
    # convert radii to pixel units

    # Calculate the pixel scale (degrees/pixel) and convert to arcmin/pixel
    pixel_scale = w.wcs.cdelt[1] * 60  # w.wcs.cdelt is in degrees/pixel
    radii_pixels = radii_arcmin / pixel_scale

    for i in range(len(center_xc)):
        # size of the image
        H, W = hdu.shape
        # x and y coordinates per every pixel of the image
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        # squared distance from the center of the circle
        d2 = (x - center_xc[i])**2 + (y - center_yc[i])**2
        # mask is True inside of the circle
        mask = d2 < radii_pixels[i]**2

        outside = np.ma.masked_where(mask, hdu)
        hdu[mask] = 1

    return hdu


if __name__ == "__main__":
    channels = [1, 2, 3, 4, 5, 6, 7]
    data = eROSITA('data',
                   patch_size=256,
                   normalize_by_exposure=True,
                   channels=channels)
    dataloader = DataLoader(data, batch_size=16, shuffle=True)
    for img, mask in dataloader:
        print(f"img.shape=={img.shape}")
        fig, ax = plt.subplots(ncols=img.shape[1] + 1)
        for c in range(img.shape[1]):
            ax[c].imshow(img[0, c].detach().numpy())
            ax[c].axis("off")

        ax[img.shape[1]].imshow(mask[0, 0].detach().numpy())
        ax[img.shape[1]].axis("off")
        ax[img.shape[1]].set_title("Mask")
        plt.show()
