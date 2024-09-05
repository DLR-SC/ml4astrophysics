from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def read_skytile_table(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[1].data

        # extract the coloumns of interest
        srvmap = data['SRVMAP'].byteswap().newbyteorder()
        ra_min = data['RA_MIN'].byteswap().newbyteorder()
        ra_max = data['RA_MAX'].byteswap().newbyteorder()
        dec_min = data['DE_MIN'].byteswap().newbyteorder()
        dec_max = data['DE_MAX'].byteswap().newbyteorder()
        ra_cen = data['RA_CEN'].byteswap().newbyteorder()
        dec_cen = data['DE_CEN'].byteswap().newbyteorder()

        # Create a pandas DataFrame from the extracted data
        df = pd.DataFrame({
            'SRVMAP': srvmap,
            'RA_MIN': ra_min,
            'RA_MAX': ra_max,
            'DEC_MIN': dec_min,
            'DEC_MAX': dec_max,
            'RA_CEN': ra_cen,
            'DEC_CEN': dec_cen
        })

    return df

def find_clusters_in_skytile(cluster_catalog_df, skytile_definitions_df, skytile_number):
    skytile_definition = skytile_definitions_df[skytile_definitions_df['SRVMAP'] == int(skytile_number)]
    ra_min = skytile_definition['RA_MIN'].values[0]
    ra_max = skytile_definition['RA_MAX'].values[0]
    dec_min = skytile_definition['DEC_MIN'].values[0]
    dec_max = skytile_definition['DEC_MAX'].values[0]

    clusters_in_skytile = cluster_catalog_df[(cluster_catalog_df['RA'] > ra_min) &
                                             (cluster_catalog_df['RA'] < ra_max) &
                                             (cluster_catalog_df['DEC'] > dec_min) &
                                             (cluster_catalog_df['DEC'] < dec_max)]

    return clusters_in_skytile

def create_mask(skytile_number, download_path, clusters_in_skytile, frac_R500_to_mask=1.0):
    # read the skytile image
    skytile_image = os.path.join(download_path, f'em01_{skytile_number}_024_Image_c010.fits.gz')
    with fits.open(skytile_image) as file:
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

    # save the mask to a new file
    mask_file = os.path.join(download_path, f'em01_{skytile_number}_mask.fits')
    fits.writeto(mask_file, hdu, file[0].header, overwrite=True)


def plot_mask(download_path, skytile_number):
    mask_file = os.path.join(download_path, f'em01_{skytile_number}_mask.fits')

    with fits.open(mask_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # Create a WCS object
    wcs = WCS(header)
    
    # Plot mask with WCS projection
    fig, ax = plt.subplots(1,1,subplot_kw={'projection': wcs})
    ax.imshow(data, cmap='gray')
    ax.set_title('Mask')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    # ax.coords.grid(color='white', ls='dotted')
    plt.show()


if __name__ == '__main__':
    file_path = 'SKYMAPS_052022_MPE.fits'
    skytile_definitions_df = read_skytile_table(file_path)
    #print(skytile_definitions_df.head())

    # read catalog
    cluster_catalog_df = pd.read_csv('data/erass1cl_cleaned.csv')
    bad_clusters_df = pd.read_csv('erass1cl_bad_clusters.csv')

    # define skytile numbers to process
    skytile_numbers = ['060120', '057120', '058123', '055123']

    # find clusters in skytiles
    for skytile_number in skytile_numbers:
        clusters_in_skytile = find_clusters_in_skytile(cluster_catalog_df, skytile_definitions_df, skytile_number)
        print(f'Number of good clusters in skytile {skytile_number}: {len(clusters_in_skytile)}')
        #print(clusters_in_skytile)
        # check if they are also bad clusters in the skytile (bad clusters have R500=-1, so they cannot be masked)
        bad_clusters_in_skytile = find_clusters_in_skytile(bad_clusters_df, skytile_definitions_df, skytile_number)
        print(f'Number of bad clusters in skytile {skytile_number}: {len(bad_clusters_in_skytile)}')
        create_mask(skytile_number, 'downloaded_files', clusters_in_skytile, frac_R500_to_mask=1.0)

    # plot the masks
    for skytile_number in skytile_numbers:
        plot_mask('downloaded_files', skytile_number)
