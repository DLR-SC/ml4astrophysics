"""
This script provides several utility functions and classes used for preparing and output training.

. codeauthor:: Wadim Koslow and Alexander Ruettgers

Following functionality is included

* get device for training
* storing and loading pytorch model weights
* print model parameters
* resizing of deep feature extraction
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from second_exercise.data_loading import ChipCreator, rasterize_geojson


def store_model(model, path, model_name):
    """
    Stores parameters of pytorch model in a pth-file. If the given path already exists, parameters will be overwritten

    :param model: pytorch model
    :type model: pytorch class
    :param path: data path, where parameters will be stored
    :type path: string
    :param model_name: name of the model
    :type model_name: string
    """
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        print(f'Model with name {model_name} already exists. overwriting previous model')
    else:
        print(f'Storing parameters of model {model_name}.')

    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, model_name))


def load_model(model, path, model_name):
    """
    Loads pytorch model from file that was written from function "store_model".

    :param model: pytorch model whose parameters will be overwritten with the loaded data
    :type model: pytorch class
    :param path: data path, from where parameters will be loaded
    :type path: string
    :param model_name: model name
    :type model_name: string
    """
    model.load_state_dict(torch.load(os.path.join(path, model_name)))


def read_spacenet7_images(folder, chip_dimension=256):
    """
    Loads SpaceNet 7 images and divides them into 4 x 4 sub-images with resolution 256 x 256 pixels

    :param folder: data path, from where images will be loaded
    :type folder: string
    :param chip_dimension:  resolution of the subimages
    :type chip_dimension: integer
    """
    chips_256 = ChipCreator(chip_dimension, raster=False)

    data = []

    path = f'{folder}/images_masked/'
    ir = sorted(os.listdir(path))
    for i in ir:
        images = chips_256.create_chips(f'{path}/{i}')
        data.append(images[:, :, :, :3])

    return np.concatenate(data, axis=0)


def read_spacenet7_masks(folder):
    """
    Loads SpaceNet 7 masks with information regarding the buildings in the image

    :param folder: data path, from where images will be loaded
    :type folder: string
    """
    chips_256 = ChipCreator(raster=False)

    data = []

    path_images = f'{folder}/images_masked/'
    path_masks = f'{folder}/labels_match_pix/'
    ir_im = sorted(os.listdir(path_images))
    ir_masks = sorted(os.listdir(path_masks))
    for i, j in zip(ir_masks, ir_im):
        test_mask = rasterize_geojson(f'{path_masks}/{i}', f'{path_images}/{j}')
        targets = chips_256.create_chips(test_mask)
        data.append(targets)

    return np.concatenate(data, axis=0)


# @torch.no_grad()
def accuracy(x, y, model):
    # model.eval() # <- let's wait till we get to dropout section
    # get the prediction matrix for a tensor of `x` images
    prediction = model(x)
    # compute if the location of maximum in each row coincides
    # with ground truth
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return 1. / (is_correct.size(dim=0)) * is_correct.cpu().numpy().sum().astype(float)


# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    img = img * 1  # unnormalize
    npimg = img.numpy()
    for i in range(npimg.shape[0]):
        plt.imshow(npimg[i, :, :], cmap="Greys")
        plt.show()


def print_model_parameters(model, print_all_data=False):
    """
    Prints out the mean weight and bias values of every layer of a pytorch model

    :param model: pytorch model
    :type model: pytorch class
    :param print_all_data: if true, prints all weight values in addition to its mean.
    :type print_all_data: bool
    """
    params = model.named_parameters()
    print('Model parameters:')
    for name, param in params:
        print(f'Mean parameter {name}: {np.mean(param.data.cpu().numpy())}')
        if print_all_data:
            print(param.data)
