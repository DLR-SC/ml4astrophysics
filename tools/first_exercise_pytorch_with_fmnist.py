"""
Exercise sheet 1 - Image Segmentation

This file provides the basic program for solving exercise sheet 1.
You are free to use your own implementation.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary

from scalable_ml.first_exercise.data_loading import FMNISTDataset
from scalable_ml.first_exercise.ml_models import FullyConnectedNeuralNetwork
from scalable_ml.training import train_model
import scalable_ml.model_utilities as util


def main(train=True):

    # define important training hyperparameters
    # batch_size = ?
    # epochs = ?

    # download data and define dataset class
    data_folder = f'../data/FMNIST'  # This can be any directory you want to store data to
    fmnist = torchvision.datasets.FashionMNIST(data_folder, download=True, train=True)
    # Your code ...

    # In a similar way we also have to load the validation data and define a class
    val_fmnist = torchvision.datasets.FashionMNIST(data_folder, download=True, train=False)
    # Your code ...

    # create torch dataset loaders for training
    # Your code ...

    # Plot a few images
    # img_grid = torchvision.utils.make_grid(train_images[0:3, :, :])
    # util.matplotlib_imshow(img_grid, one_channel=True)

    # initialize ml_model and potentially port it to GPU
    # Your code ...

    # run summary of model (C, H, W) = (channels, heigth, width)
    # summary(ml_model, (1, 28 * 28))

    if train:
        # if training mode is enabled, model is trained and its parameters are stored for future use
        losses, accuracy = train_model(...)
        # finally, store your model on disk so that it can be used in inference phase (train=False)
        #util.store_model(ml_model, f'../output/', f'my_first_model.pth')
    else: # training data is not required here
        # if training mode is disabled, simply load pretrained parameters
        util.load_model(...)

    # apply your model to validation or test data and investigate model performance

    # plot loss and accuracy over training



if __name__ == "__main__":
    main(train=True)

