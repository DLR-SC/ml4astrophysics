"""
This file provides a tool for semantic segmentation of SpaceNet 7 images
"""
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from scalable_ml.second_exercise.data_loading import SpaceNet7Data, multiplot_images
from scalable_ml.second_exercise.unet_models import UNet
from scalable_ml.training import train_model
import scalable_ml.model_utilities as util
import numpy as np


def main(train=True):

    # define training parameters
    # batch_size = ?
    # epochs = ?

    # read Spacenet 7 data
    images = util.read_spacenet7_images('/your_data_path/L15-0506E-1204N_2027_3374_13/')
    targets = util.read_spacenet7_masks('/your_data_path/L15-0506E-1204N_2027_3374_13/')

    # Store to numpy file
    #np.savez('../data/Unet/image_data.npz', images=images, targets=targets)

    # data = np.load('../data/Unet/image_data.npz')
    # images = data['images']
    # targets = data['targets']
    # targets = torch.from_numpy(targets).long()

    # Plot a single image, requires: import matplotlib.pyplot as plt
    #plt.imshow(images[0])

    # Plot a few images
    # multiplot_images(images[:16], title='Spacenet 7 image')
    # multiplot_images(train_target)

    # CNNs in PyTorch expect each input to have a shape of batch size x channels x height x width
    # i.e. reshape images to match this size
    # Your code ..

    spacenet_data = SpaceNet7Data(images, targets)

    # Split dataset in training and in validation set
    # Your code ..

    # create torch dataset loaders for training
    # Your code ..

    # # choose hardware on which to perform training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # initialize ml_model
    ml_model = UNet(in_channels=3, out_channels=2)
    ml_model.to(device)

    # run summary of model
    #summary(ml_model, (3, 256, 256))

    if train:
        # if training mode is enabled, model is trained and its parameters are stored for future use
        losses, accuracy = train_model(...)
        util.store_model(ml_model, f'../output/', f'unet_model_small.pth')

        # plot loss and accuracy over training
        # Your code ..
    else:
        # if training mode is disabled, simply load pretrained parameters
        util.load_model(...)
        ml_model.eval()


    # apply your model to validation or test data and investigate model performance

    # plot loss and accuracy over training



if __name__ == "__main__":
    main(train=False)
