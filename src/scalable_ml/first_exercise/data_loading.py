from torch.utils.data import Dataset


class FMNISTDataset(Dataset):

    def __init__(self, images, labels):
        images = images.float()/255.
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        # Size of tensor is number_of_images x 768
        # Info: torch.view is very similar to torch.reshape
        images = images.view(-1, 28 * 28)
        self.images, self.labels = images, labels

    def __getitem__(self, ix):
        image, label = self.images[ix], self.labels[ix]
        return image, label

    def __len__(self):
        return len(self.images)
