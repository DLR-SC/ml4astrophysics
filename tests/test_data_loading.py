import pytest
import torch
from scalable_ml.first_exercise.data_loading import FMNISTDataset


# @pytest.fixture are used to initialize test functions with known input so that the test output
# is reliable and consistent. If the output is different as expected, the test should fail.
# More information: https://docs.pytest.org/en/6.2.x/fixture.html
@pytest.fixture
def dummy_images():
    # the first image consists of only "1s", the second image consists of only "2s", and the third of only "3d"
    dummy_img = torch.ones(3, 28, 28)
    dummy_img[1, :, :] = dummy_img[1, :, :] * 2
    dummy_img[2, :, :] = dummy_img[2, :, :] * 3
    return dummy_img


@pytest.fixture
def dummy_labels():
    dummy_lab = torch.tensor([3, 4, 1])
    return dummy_lab


class TestFMNISTDataset:

    # the function arguments of "test_len" are the pytest.fixtures defined before
    def test_len(self, dummy_images, dummy_labels):
        dataset = FMNISTDataset(dummy_images, dummy_labels)
        # the actual "test" is executed using "assert"
        assert len(dataset) == len(dummy_images)

    def test_getitem(self, dummy_images, dummy_labels):
        dataset = FMNISTDataset(dummy_images, dummy_labels)
        data_loader = torch.utils.data.DataLoader(dataset)

        for i, loaded_data in enumerate(data_loader):
            img = dummy_images[i].float().view(-1, 28 * 28) / 255.
            label = dummy_labels[i]
            # test if the "image" is correct
            assert torch.allclose(img, loaded_data[0])

            # test if the "label" is correct
            assert torch.allclose(label, loaded_data[1])
