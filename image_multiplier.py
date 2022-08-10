"""
A  file that opens and resizes images from the specified directory. It then performs rotation, translation, and scaling on the images. It then saves the images
into a single file in the current directory. At the end an optional piece of code displays part of the new image set.
"""
import os
from PIL import Image
import numpy as np
from numpy import expand_dims
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


images_path = 'dataset/'
size = (128, 128)

def show_images(images):
    """
    Shows the arrays as images.
    """
    grid = make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()

def img_to_array(img):
    """Turns the image into an numpy array."""
    return np.array(img)

def array_to_img(img):
    """Turns the numpy array into a PIL image."""
    return Image.fromarray(img)

def load_img(path):
    return Image.open(path)

def resize(img, size):
    return img.resize(size)

def rotate(img, angle):
    return img.rotate(angle)

def translate(img, x, y):
    """Uses numpy to translate the image."""
    return np.roll(img, (x, y), axis=(0, 1))

def scale(img, x, y):
    """Uses PIL to zoom in on the PIL image. It zooms to a box centered on the image that keeps getting smaller."""
    width, height = img.size
    return img.resize(size, box=((x, y, width - x, height - y)))

def save(img, path):
    """
    Saves the array as a npy file.
    """
    np.save(path, img)

def multiple_resize(images_path, size):
    """
    Resizes multiple images and turns then into ndarry. Saves all the files as a single npy file.
    """

    training_data = []
    for filename in os.listdir(images_path):
        images = []
        path = os.path.join(images_path, filename)
        img = load_img(path).convert('L')
        img = resize(img, size)
        images.append(img_to_array(img))

        """
        Carries out the following transformations on the image:
        1. Rotate the image by 90 degrees
        2. Translate the image by -100 and 100 pixels
        3. Rotate the image by 45 degrees
        4. Scale the image x times
        Adds the transformed images to the list of images
        """
        for i in range(4):
            img1 = rotate(img, 90 + i * 90)
            images.append(img_to_array(img1))
            for i2 in range(5):
                img2 = translate(img1, i2 * 10 - 100, i2 * 10 - 100)
                images.append(img_to_array(img2))
                for i3 in range(8):
                    img3 = rotate(array_to_img(img2), i3 * 45 - 100)
                    images.append(img_to_array(img3))
                    sub_images = []
                    x = 4
                    for i4 in range(x):
                        # Scales the image, one unit is one pixel of zoom inwards. Const is for retain of graphics due to zoom
                        const = 0.5
                        i4 += 1
                        img4 = scale(img3.copy(), (size[0] / x) * const * i4, (size[1] / x) * const * i4)
                        sub_images.append(img_to_array(img4))
                    images = [*images, *sub_images[:-1]]

        training_data = [*training_data, *images]

    save(training_data, 'dataset_multiplied_ch1.npy')

# Run the actual resizing
multiple_resize(images_path, size)

# Optional Dataset visualisation
X_train = np.load('dataset_multiplied_ch1.npy')
print(len(X_train))
i = 100
for img in X_train:
    # if i % 20 == 0:
    print(img)
    plt.imshow(img)
    plt.show()
    i += 1
