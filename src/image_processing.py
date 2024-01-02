import numpy as np
from PIL import Image

def image_to_vector(image_path):
    """
    Convert an image to a black and white (grayscale) and store as a vector.
    Args:
    image_path (str): The path to the image file.

    Returns:
    vector (numpy.ndarray): The n-dimensional vector of the image.
    dimensions (tuple): The dimensions of the original image.
    """
    # Load the image
    img = Image.open(image_path)

    # Convert image to grayscale
    bw_img = img.convert('L')

    # Convert the image to a NumPy array and flatten it
    img_array = np.array(bw_img)
    vector = img_array.flatten()

    return vector, img_array.shape

def vector_to_image(vector, dimensions):
    """
    Convert an n-dimensional vector back to an image.
    Args:
    vector (numpy.ndarray): The n-dimensional vector of the image.
    dimensions (tuple): The dimensions of the original image.

    Returns:
    Image: The reconstructed PIL Image.
    """
    # Reshape the vector to its original dimensions
    img_array = vector.reshape(dimensions)

    # Convert the NumPy array back to a PIL Image
    reconstructed_img = Image.fromarray(img_array.astype('uint8'), 'L')

    return reconstructed_img
