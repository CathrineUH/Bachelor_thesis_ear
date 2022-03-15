import numpy as np 
def image_mapping(image, mu = 230):
    """
    function that given an image maps the pixel values to the range [0, 255]
    Input: 
        image: an MRI
        mu: where to center the image 
    Output: 
        mapped image with pixel values [0, 255]
    """
    if mu != 230:
        mu = mu 
    im_max = np.max(image)
    im_mapped = (image - mu)* (255/im_max)
    im_mapped += np.abs(np.min(im_mapped))    
    return im_mapped 