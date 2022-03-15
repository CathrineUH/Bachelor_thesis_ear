import numpy as np 
from Dataloader import getFiles
import SimpleITK as sitk 
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
    
    im_map = image - mu
    im_max = np.max(im_map)
    im_min = np.min(im_map)
    scaling = 255 / (im_max - im_min)
    im_map = im_map * scaling 
    im_map += np.abs(np.min(im_map))    
    return im_map


def map_all_images(path):
    files = getFiles(path)
    for f in files:
        im = (sitk.ReadImage(path + "/" + f))
        space = im.GetSpacing()
        orgin = im.GetOrigin()
        im = sitk.GetArrayFromImage(im)
        im_map = image_mapping(im)
        im_map = sitk.GetImageFromArray(im_map)
        im_map.SetOrigin(orgin)
        im_map.SetSpacing(space)
        sitk.WriteImage(im_map, "Data_Map/" + f)
