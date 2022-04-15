import numpy as np 
from .Dataloader import getFiles
import SimpleITK as sitk 

def compute_global_max(image_path,files, quantile):
    all_images = np.zeros((96, 110, 110, 110))
    for count, i in enumerate(files):
        im = sitk.GetArrayFromImage(sitk.ReadImage(image_path+"/" + i))
        all_images[count] = im 
    global_max = np.quantile(all_images, quantile)
    return global_max 

def scale_image(im, global_max):
    im[im > global_max] = global_max 
    im = im * 255.0 / global_max 
    return im  


def save_all_scaled_images(image_path,quantile):
    # im_ref_idx = 15 

    files = getFiles(image_path)
    global_max = compute_global_max(image_path,files, quantile)

    for count, f in enumerate(files):
        if count == 32: 
            continue 
        # load 
        im = sitk.ReadImage(image_path+"/" + f)
        space = im.GetSpacing()
        orgin = im.GetOrigin()
        im = sitk.GetArrayFromImage(im)

        # scale 
        im_scaled = scale_image(im, global_max)

        # save 
        im_scaled = sitk.GetImageFromArray(im_scaled)
        im_scaled.SetOrigin(orgin)
        im_scaled.SetSpacing(space)
        sitk.WriteImage(im_scaled, "Data_good_Scaled/" + f)





