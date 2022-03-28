import numpy as np 
from .Dataloader import getFiles
import SimpleITK as sitk 
from skimage.exposure import match_histograms 

def match_hist_one_image(im_ref, im_src):
    multi = True if im_src.shape[-1] > 1 else False
    return match_histograms(im_src, im_ref, multichannel = multi)


def compute_global_max(files, quantile):
    all_images = np.zeros((96, 110, 110, 110))
    for count, i in enumerate(files):
        im = sitk.GetArrayFromImage(sitk.ReadImage("junk/Data_use_100/" + i))
        all_images[count] = im 
    global_max = np.quantile(all_images, quantile)
    return global_max 

def scale_image(im, global_max):
    im[im > global_max] = global_max 
    im = im * 255.0 / global_max 
    return im  


def save_all_scaled_images(im_ref_idx, quantile):
    # im_ref_idx = 15 

    files = getFiles("junk/Data_use_100")
    im_ref = sitk.GetArrayFromImage(sitk.ReadImage("junk/Data_use_100/" + files[im_ref_idx]))
    global_max = compute_global_max(files, quantile)

    for f in files:
        # load 
        im_src = sitk.ReadImage("junk/Data_use_100/" + f)
        space = im_src.GetSpacing()
        orgin = im_src.GetOrigin()
        im_src = sitk.GetArrayFromImage(im_src)

        # scale 
        im_matched = match_hist_one_image(im_ref, im_src)
        im_scaled = scale_image(im_matched, global_max)

        # save 
        im_scaled = sitk.GetImageFromArray(im_scaled)
        im_scaled.SetOrigin(orgin)
        im_scaled.SetSpacing(space)
        sitk.WriteImage(im_scaled, "Data_Scaled/" + f)





